"""EfficientSAM3 runs in Python 3.12 without changing FoundationPose's environment."""
import queue
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from fs_transport import FSClient, write_packet


class EfficientSAMClient(FSClient):
    experiment = 'efficientsam3'
    label = 'EfficientSAM3'

    def __init__(self):
        python = ROOT/f'experiments/{self.experiment}/.venv/Scripts/python.exe'
        if not python.exists():
            raise RuntimeError(f'Instala el entorno con uv sync --project experiments/{self.experiment}')
        super().__init__(ROOT, command=[str(python), '-u', str(ROOT/f'experiments/{self.experiment}/worker.py')])

    def segment(self, rgb, text, threshold):
        if self.closed:
            raise RuntimeError(f'{self.label} worker is closed')
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected HxWx3 uint8 RGB')
        self.request_id += 1
        write_packet(self.process.stdin, dict(command='segment', id=self.request_id,
            shape=rgb.shape, text=text, threshold=threshold), np.ascontiguousarray(rgb).tobytes())
        try:
            response = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError(f'{self.label} exceeded its response timeout') from None
        if isinstance(response, Exception):
            raise RuntimeError(f'{self.label} worker failed') from response
        header, payload = response
        if header.get('id') != self.request_id:
            raise RuntimeError(f'{self.label} response ID mismatch')
        if header.get('error'):
            raise RuntimeError(header['error'])
        if not header['detected']:
            return None, None, None
        if tuple(header['shape']) != rgb.shape[:2] or len(payload) != rgb.shape[0]*rgb.shape[1]:
            raise ValueError(f'Invalid {self.label} mask payload')
        mask = np.frombuffer(payload, dtype=np.uint8).reshape(rgb.shape[:2]).astype(bool)
        return mask, header['box'], header['score']
