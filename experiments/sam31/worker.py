"""Resident isolated segmentation worker, using the project's binary IPC format."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
wire_out = sys.stdout.buffer
sys.stdout = sys.stderr  # model imports and builders must not write into the binary pipe
sys.path.insert(0, str(ROOT/'scripts'))
from fs_transport import read_packet, write_packet
import numpy as np
import time
import torch
from backend import SAM31Mask


def main():
    model = None
    while True:
        try:
            header, payload = read_packet(sys.stdin.buffer)
        except EOFError:
            return
        if header.get('command') == 'quit':
            return
        try:
            if header.get('command') != 'segment':
                raise ValueError('Unknown segmentation command')
            shape = tuple(header['shape'])
            if len(shape)!=3 or shape[2]!=3 or any(v<=0 for v in shape) or len(payload)!=int(np.prod(shape)):
                raise ValueError('Invalid RGB payload')
            if model is None:
                model = SAM31Mask()
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()
            torch.cuda.synchronize(); start = time.perf_counter()
            mask, box, score = model(rgb, header['text'], thr=float(header['threshold']))
            torch.cuda.synchronize()
            write_packet(wire_out, dict(id=header['id'], detected=mask is not None,
                shape=shape[:2], box=box, score=score, inference_s=time.perf_counter()-start),
                mask.astype(np.uint8).tobytes() if mask is not None else b'')
        except Exception as exc:
            write_packet(wire_out, dict(id=header.get('id'), error=f'{type(exc).__name__}: {exc}'))


if __name__ == '__main__':
    main()
