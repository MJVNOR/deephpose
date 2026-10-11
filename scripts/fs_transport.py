"""Length-prefixed binary IPC: JSON metadata + raw arrays, no temporary files."""
import json
import queue
import struct
import subprocess
import threading
import tempfile
from pathlib import Path
import numpy as np

MAX_BYTES = 64 * 1024 * 1024


def read_exact(stream, size):
    out = bytearray()
    while len(out) < size:
        block = stream.read(size-len(out))
        if not block:
            raise EOFError('FS worker closed its pipe')
        out.extend(block)
    return bytes(out)


def read_packet(stream):
    header_size, payload_size = struct.unpack('<II', read_exact(stream,8))
    if not 0 < header_size <= 65536 or payload_size > MAX_BYTES:
        raise ValueError('Invalid IPC packet lengths')
    header = json.loads(read_exact(stream,header_size))
    return header,read_exact(stream,payload_size)


def write_packet(stream, header, payload=b''):
    raw = json.dumps(header).encode('utf-8')
    if len(raw)>65536 or len(payload)>MAX_BYTES:
        raise ValueError('IPC packet too large')
    stream.write(struct.pack('<II',len(raw),len(payload)))
    stream.write(raw)
    stream.write(payload)
    stream.flush()


class FSClient:
    """One inference in flight; reader provides bounded waits and EOF propagation."""
    def __init__(self, root, command=None, timeout=180):
        self.timeout = timeout
        self.responses = queue.Queue()
        self.request_id = 0
        self.closed = False
        root = Path(root)
        command = command or [str(root/'.venv/Scripts/python.exe'), '-u', str(root/'scripts/fs_worker.py'), '--binary']
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=root)
        self.reader = threading.Thread(target=self._read,daemon=True)
        self.reader.start()

    def _read(self):
        try:
            while True:
                self.responses.put(read_packet(self.process.stdout))
        except Exception as exc:
            self.responses.put(exc)

    def infer(self,left,right,scale,iters):
        if self.closed:
            raise RuntimeError('FS worker is closed')
        if left.shape != right.shape or left.ndim != 2 or left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError('Expected equally sized uint8 stereo images')
        self.request_id += 1
        write_packet(self.process.stdin, {'command':'infer','id':self.request_id,'shape':list(left.shape),
                     'scale':scale,'iters':iters}, left.tobytes()+right.tobytes())
        try:
            response = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError('FS inference exceeded timeout') from None
        if isinstance(response,Exception):
            raise RuntimeError('FS worker failed') from response
        header,payload = response
        if header.get('id') != self.request_id:
            raise RuntimeError('FS response ID mismatch')
        if header.get('error'):
            raise RuntimeError(header['error'])
        shape = tuple(header['shape'])
        if len(shape)!=2 or any(v<=0 for v in shape) or len(payload)!=int(np.prod(shape))*4:
            raise ValueError('Invalid disparity payload')
        return np.frombuffer(payload,dtype='<f4').reshape(shape).copy(),header['info']

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            write_packet(self.process.stdin,{'command':'quit'})
            self.process.wait(timeout=3)
        except (OSError,ValueError,subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        for stream in (self.process.stdin,self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass  # broken worker pipe may fail while flushing stdin on Windows


class LegacyFSClient:
    """Disk path retained only for A/B profiling, isolated per client instance."""
    def __init__(self,root,timeout=180):
        import cv2
        self.cv2 = cv2
        self.timeout = timeout
        self.closed = False
        root = Path(root)
        self.directory = tempfile.TemporaryDirectory(prefix='fs_ab_',dir=root/'data/processed')
        self.path = Path(self.directory.name)
        self.process = subprocess.Popen([str(root/'.venv/Scripts/python.exe'),'-u',str(root/'scripts/fs_worker.py')],
             stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,cwd=root)
        self.responses = queue.Queue()
        self.reader = threading.Thread(target=self._read,daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            self.responses.put(line)
        self.responses.put(None)

    def infer(self,left,right,scale,iters):
        l,r,d = (self.path/n for n in ['left.png','right.png','disp.npy'])
        self.cv2.imwrite(str(l),left)
        self.cv2.imwrite(str(r),right)
        self.process.stdin.write(f'infer {l} {r} {d} {scale} {iters}\n')
        self.process.stdin.flush()
        try:
            line = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError('Legacy worker exceeded timeout') from None
        parts = line.split() if line else []
        if len(parts)!=3 or parts[0]!='ok':
            raise RuntimeError(f'Legacy worker failed: {line}')
        return np.load(d),{'t_infer_s':float(parts[2])}

    def close(self):
        if self.closed:
            return
        self.closed=True
        try:
            self.process.stdin.write('quit\n')
            self.process.stdin.flush()
            self.process.wait(timeout=3)
        except (OSError,ValueError,subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        for stream in (self.process.stdin,self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        self.directory.cleanup()
