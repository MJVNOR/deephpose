"""Exercise framing through short reads and a real subprocess, without GPU."""
import io
import struct
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,'scripts')
from fs_transport import read_packet,write_packet,FSClient


class ShortReads(io.BytesIO):
    def read(self,n=-1):
        return super().read(min(n,3))


class TransportTests(unittest.TestCase):
    def test_partial_reads_and_payload(self):
        stream=io.BytesIO()
        write_packet(stream,{'id':7},bytes(range(255)))
        self.assertEqual(read_packet(ShortReads(stream.getvalue())),({'id':7},bytes(range(255))))

    def test_truncated_packet(self):
        with self.assertRaises(EOFError):
            read_packet(io.BytesIO(b'123'))

    def test_oversized_header(self):
        with self.assertRaises(ValueError):
            read_packet(io.BytesIO(struct.pack('<II',999999,0)))

    def test_subprocess_arrays_and_repeated_requests(self):
        client=FSClient(Path.cwd(),command=[sys.executable,str(Path(__file__).resolve()),'--echo'],timeout=5)
        try:
            left=np.arange(48,dtype=np.uint8).reshape(6,8)
            right=np.ones_like(left)
            for _ in range(3):
                actual,info=client.infer(left,right,.5,8)
                np.testing.assert_array_equal(actual,left.astype('float32')+right)
                self.assertEqual(info['t_infer_s'],0)
        finally:
            client.close()
        self.assertIsNotNone(client.process.poll())

    def test_worker_eof_propagates(self):
        client=FSClient(Path.cwd(),command=[sys.executable,str(Path(__file__).resolve()),'--exit'],timeout=5)
        try:
            with self.assertRaises((RuntimeError,BrokenPipeError,OSError)):
                client.infer(np.zeros((2,2),np.uint8),np.zeros((2,2),np.uint8),1,1)
        finally:
            client.close()

    def test_timeout_terminates_worker(self):
        client=FSClient(Path.cwd(),command=[sys.executable,str(Path(__file__).resolve()),'--stall'],timeout=.1)
        with self.assertRaises(TimeoutError):
            client.infer(np.zeros((2,2),np.uint8),np.zeros((2,2),np.uint8),1,1)
        self.assertIsNotNone(client.process.poll())
        self.assertTrue(client.closed)


if __name__=='__main__':
    if '--stall' in sys.argv:
        import time
        read_packet(sys.stdin.buffer)
        time.sleep(10)
        sys.exit(0)
    if '--exit' in sys.argv:
        sys.exit(0)
    if '--echo' in sys.argv:
        while True:
            header,payload=read_packet(sys.stdin.buffer)
            if header['command']=='quit':
                break
            h,w=header['shape']
            array=np.frombuffer(payload,np.uint8).reshape(2,h,w).astype('float32').sum(axis=0)
            write_packet(sys.stdout.buffer,{'id':header['id'],'shape':[h,w],'info':{'t_infer_s':0}},array.astype('<f4').tobytes())
    else:
        unittest.main()
