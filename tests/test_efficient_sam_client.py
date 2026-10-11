import io
import queue
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
sys.path.insert(0,str(Path('pose').resolve()))
from efficient_sam_client import EfficientSAMClient
from fs_transport import read_packet
import sam_mask


class EfficientClientTests(unittest.TestCase):
    def client(self, header, payload=b''):
        client=EfficientSAMClient.__new__(EfficientSAMClient)
        client.closed=False; client.request_id=0; client.timeout=.1
        client.responses=queue.Queue(); client.responses.put((header,payload))
        client.process=SimpleNamespace(stdin=io.BytesIO())
        return client

    def test_mask_and_rgb_roundtrip(self):
        rgb=np.arange(18,dtype=np.uint8).reshape(2,3,3)
        mask=np.array([[0,1,0],[1,1,0]],np.uint8)
        c=self.client(dict(id=1,detected=True,shape=[2,3],box=[0,0,2,2],score=.25),mask.tobytes())
        result,box,score=c.segment(rgb,'computer mouse',.1)
        np.testing.assert_array_equal(result,mask.astype(bool))
        self.assertEqual(score,.25)
        c.process.stdin.seek(0)
        header,payload=read_packet(c.process.stdin)
        self.assertEqual(header['text'],'computer mouse')
        self.assertEqual(header['threshold'],.1)
        self.assertEqual(payload,rgb.tobytes())

    def test_missing_detection_and_bad_responses(self):
        rgb=np.zeros((2,3,3),np.uint8)
        self.assertEqual(self.client(dict(id=1,detected=False)).segment(rgb,'mouse',.1),(None,None,None))
        for header,error in [(dict(id=2),RuntimeError),
                             (dict(id=1,error='model failed'),RuntimeError),
                             (dict(id=1,detected=True,shape=[2,3]),ValueError)]:
            with self.assertRaises(error):
                self.client(header).segment(rgb,'mouse',.1)

    def test_backend_defaults_and_explicit_threshold_are_respected(self):
        fake=Mock()
        fake.segment.return_value=(None,None,None)
        with patch.object(sam_mask,'_BACKEND','sam3'), patch.object(sam_mask,'_THRESHOLD',.5), \
             patch.object(sam_mask,'_EFFICIENT',fake), patch.object(sam_mask,'unload_sam') as unload:
            self.assertEqual(sam_mask.configure_sam('efficientsam3'),.1)
            unload.assert_called_once()
            rgb=np.zeros((2,3,3),np.uint8)
            sam_mask.sam_mask(rgb,'mouse')
            self.assertEqual(fake.segment.call_args.args[-1],.1)
            sam_mask.sam_mask(rgb,'mouse',thr=.2)
            self.assertEqual(fake.segment.call_args.args[-1],.2)
            self.assertEqual(sam_mask.configure_sam('sam3'),.5)
            with self.assertRaises(ValueError):
                sam_mask.configure_sam('efficientsam3',0)


if __name__=='__main__':
    unittest.main()
