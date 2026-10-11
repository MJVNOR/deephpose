import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import kornia
from kornia.geometry.transform import imgwarp
sys.path.insert(0,str(Path('pose').resolve()))
from tracking_fast import CropWarpCache, without_cache_flush


def make_crop_data_batch(value):
    return value


class FakePredictor:
    def predict(self, value):
        torch.cuda.empty_cache()
        result=value+1
        if value<0:
            torch.cuda.empty_cache()
        return result


class ChangedPredictor:
    def predict(self,value):
        torch.cuda.empty_cache()
        return value


class TrackingFastTests(unittest.TestCase):
    def test_adaptation_preserves_original_and_only_removes_flushes(self):
        instance=FakePredictor()
        method=instance.predict.__func__
        adapted=without_cache_flush(instance.predict)
        with patch.object(torch.cuda,'empty_cache') as flush:
            self.assertEqual(instance.predict(-2),-1)
            self.assertEqual(flush.call_count,2)
            flush.reset_mock()
            self.assertEqual(adapted(-2),-1)
            flush.assert_not_called()
            self.assertIs(instance.predict.__func__,method)

    def test_unexpected_upstream_structure_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'Expected two'):
            without_cache_flush(ChangedPredictor().predict)

    def test_shared_grid_preserves_interpolation_and_resets(self):
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            cache=CropWarpCache()
            torch.manual_seed(8)
            rgb=torch.rand(1,3,24,32,device=device)
            xyz=torch.rand_like(rgb)
            M=torch.tensor([[[1.2,.02,-2.],[.01,1.1,3.],[.0001,0.,1.]]],device=device)
            expected_rgb=imgwarp.warp_perspective(rgb,M,(16,20),align_corners=False)
            expected_xyz=imgwarp.warp_perspective(xyz,M,(16,20),mode='nearest',align_corners=False)
            original=imgwarp.normalize_homography
            with patch.object(imgwarp,'normalize_homography',wraps=original) as normalize:
                actual_rgb=cache.warp(rgb,M,(16,20),align_corners=False)
                actual_xyz=cache.warp(xyz,M,(16,20),mode='nearest',align_corners=False)
                self.assertEqual(normalize.call_count,1)
                torch.testing.assert_close(actual_rgb,expected_rgb,rtol=0,atol=0)
                torch.testing.assert_close(actual_xyz,expected_xyz,rtol=0,atol=0)
                cache.reset();M[0,0,2]+=1
                cache.warp(rgb,M,(16,20),align_corners=False)
                self.assertEqual(normalize.call_count,2)
            cache.reset()
            self.assertIsNone(cache.grid)
            self.assertIsNone(cache.matrix)

    def test_new_geometry_cannot_reuse_previous_grid(self):
        cache=CropWarpCache()
        src=torch.arange(60,dtype=torch.float32).reshape(1,1,6,10)
        M=torch.eye(3)[None]
        cache.warp(src,M,(6,10))
        changed=M.clone();changed[0,0,2]=2
        for matrix,size in [(changed,(6,10)),(changed,(8,12)),(M,(8,12))]:
            torch.testing.assert_close(cache.warp(src,matrix,size),imgwarp.warp_perspective(src,matrix,size),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
