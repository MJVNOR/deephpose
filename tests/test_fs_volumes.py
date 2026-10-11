import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fs_volumes import build_concat_volume, build_gwc_volume


def reference(left, right, disparities, groups):
    b, c, h, w = left.shape
    gwc = left.new_zeros(b, groups, disparities, h, w)
    concat = left.new_zeros(b, 2*c, disparities, h, w)
    for d in range(disparities):
        concat[:, :c, d] = left
        if d >= w:
            continue
        l = left[..., d:].reshape(b, groups, c//groups, h, w-d).float()
        r = right[..., :w-d].reshape(b, groups, c//groups, h, w-d).float()
        gwc[:, :, d, :, d:] = (F.normalize(l, dim=2)*F.normalize(r, dim=2)).sum(2)
        concat[:, c:, d, :, d:] = right[..., :w-d]
    return gwc, concat


class VolumeTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for production shapes')
    def test_production_feature_shapes(self):
        for height, width in ((48, 64), (64, 80), (128, 160)):
            with self.subTest(height=height, width=width):
                torch.manual_seed(17)
                left = torch.randn(1, 128, height, width, device='cuda', dtype=torch.float16)
                right = torch.randn_like(left)
                gwc, _ = reference(left, right, 48, 8)
                torch.testing.assert_close(build_gwc_volume(left,right,48,8),gwc,rtol=0,atol=0)
                # Concatenation uses the separately reduced 12-channel features.
                left, right = left[:, :12], right[:, :12]
                _, concat = reference(left,right,48,3)
                torch.testing.assert_close(build_concat_volume(left,right,48),concat,rtol=0,atol=0)

    def test_boundaries_dtypes_and_noncontiguous_inputs(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                for disparities in (0, 1, 4, 10):
                    with self.subTest(device=device, dtype=dtype, disparities=disparities):
                        torch.manual_seed(7)
                        left = torch.randn(2, 12, 3, 14, device=device, dtype=dtype)[..., ::2]
                        right = torch.randn_like(left)
                        left[:, :, 0, 0] = 0  # normalization epsilon, empty disparities, borders
                        gwc, concat = reference(left, right, disparities, 3)
                        torch.testing.assert_close(build_gwc_volume(left,right,disparities,3),gwc,rtol=0,atol=0)
                        torch.testing.assert_close(build_concat_volume(left,right,disparities),concat,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
