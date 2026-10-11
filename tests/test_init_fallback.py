import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path('pose').resolve()))
import pose_model


class InitFallbackTests(unittest.TestCase):
    def test_full_search_only_runs_when_shortlist_quality_fails(self):
        for accepted in (True, False):
            est = SimpleNamespace(pose_last=np.eye(4))
            with patch('initialization_fast.register_with_shortlist', return_value=np.eye(4)), \
                 patch('initialization_fast.register_with_shared_crops', return_value=np.eye(4)) as full, \
                 patch('initialization_fast.shortlist_quality', return_value={'accepted': accepted}), \
                 patch.object(pose_model, 'render_depth', return_value=np.ones((6, 8))), \
                 patch.object(pose_model.torch.cuda, 'synchronize'), \
                 patch.object(pose_model.torch.cuda, 'memory_allocated', return_value=0):
                pose_model.estimate_pose(est, np.eye(3), np.zeros((6, 8, 3), np.uint8),
                                         np.ones((6, 8)), np.ones((6, 8), bool), fast_init=True)
                self.assertEqual(full.call_count, 0 if accepted else 1)
                self.assertEqual(est.init_stats['fallback'], not accepted)

    def test_kernel_preparation_never_keeps_synthetic_pose_even_on_error(self):
        for fail in (False, True):
            original_pose = object()
            est = SimpleNamespace(pose_last=original_pose, H=480)
            def register(est, **options):
                est.pose_last = 'synthetic'
                est.H = 6
                est.scores = [1]
                if fail:
                    raise RuntimeError('warmup failed')
            with patch('initialization_fast.register_with_shortlist', side_effect=register), \
                 patch.object(pose_model.torch.cuda, 'synchronize'):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'warmup failed'):
                        pose_model.prepare_fast_pose(est, np.eye(3), (6, 8))
                else:
                    pose_model.prepare_fast_pose(est, np.eye(3), (6, 8))
            self.assertIs(est.pose_last, original_pose)
            self.assertEqual(est.H, 480)
            self.assertFalse(hasattr(est, 'scores'))


if __name__ == '__main__':
    unittest.main()
