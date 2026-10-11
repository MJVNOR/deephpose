import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path('pose').resolve()))
from initialization_fast import register_with_shared_crops, register_with_shortlist, spread_rotation_indices, shortlist_quality


class Predictor:
    def predict(self, *, ob_in_cams):
        return len(ob_in_cams)


class InitializationTests(unittest.TestCase):
    def test_shortlist_keeps_iteration_budget_and_restores_on_rank_failure(self):
        for fail in (False, True):
            calls = []
            scores = []
            def refine(**options):
                calls.append((len(options['ob_in_cams']), options['iteration']))
                return torch.as_tensor(options['ob_in_cams']).clone(), None
            def score(**options):
                scores.append(len(options['ob_in_cams']))
                if fail:
                    raise RuntimeError('scorer failed')
                return torch.arange(len(options['ob_in_cams'])), None
            grid = torch.eye(4).repeat(60, 1, 1)
            est = SimpleNamespace(refiner=SimpleNamespace(predict=refine),
                                  scorer=SimpleNamespace(predict=score), rot_grid=grid)
            def register(**kwargs):
                options = dict.fromkeys(('mesh', 'rgb', 'depth', 'K', 'normal_map', 'xyz_map', 'mesh_tensors', 'glctx', 'mesh_diameter'))
                poses, _ = est.refiner.predict(**options, ob_in_cams=est.rot_grid.numpy(), iteration=kwargs['iteration'], get_vis=False)
                est.scorer.predict(ob_in_cams=poses.numpy())
                return poses
            est.register = register
            with patch('initialization_fast.shared_crop_predictor', side_effect=lambda m: m):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'scorer failed'):
                        register_with_shortlist(est, hypotheses=42, finalists=8, iteration=5)
                else:
                    self.assertEqual(len(register_with_shortlist(est, hypotheses=42, finalists=8, iteration=5)), 8)
                    self.assertEqual(calls, [(42, 1), (8, 4)])
                    self.assertEqual(scores, [42, 8])
            self.assertIs(est.rot_grid, grid)
            self.assertIs(est.refiner.predict, refine)
            self.assertIs(est.scorer.predict, score)

    def test_orientation_selection_is_unique_even_for_duplicate_rotations(self):
        rotations = np.tile(np.eye(4), (12, 1, 1))
        indices = spread_rotation_indices(rotations, 8)
        self.assertEqual(len(set(indices)), 8)
        np.testing.assert_array_equal(indices, spread_rotation_indices(rotations, 8))
        self.assertEqual(len(spread_rotation_indices(rotations, 20)), 12)
        with self.assertRaises(ValueError):
            spread_rotation_indices(rotations, 0)

    def test_quality_gate_rejects_invalid_depth_bad_fit_and_wrong_mask(self):
        depth = np.ones((4, 4))
        rendered = depth.copy()
        mask = np.ones((4, 4), bool)
        self.assertTrue(shortlist_quality(depth, rendered, mask)['accepted'])
        self.assertFalse(shortlist_quality(depth*np.nan, rendered, mask)['accepted'])
        self.assertFalse(shortlist_quality(depth+.1, rendered, mask)['accepted'])
        self.assertFalse(shortlist_quality(depth, rendered, ~mask)['accepted'])
        self.assertFalse(shortlist_quality(depth, rendered*0, mask)['accepted'])

    def test_progress_and_restoration_on_success_and_failure(self):
        for reference in (True, False):
            for fail in (False, True):
                with self.subTest(reference=reference, fail=fail):
                    est = SimpleNamespace(refiner=Predictor(), scorer=Predictor())
                    originals = (est.refiner.predict, est.scorer.predict)
                    phases = []
                    def register(**kwargs):
                        self.assertEqual(kwargs, {'iteration': 5})
                        self.assertEqual(est.refiner.predict(ob_in_cams=range(252)), 252)
                        if fail:
                            raise ValueError('registration failed')
                        return est.scorer.predict(ob_in_cams=range(252))
                    est.register = register
                    with patch('initialization_fast.shared_crop_predictor', side_effect=lambda m: m) as adapt:
                        if fail:
                            with self.assertRaisesRegex(ValueError, 'registration failed'):
                                register_with_shared_crops(est, reference=reference, progress=lambda *p: phases.append(p), iteration=5)
                        else:
                            self.assertEqual(register_with_shared_crops(est, reference=reference, progress=lambda *p: phases.append(p), iteration=5), 252)
                        self.assertEqual(adapt.call_count, 0 if reference else 2)
                    self.assertEqual((est.refiner.predict, est.scorer.predict), originals)
                    self.assertEqual(phases, [('refinando', 252)] if fail else [('refinando', 252), ('seleccionando', 252)])


if __name__ == '__main__':
    unittest.main()
