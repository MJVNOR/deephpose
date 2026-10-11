import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('pose').resolve()))
from mesh_initialization import sampled_diameter


class MeshInitializationTests(unittest.TestCase):
    def test_diameter_and_random_state_exactly_match_full_matrix(self):
        for dtype in (np.float32, np.float64):
            pts = np.random.default_rng(7).normal(size=(387, 3)).astype(dtype)
            for count in (None, 300, 1000):
                np.random.seed(4)
                selected = pts if count is None else pts[np.random.choice(len(pts), min(count, len(pts)), replace=False)]
                expected = np.linalg.norm(selected[None] - selected[:, None], axis=-1).max()
                next_random = np.random.random()
                np.random.seed(4)
                self.assertEqual(sampled_diameter(pts, n_sample=count), expected)
                self.assertEqual(np.random.random(), next_random)


if __name__ == '__main__':
    unittest.main()
