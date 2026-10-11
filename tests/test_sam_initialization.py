import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pose'))
from sam_initialization import run_sam_attempt


class SamInitializationTests(unittest.TestCase):
    def test_valid_detection_preserves_mask_and_score(self):
        mask = np.ones((2, 2), dtype=bool)
        result = run_sam_attempt(None, 'mouse', lambda *_: (mask, [0, 0, 2, 2], .8))
        self.assertIs(result.mask, mask)
        self.assertEqual(result.score, .8)
        self.assertIsNone(result.error)

    def test_no_detection_is_distinct_from_backend_error(self):
        for mask in (None, np.zeros((2, 2), dtype=bool)):
            result = run_sam_attempt(None, 'mouse', lambda *_: (mask, None, None))
            self.assertIsNone(result.mask)
            self.assertIsNone(result.error)

    def test_backend_failure_retains_diagnostic(self):
        def fail(*_):
            raise RuntimeError('CUDA out of memory')
        result = run_sam_attempt(None, 'mouse', fail)
        self.assertIsNone(result.mask)
        self.assertEqual(result.error, 'RuntimeError: CUDA out of memory')


if __name__ == '__main__':
    unittest.main()
