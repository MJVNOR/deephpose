import concurrent.futures
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pose'))
from recovery import RecoverySearch, fit_is_valid
from sam_initialization import SamResult


class Executor:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        future = concurrent.futures.Future()
        self.calls.append((fn, args, future))
        return future


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.pool = Executor()
        self.search = RecoverySearch(3, 1)
        self.bundle = {'rect': np.zeros((2, 2, 3), np.uint8), 'fn': [1]}

    def update(self, valid=False, now=0):
        return self.search.update(valid, self.bundle, now, self.pool, 'mouse')

    def start(self):
        for _ in range(3):
            self.update()
        return self.pool.calls[0][2]

    def test_single_flight_does_not_block_tracking_updates(self):
        future = self.start()
        for i in range(100):
            self.update(now=i)
        self.assertEqual(len(self.pool.calls), 1)
        mask = np.ones((2, 2), bool)
        future.set_result(SamResult(mask=mask))
        bundle, result = self.update(now=101)
        self.assertIs(bundle, self.bundle)
        self.assertIs(result.mask, mask)

    def test_transient_loss_does_not_search(self):
        self.update()
        self.update()
        self.update(True)
        self.update()
        self.assertFalse(self.pool.calls)

    def test_no_detection_and_errors_retry_after_backoff(self):
        future = self.start()
        future.set_result(SamResult(error='worker disconnected'))
        self.update(now=1)
        self.assertEqual(self.search.last_error, 'worker disconnected')
        self.update(now=1.5)
        self.assertEqual(len(self.pool.calls), 1)
        self.update(now=2)
        self.assertEqual(len(self.pool.calls), 2)
        self.pool.calls[-1][2].set_result(SamResult())
        self.update(now=3)
        self.assertIsNone(self.search.future)

    def test_recovered_tracking_invalidates_late_sam(self):
        future = self.start()
        self.update(True)
        self.update(True)
        self.update(False)
        future.set_result(SamResult(mask=np.ones((2, 2), bool)))
        self.assertIsNone(self.update(False))

    def test_manual_reset_drains_old_result(self):
        future = self.start()
        self.search.reset()
        future.set_result(SamResult(mask=np.ones((2, 2), bool)))
        self.assertIsNone(self.update())
        self.assertIsNone(self.search.future)

    def test_missing_depth_is_not_valid_tracking(self):
        for residual in (None, float('nan'), float('inf'), .021):
            self.assertFalse(fit_is_valid({'valid_frac': 1, 'resid_med_m': residual}))
        self.assertTrue(fit_is_valid({'valid_frac': .3, 'resid_med_m': .02}))
        self.assertFalse(fit_is_valid({'valid_frac': .29, 'resid_med_m': .01}))
        self.assertFalse(fit_is_valid({'valid_frac': 1, 'resid_med_m': 0, 'sil_px': 0}))


if __name__ == '__main__':
    unittest.main()
