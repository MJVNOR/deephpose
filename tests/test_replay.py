"""Guard deterministic frame selection (no camera/GPU)."""
import sys
import unittest
sys.path.insert(0, 'scripts')
from replay_selection import selected_frame, verify_sequence


class ReplayTests(unittest.TestCase):
    def test_frame_ids_not_arrival_order(self):
        self.assertEqual([n for n in range(1, 101) if selected_frame(n, 60, 15)], [60, 75, 90])

    def test_missing_duplicate_unordered_fail(self):
        for frames in ([], [75, 90], [60, 90], [60, 75, 75, 90], [75, 60, 90]):
            with self.assertRaises(ValueError):
                verify_sequence(frames, 60, 15)

    def test_valid_sequence(self):
        verify_sequence([60, 75, 90], 60, 15)

    def test_invalid_stride(self):
        with self.assertRaises(ValueError):
            selected_frame(60, 60, 0)


if __name__ == '__main__':
    unittest.main()
