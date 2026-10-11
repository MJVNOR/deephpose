import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pose'))
from display_frame import select_display_frame


class DisplayFrameTests(unittest.TestCase):
    def test_completed_pair_wins_over_newer_raw(self):
        completed=np.full((2,2,3),21,np.uint8)
        raw=np.full((2,2,3),99,np.uint8)
        blank=np.zeros_like(raw)
        self.assertIs(select_display_frame(completed,raw,blank),completed)
        raw[:]=123
        np.testing.assert_array_equal(completed,np.full_like(completed,21))
        self.assertIs(select_display_frame(completed,raw,blank),completed)

    def test_startup_and_reset_show_raw(self):
        raw=np.ones((2,2,3),np.uint8);blank=np.zeros_like(raw)
        self.assertIs(select_display_frame(None,raw,blank),raw)
        self.assertIs(select_display_frame(np.array([]),raw,blank),raw)

    def test_no_capture_uses_blank(self):
        blank=np.zeros((2,2,3),np.uint8)
        self.assertIs(select_display_frame(None,None,blank),blank)


if __name__=='__main__': unittest.main()
