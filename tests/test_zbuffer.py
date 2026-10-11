"""Verify depth and winner provenance against stable sorting, including ties."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from project_to_rgb import splat_to_rgb, _splat_to_rgb_sorted


class ZBufferTests(unittest.TestCase):
    def compare(self,u,v,z,h,w):
        expected=_splat_to_rgb_sorted(u,v,z,h,w)
        actual=splat_to_rgb(u,v,z,h,w)
        for x,y in zip(expected,actual):
            self.assertEqual(x.dtype,y.dtype)
            np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(np.signbit(expected[0]),np.signbit(actual[0]))
        return actual

    def test_empty(self):
        self.compare(np.array([],int),np.array([],int),np.array([],float),3,4)

    def test_ties_keep_first_source(self):
        b,a=self.compare(np.array([1,1,1,2,2]),np.zeros(5,int),np.array([3.,1.,1.,2.,2.]),2,4)
        self.assertEqual(a[0,1],1)
        self.assertEqual(a[0,2],3)
        self.assertEqual(a[1,1],-1)
        self.assertEqual(b[0,1],1.)

    def test_signed_zero(self):
        self.compare(np.array([0,0,1,1]),np.zeros(4,int),np.array([-0.,0.,0.,-0.]),1,2)

    def test_nonfinite_compatibility(self):
        self.compare(np.array([0,0,1,1,2,2,3,3]),np.zeros(8,int),
                     np.array([np.nan,1.,np.nan,np.nan,np.inf,np.nan,-np.inf,2.]),1,4)

    def test_random_collisions(self):
        rng=np.random.default_rng(412)
        for dtype in (np.float32,np.float64):
            for count in (1,100,10000,307200):
                u=rng.integers(0,640,count)
                v=rng.integers(0,480,count)
                z=(rng.integers(1,100,count)/10).astype(dtype)
                self.compare(u,v,z,480,640)

    def test_all_points_same_pixel(self):
        rng=np.random.default_rng(18)
        z=rng.integers(1,20,10000).astype(float)
        _,arg=self.compare(np.zeros(len(z),int),np.zeros(len(z),int),z,1,1)
        self.assertEqual(arg[0,0],np.argmin(z))


if __name__=='__main__':
    unittest.main()
