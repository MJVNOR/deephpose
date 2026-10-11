import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('scripts').resolve()))
sys.path.insert(0,str(Path('pose').resolve()))
from project_to_rgb import project_ir_to_rgb
from overlay_fast import draw_opaque_axes
import ast
import cv2


class FastCPUTests(unittest.TestCase):
    def test_axes_exact_with_overlap_clipping_and_input_preserved(self):
        tree=ast.parse(Path('pose/FoundationPose/Utils.py').read_text(encoding='utf-8'))
        names=('draw_xyz_axis','project_3d_to_2d')
        env={'np':np,'cv2':cv2}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'Utils.py','exec'),env)
        rng=np.random.default_rng(4)
        rgb=rng.integers(0,256,(80,100,3),dtype=np.uint8)
        original=rgb.copy();K=np.array([[100.,0,50],[0,100,40],[0,0,1]])
        for _ in range(50):
            pose=np.eye(4);pose[:3,:3]=np.linalg.qr(rng.normal(size=(3,3)))[0]
            pose[:3,3]=rng.uniform([-.8,-.8,.3],[.8,.8,1.5])
            expected=env['draw_xyz_axis'](rgb,pose,K=K,is_input_rgb=True)
            actual=draw_opaque_axes(rgb.copy(),pose,K)
            np.testing.assert_array_equal(expected,actual)
        np.testing.assert_array_equal(rgb,original)

    def test_projection_sparse_empty_behind_and_distorted(self):
        rng=np.random.default_rng(5)
        for coeffs in ([0]*5,[.1,-.02,0,0,0]):
            for shift in (0,-10,10):
                for density in (0,.5,1):
                    depth=rng.uniform(.2,4,(24,32)).astype(np.float32)
                    valid=rng.random(depth.shape)<density
                    depth[~valid]=np.inf
                    K=np.array([[30.,0,16],[0,30,12],[0,0,1]])
                    rgb=np.zeros((24,32,3),np.uint8)
                    cal=dict(fx=30.,fy=30.,ppx=16.,ppy=12.,coeffs=coeffs,model='brown_conrady')
                    args=(depth,valid,K,np.eye(3).ravel(),[0,0,shift],rgb,cal)
                    with np.errstate(invalid='ignore'):
                        old=project_ir_to_rgb(*args)
                        new=project_ir_to_rgb(*args,return_details=False)
                    for x,y in zip(old[:4],new[:4]):np.testing.assert_array_equal(x,y)
                    self.assertIsNone(new[4])


if __name__=='__main__':unittest.main()
