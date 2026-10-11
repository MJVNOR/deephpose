"""Verify sharing a CAD depth render preserves overlay pixels and fit metrics."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np

# Load the pure consumers without importing FoundationPose/CUDA in this unit test.
source=ast.parse(Path('pose/pose_model.py').read_text(encoding='utf-8'))
nodes=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('draw_overlay','fit_metrics')]
namespace={'np':np,'draw_posed_3d_box':lambda K,img,**kw:img,'draw_xyz_axis':lambda img,**kw:img}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'pose/pose_model.py','exec'),namespace)


class OverlayTests(unittest.TestCase):
    def test_oriented_display_box_and_axes_share_transform(self):
        pose=np.eye(4);pose[:3,3]=[.1,.2,.3]
        transform=np.eye(4)
        transform[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]]
        transform[:3,3]=[.01,.02,.03]
        bounds=np.array([[-1,-2,-3],[3,4,5]],float)
        est=SimpleNamespace(display_box=SimpleNamespace(to_object=transform,bounds=bounds))
        box_poses=[]
        def capture_box(K,img,**kw):
            box_poses.append(kw['ob_in_cam'].copy())
            return img
        draw_box=Mock(side_effect=capture_box)
        draw_axes=Mock(side_effect=lambda img,**kw:img)
        old_box,old_axes=namespace['draw_posed_3d_box'],namespace['draw_xyz_axis']
        try:
            namespace['draw_posed_3d_box'],namespace['draw_xyz_axis']=draw_box,draw_axes
            namespace['draw_overlay'](np.zeros((2,2,3),np.uint8),np.eye(3),pose,est,np.zeros((2,3)),
                                      depth_render=np.zeros((2,2)))
            expected=pose@transform
            np.testing.assert_allclose(box_poses[0],expected)
            expected[:3,3]+=expected[:3,:3]@bounds.mean(axis=0)
            np.testing.assert_allclose(draw_axes.call_args.kwargs['ob_in_cam'],expected)
            np.testing.assert_array_equal(draw_box.call_args.kwargs['bbox'],bounds)
            np.testing.assert_array_equal(pose[:3,3],[.1,.2,.3])
        finally:
            namespace['draw_posed_3d_box'],namespace['draw_xyz_axis']=old_box,old_axes

    def test_one_render_preserves_output(self):
        rendered=np.array([[1,0],[1,1]],np.float32)
        render=Mock(return_value=rendered)
        namespace['render_depth']=render
        rgb=np.full((2,2,3),100,np.uint8)
        K,pose=np.eye(3),np.eye(4)
        bbox=np.array([[0,0,0],[1,1,1]])
        est=object()
        expected,expected_sil=namespace['draw_overlay'](rgb,K,pose,est,bbox)
        expected_fit=namespace['fit_metrics'](rendered,K,pose,1,est=est)
        self.assertEqual(render.call_count,2)
        render.reset_mock()
        shared=render(est,K,2,2)
        actual,actual_sil=namespace['draw_overlay'](rgb,K,pose,est,bbox,depth_render=shared)
        actual_fit=namespace['fit_metrics'](rendered,K,pose,1,est=est,depth_render=shared)
        self.assertEqual(render.call_count,1)
        np.testing.assert_array_equal(expected,actual)
        np.testing.assert_array_equal(expected_sil,actual_sil)
        self.assertEqual(expected_fit,actual_fit)
        np.testing.assert_array_equal(rgb,np.full((2,2,3),100,np.uint8))


if __name__=='__main__':
    unittest.main()
