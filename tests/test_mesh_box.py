import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pose'))
from mesh_box import base_aligned_box


class MeshBoxTests(unittest.TestCase):
    def test_rotated_flat_box_encloses_mesh_without_mutating_it(self):
        mesh = trimesh.creation.box(extents=(.08, .12, .02))
        rotation = trimesh.transformations.rotation_matrix(np.deg2rad(18), [1, 0, 0])
        rotation[:3, 3] = [.1, -.2, .3]
        mesh.apply_transform(rotation)
        vertices = mesh.vertices.copy()
        box = base_aligned_box(mesh)
        self.assertAlmostEqual(abs(np.dot(box.to_object[:3, 2], rotation[:3, 2])), 1.)
        local = (vertices - box.to_object[:3, 3]) @ box.to_object[:3, :3]
        self.assertTrue((local >= box.bounds[0] - 1e-10).all())
        self.assertTrue((local <= box.bounds[1] + 1e-10).all())
        np.testing.assert_allclose(local @ box.to_object[:3, :3].T + box.to_object[:3, 3], vertices)
        self.assertAlmostEqual(np.linalg.det(box.to_object[:3, :3]), 1.)
        np.testing.assert_array_equal(vertices, mesh.vertices)

    def test_curved_mesh_retains_original_bounds(self):
        mesh = trimesh.creation.icosphere(subdivisions=2)
        box = base_aligned_box(mesh)
        np.testing.assert_array_equal(box.to_object, np.eye(4))
        np.testing.assert_array_equal(box.bounds, mesh.bounds)


if __name__ == '__main__':
    unittest.main()
