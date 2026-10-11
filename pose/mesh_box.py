"""Display coordinates aligned to a mesh's dominant planar surface."""
from dataclasses import dataclass

import numpy as np


@dataclass
class MeshBox:
    bounds: np.ndarray
    to_object: np.ndarray
    planar_area_fraction: float


def base_aligned_box(mesh):
    """Enclose all vertices with an axis normal to the dominant planar facet.

    This changes display coordinates only, never the mesh/estimated pose. For
    curved meshes without a substantial plane retain the original AABB. A
    dominant facet is a geometric convention, not a semantic base detector.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    areas = np.asarray(mesh.facets_area)
    identity = np.eye(4)
    if not len(areas) or mesh.area <= 0:
        return MeshBox(np.asarray(mesh.bounds).copy(), identity, 0.)
    index = int(areas.argmax())
    fraction = float(areas[index] / mesh.area)
    if fraction < .05:
        return MeshBox(np.asarray(mesh.bounds).copy(), identity, fraction)
    face_indices = mesh.facets[index]
    points = vertices[np.unique(np.asarray(mesh.faces)[face_indices])]
    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    z = vh[-1]
    # Point toward the body, away from the support plane.
    if np.dot(z, vertices.mean(axis=0) - origin) < 0:
        z = -z
    # Project the least parallel original axis; deterministic, no arbitrary
    # PCA rotation driven by the asymmetric ergonomic upper shell.
    axis = np.eye(3)[int(np.argmin(np.abs(z)))]
    x = axis - z * np.dot(axis, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.column_stack((x, y, z))
    local = (vertices - origin) @ rotation
    to_object = np.eye(4)
    to_object[:3, :3] = rotation
    to_object[:3, 3] = origin
    return MeshBox(np.stack((local.min(axis=0), local.max(axis=0))), to_object, fraction)
