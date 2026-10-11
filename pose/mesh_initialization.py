"""Bound memory for the upstream sampled mesh diameter calculation."""
import types
import numpy as np


def sampled_diameter(model_pts=None, mesh=None, n_sample=1000):
    if mesh is not None:
        # Preserve the upstream alternate branch, unused by reset_object.
        from Utils import compute_mesh_diameter
        return compute_mesh_diameter(model_pts=model_pts, mesh=mesh, n_sample=n_sample)
    pts = model_pts
    if n_sample is not None:
        ids = np.random.choice(len(model_pts), size=min(n_sample, len(model_pts)), replace=False)
        pts = model_pts[ids]
    if len(pts) == 0:
        raise ValueError('Cannot compute the diameter of an empty mesh')
    diameter = 0.0
    for start in range(0, len(pts), 128):
        distances = np.linalg.norm(pts[None] - pts[start:start + 128, None], axis=-1)
        diameter = np.maximum(diameter, distances.max())
    return diameter


def bounded_reset_object(original):
    namespace = original.__globals__.copy()
    namespace['compute_mesh_diameter'] = sampled_diameter
    adapted = types.FunctionType(original.__code__, namespace, original.__name__,
                                 original.__defaults__, original.__closure__)
    adapted.__kwdefaults__ = original.__kwdefaults__
    return adapted
