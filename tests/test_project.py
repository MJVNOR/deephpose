"""Checks geométricos del paso 1. Sin pytest: `uv run python tests/test_project.py`."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
from project_to_rgb import project_ir_to_rgb, rectify_color, splat_to_rgb  # noqa: E402


def test_zbuffer_gana_cercano():
    buf, arg = splat_to_rgb(np.array([5, 5]), np.array([3, 3]), np.array([2.0, 1.0]), 10, 10)
    assert buf[3, 5] == 1.0, f"queda {buf[3, 5]}, debía el mínimo"
    assert arg[3, 5] == 1, "índice ganador debe ser el punto cercano"
    assert arg.sum() == 1 - (100 - 1), "un solo píxel tocado"


def test_extrinsics_coincide_sdk():
    rs = __import__("pyrealsense2")
    rng = np.random.default_rng(7)
    ang = np.deg2rad(10)
    Rz = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    t = np.array([0.015, -0.002, 0.001])
    rot_colmajor = Rz.reshape(9, order="F").tolist()  # como lo entrega RealSense
    e = rs.extrinsics()
    e.rotation = list(rot_colmajor)
    e.translation = list(t)
    for p in rng.uniform(-0.5, 0.5, (20, 3)) + [0, 0, 1.5]:
        sdk = np.array(rs.rs2_transform_point_to_point(e, list(p)))
        ours = Rz @ p + t  # reshape order="F" interno
        assert np.allclose(sdk, ours, atol=1e-6), f"{sdk} vs {ours}"
    row = np.array(rot_colmajor).reshape(3, 3)  # el bug viejo: debe discrepar
    assert not np.allclose(row, Rz), "el reshape row-major no debe coincidir"


def test_escalado_y_geometria_salida():
    K_ir = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1]], dtype=np.float32)
    depth = np.full((3, 4), 2.0, dtype=np.float32)
    valid = np.ones_like(depth, bool)
    color = np.zeros((5, 6, 3), np.uint8)
    cal = {"fx": 600.0, "fy": 600.0, "ppx": 3.0, "ppy": 2.5, "coeffs": [0, 0, 0, 0, 0]}
    buf, hit, K_rect, rect, _ = project_ir_to_rgb(
        depth, valid, K_ir, np.eye(3).reshape(9).tolist(), [0, 0, 0], color, cal)
    assert buf.shape == (5, 6) and hit.shape == (5, 6), "salida a geometría RGB"
    assert rect.shape == (5, 6, 3) and K_rect.shape == (3, 3)
    assert hit.dtype == bool and buf.dtype == np.float32
    assert np.allclose(K_rect, [[600, 0, 3], [0, 600, 2.5], [0, 0, 1]])
    # rectificación identidad con coeffs=0
    same, K2, flag = rectify_color(color, cal)
    assert not flag and np.allclose(K2, K_rect) and same is color
    # escena vacía: sin puntos, sin fallar
    buf0, hit0, _, _, _ = project_ir_to_rgb(
        np.full((3, 4), np.inf, np.float32), np.zeros((3, 4), bool),
        K_ir, np.eye(3).reshape(9).tolist(), [0, 0, 0], color, cal)
    assert not hit0.any() and np.all(np.isinf(buf0)), "vacío = todo inf, nada válido"


if __name__ == "__main__":
    test_zbuffer_gana_cercano()
    test_extrinsics_coincide_sdk()
    test_escalado_y_geometria_salida()
    print("OK tests/test_project.py: 3 checks")
