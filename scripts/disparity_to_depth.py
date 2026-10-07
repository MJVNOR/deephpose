"""Disparidad (px) -> profundidad métrica + nube en frame IR-izq. Reutilizable por project_to_rgb.

Supuestos verificados en D415 640x480: IR rectificadas (coeffs=0), cx IR1==IR2 -> Z=fx*B/d.
Si cx difirieran, habría que corregir según la convención de disparidad del modelo.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_capture_depth(disp_path, calibration_path, scale=1.0, z_min=0.2, z_max=10.0):
    """Devuelve (depth_m, valid, K_net, baseline_m). depth_m en resolución de red."""
    disp = np.load(disp_path).astype(np.float64)
    cal = json.loads(Path(calibration_path).read_text())
    fx = cal["ir1"]["fx"] * scale
    fy = cal["ir1"]["fy"] * scale
    K = np.array([[fx, 0, cal["ir1"]["ppx"] * scale],
                  [0, fy, cal["ir1"]["ppy"] * scale], [0, 0, 1]])
    B = abs(float(cal["baseline_m"]))  # magnitud; el signo es convención ir1->ir2
    if B <= 0:
        sys.exit("error: baseline no positivo en calibration.json")
    depth = np.full_like(disp, np.inf)
    ok = disp > 0
    depth[ok] = fx * B / disp[ok]
    valid = ok & (depth >= z_min) & (depth <= z_max)
    depth[~valid] = np.inf
    return depth.astype(np.float32), valid, K.astype(np.float32), B


def depth_to_xyz(depth, K):
    """Nube (H,W,3) en frame IR-izq. Nunca usa intrínsecos RGB."""
    ys, xs = np.mgrid[0:depth.shape[0], 0:depth.shape[1]].astype(np.float32)
    z = depth
    return np.stack([(xs - K[0, 2]) * z / K[0, 0],
                     (ys - K[1, 2]) * z / K[1, 1], z], axis=-1)


def main():
    from config import load
    C = load()["depth"]
    p = argparse.ArgumentParser()
    p.add_argument("--disp", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--z-min", type=float, default=C["z_min"])
    p.add_argument("--z-max", type=float, default=C["z_max"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    depth, valid, K, B = load_capture_depth(a.disp, a.calibration, a.scale, a.z_min, a.z_max)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "depth_fs.npy", depth)
    v = depth[np.isfinite(depth)]
    print(f"OK {out} B={B:.4f}m validos={100*len(v)/depth.size:.1f}% mediana={np.median(v):.2f}m")


if __name__ == "__main__":
    main()
