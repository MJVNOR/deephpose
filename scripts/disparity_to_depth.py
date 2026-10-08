"""Disparidad (px) -> profundidad métrica + nube en frame IR-izq. Reutilizable por project_to_rgb.

Fórmula única del proyecto (también la usa live.py): Z = fx·|B|/d.
Supuestos D415 verificados y exigidos: IR rectificadas (coeffs≈0) y
ppx IR1≈IR2; si no, se aborta en vez de devolver números silenciosamente mal.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def check_stereo_cal(cal):
    """Valida calibración estéreo. Devuelve (fx0, baseline_m) o aborta."""
    for k in ("ir1", "ir2"):
        c = cal.get(k, {})
        for f in ("fx", "fy", "ppx", "ppy", "width", "height"):
            if not np.isfinite(float(c.get(f, float("nan")))):
                sys.exit(f"error: calibración no finita en {k}.{f}")
        if not np.allclose(c.get("coeffs", [0] * 5), 0, atol=1e-6):
            sys.exit(f"error: {k} no rectificada (coeffs≠0); Z=fx·B/d no vale")
    if abs(float(cal["ir1"]["ppx"]) - float(cal["ir2"]["ppx"])) > 1.0:
        sys.exit("error: ppx IR1≠IR2 (>1px); la convención de disparidad del modelo no aplica")
    B = float(cal.get("baseline_m", 0.0))
    if not np.isfinite(B) or B == 0:
        sys.exit("error: baseline no válido en calibration.json")
    return float(cal["ir1"]["fx"]), B


def disp_to_depth(disp, fx, baseline_m, z_min=0.2, z_max=10.0):
    """Pura (sin ficheros): disparidad px -> (depth f32 con inf, valid bool)."""
    d = np.asanyarray(disp, dtype=np.float64)
    depth = np.full(d.shape, np.inf)
    ok = np.isfinite(d) & (d > 0)  # ponytail: NaN/+inf nunca son medida
    if np.isfinite(fx) and np.isfinite(baseline_m) and baseline_m != 0:
        depth[ok] = fx * abs(baseline_m) / d[ok]
    valid = ok & np.isfinite(depth) & (depth >= z_min) & (depth <= z_max)
    depth[~valid] = np.inf
    return depth.astype(np.float32), valid


def load_capture_depth(disp_path, calibration_path, scale=1.0, z_min=0.2, z_max=10.0):
    """Devuelve (depth_m, valid, K_net, baseline_m). depth_m en resolución de red."""
    disp = np.load(disp_path)
    cal = json.loads(Path(calibration_path).read_text())
    fx0, B = check_stereo_cal(cal)
    W0, H0 = int(cal["ir1"]["width"]), int(cal["ir1"]["height"])
    sx, sy = disp.shape[1] / W0, disp.shape[0] / H0  # escala efectiva real, no la pedida
    if abs(sx - sy) > 1e-3:
        sys.exit(f"error: disparidad {disp.shape} no proporcional a calibración {(H0, W0)}")
    if abs(sx - scale) / max(scale, 1e-9) > 0.02:
        print(f"aviso: escala efectiva {sx:.3f} ≠ pedida {scale} (uso la real)", file=sys.stderr)
    fx, fy = fx0 * sx, float(cal["ir1"]["fy"]) * sy
    K = np.array([[fx, 0, float(cal["ir1"]["ppx"]) * sx],
                  [0, fy, float(cal["ir1"]["ppy"]) * sy], [0, 0, 1]])
    depth, valid = disp_to_depth(disp, fx, B, z_min, z_max)
    return depth, valid, K.astype(np.float32), abs(B)


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
    med = f"{np.median(v):.2f}m" if len(v) else "sin-puntos"
    print(f"OK {out} B={B:.4f}m validos={100*len(v)/depth.size:.1f}% mediana={med}")


if __name__ == "__main__":
    main()
