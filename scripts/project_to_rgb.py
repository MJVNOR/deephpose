"""Proyecta la profundidad FS (frame IR-izq) sobre la imagen RGB.

Salidas: depth_on_rgb.npy (misma geometría que color.png), overlay.png y
cloud_color.ply (puntos en frame IR-izq con color muestreado). Usa z-buffer.

Uso:
  uv run python scripts/project_to_rgb.py --fs data/processed/<captura>_fs --capture data/raw/<captura>
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from disparity_to_depth import depth_to_xyz, load_capture_depth

sys.path.insert(0, "scripts")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    fs, cap = Path(a.fs), Path(a.capture)
    meta = json.loads((fs / "meta.json").read_text())
    cal = json.loads((cap / "calibration.json").read_text())
    if "color" not in cal or "extrinsics_ir1_to_color" not in cal:
        sys.exit("error: la captura no tiene RGB (recaptura con el capture.py actual)")

    depth, valid, K_ir, _ = load_capture_depth(fs / "disp.npy", cap / "calibration.json", meta["scale"])
    color = cv2.cvtColor(cv2.imread(str(cap / "color.png")), cv2.COLOR_BGR2RGB)
    Hc, Wc = color.shape[:2]
    Kc = np.array([[cal["color"]["fx"], 0, cal["color"]["ppx"]],
                   [0, cal["color"]["fy"], cal["color"]["ppy"]], [0, 0, 1]], dtype=np.float64)
    R = np.array(cal["extrinsics_ir1_to_color"]["rotation"], dtype=np.float64).reshape(3, 3)
    t = np.array(cal["extrinsics_ir1_to_color"]["translation"], dtype=np.float64)

    xyz = depth_to_xyz(depth, K_ir.astype(np.float64))
    pts = xyz[valid]  # (N,3) en IR-izq
    pc = (R @ pts.T).T + t  # a frame RGB
    zc = pc[:, 2]
    front = zc > 0
    pts, pc, zc = pts[front], pc[front], zc[front]
    u = (Kc[0, 0] * pc[:, 0] / zc + Kc[0, 2]).round().astype(int)
    v = (Kc[1, 1] * pc[:, 1] / zc + Kc[1, 2]).round().astype(int)
    inside = (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
    pts, u, v, zc = pts[inside], u[inside], v[inside], zc[inside]

    buf = np.full((Hc, Wc), np.inf)  # z-buffer a resolución RGB
    arg = np.full((Hc, Wc), -1)
    order = np.argsort(zc)
    buf[v[order], u[order]] = zc[order]  # el más cercano gana (ordenado)
    arg[v[order], u[order]] = order
    hit = arg >= 0
    cols = np.zeros((len(pts), 3), dtype=np.uint8)
    # color de cada punto 3D: el del píxel RGB donde cae (solo ganadores del z-buffer)
    win = np.full(len(pts), False)
    win[arg[hit]] = True
    cols[win] = color[v[win], u[win]]

    out = Path(a.out) if a.out else fs.parent / f"{cap.name}_rgb"
    out.mkdir(parents=True, exist_ok=True)  # derivado determinista: reescribir es seguro
    np.save(out / "depth_on_rgb.npy", buf.astype(np.float32))
    v = buf[hit]
    lo, hi = np.percentile(v, [2, 98])  # rango de la escena, no 0.2-10 fijos
    norm = np.clip((buf - lo) / max(hi - lo, 1e-6), 0, 1)
    dm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    dm[~hit] = 0
    shown = cv2.inpaint(dm, (~hit).astype(np.uint8), 3, cv2.INPAINT_TELEA)  # solo display; el .npy conserva inf
    blend = (0.5 * color + 0.5 * shown).astype(np.uint8)
    blend[~hit] = color[~hit]
    side = np.concatenate([color, shown, blend], axis=1)  # color | profundidad | mezcla
    cv2.imwrite(str(out / "overlay.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[win])
    pcd.colors = o3d.utility.Vector3dVector(cols[win].astype(float) / 255)
    o3d.io.write_point_cloud(str(out / "cloud_color.ply"), pcd)
    print(f"OK {out} puntos={win.sum()}/{len(pts)} en-RGB={100*hit.mean():.1f}% zmed={np.median(buf[hit]):.2f}m")


if __name__ == "__main__":
    main()
