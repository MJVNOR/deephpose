"""Proyecta la profundidad FS (frame IR-izq) sobre la imagen RGB.

Geometría de salida para FoundationPose: RGB rectificado + profundidad Z en
ese mismo sistema + máscara de validez + intrínsecos rectificados.

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

# Modelos de distorsión que OpenCV describe con el vector clásico (k1,k2,p1,p2,k3[,k4,k5,k6]).
_OPENCV_COMPATIBLE = {"none", "brown_conrady", "inverse_brown_conrady",
                      "modified_brown_conrady", "distortion.brown_conrady",
                      "distortion.inverse_brown_conrady", "distortion.modified_brown_conrady",
                      "distortion.none"}


def rectify_color(color_rgb, cal_color):
    """RGB -> (rectificada, K_rect, fue_rectificada). Sin distorsión = identidad."""
    H, W = color_rgb.shape[:2]
    K = np.array([[cal_color["fx"], 0, cal_color["ppx"]],
                  [0, cal_color["fy"], cal_color["ppy"]], [0, 0, 1]], dtype=np.float64)
    coeffs = list(cal_color.get("coeffs", [0.0] * 5))
    model = cal_color.get("model", None)
    if model is None:  # capturas viejas sin "model": solo deducible si coeffs=0
        model = "none" if np.allclose(coeffs, 0) else "brown_conrady"
        print(f"aviso: calibration.json sin 'model', asumo {model}", file=sys.stderr)
    dist = np.array(coeffs, dtype=np.float64).ravel()
    if dist.size == 0 or np.allclose(dist, 0) or str(model) == "none":
        return color_rgb, K, False
    if str(model) not in _OPENCV_COMPATIBLE:
        sys.exit(f"error: distorsión '{model}' no compatible con OpenCV (fisheye); "
                 "recalibra o añade su modelo antes de proyectar")
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (W, H), 0)
    m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (W, H), cv2.CV_16SC2)
    return cv2.remap(color_rgb, m1, m2, cv2.INTER_LINEAR), newK, True


def splat_to_rgb(u, v, zc, H, W):
    """Z-buffer explícito min-Z. Devuelve (buf inf=H×W, arg=-1 o índice ganador). Vacío-seguro."""
    buf = np.full((H, W), np.inf)
    arg = np.full((H, W), -1)
    if len(zc) == 0:
        return buf, arg
    order = np.argsort(zc, kind="stable")  # ascendente: el primero por píxel es el mínimo
    lin = v[order] * W + u[order]
    _, first = np.unique(lin, return_index=True)  # primera aparición = Z mínima
    sel = order[first]
    buf.flat[lin[first]] = zc[sel]
    arg.flat[lin[first]] = sel
    return buf, arg


def project_ir_to_rgb(depth, valid, K_ir, rotation, translation, color_rgb, cal_color):
    """Núcleo reutilizable estático/vivo. rotation en column-major RealSense.

    Devuelve (depth_on_rgb f32 con inf, mask_valid bool, K_rect 3x3,
    color_rect RGB uint8, detail{u,v,zc,pts,arg}).
    """
    color_rect, K_rect, _ = rectify_color(color_rgb, cal_color)
    Hc, Wc = color_rect.shape[:2]
    R = np.array(rotation, dtype=np.float64).reshape(3, 3, order="F")  # col-major SDK
    t = np.array(translation, dtype=np.float64)
    xyz = depth_to_xyz(depth, np.asarray(K_ir, dtype=np.float64))
    pts = xyz[valid]  # (N,3) en IR-izq
    if len(pts) == 0:
        return (np.full((Hc, Wc), np.inf, dtype=np.float32),
                np.zeros((Hc, Wc), bool), K_rect, color_rect,
                {"u": np.zeros(0, int), "v": np.zeros(0, int),
                 "zc": np.zeros(0), "pts": pts, "arg": np.full((Hc, Wc), -1)})
    pc = (R @ pts.T).T + t  # a frame RGB
    zc = pc[:, 2]
    front = zc > 0
    pts, pc, zc = pts[front], pc[front], zc[front]
    if len(zc) == 0:
        return (np.full((Hc, Wc), np.inf, dtype=np.float32),
                np.zeros((Hc, Wc), bool), K_rect, color_rect,
                {"u": np.zeros(0, int), "v": np.zeros(0, int),
                 "zc": zc, "pts": pts, "arg": np.full((Hc, Wc), -1)})
    u = (K_rect[0, 0] * pc[:, 0] / zc + K_rect[0, 2]).round().astype(int)
    v = (K_rect[1, 1] * pc[:, 1] / zc + K_rect[1, 2]).round().astype(int)
    inside = (u >= 0) & (u < Wc) & (v >= 0) & (v < Hc)
    pts, u, v, zc = pts[inside], u[inside], v[inside], zc[inside]
    buf, arg = splat_to_rgb(u, v, zc, Hc, Wc)
    return buf.astype(np.float32), arg >= 0, K_rect, color_rect, \
        {"u": u, "v": v, "zc": zc, "pts": pts, "arg": arg}


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
    bgr = cv2.imread(str(cap / "color.png"))
    if bgr is None:
        sys.exit(f"error: no se lee {cap}/color.png")
    color = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ex = cal["extrinsics_ir1_to_color"]
    buf, hit, K_rect, rect, d = project_ir_to_rgb(
        depth, valid, K_ir, ex["rotation"], ex["translation"], color, cal["color"])
    pts, arg = d["pts"], d["arg"]

    cols = np.zeros((len(pts), 3), dtype=np.uint8)
    win = np.full(len(pts), False)  # puntos ganadores del z-buffer
    if hit.any():
        win[arg[hit]] = True
        cols[win] = rect[d["v"][win], d["u"][win]]

    out = Path(a.out) if a.out else fs.parent / f"{cap.name}_rgb"
    out.mkdir(parents=True, exist_ok=True)  # derivado determinista: reescribir es seguro
    np.save(out / "depth_on_rgb.npy", buf)
    cv2.imwrite(str(out / "color_rect.png"), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
    (out / "K_rect.json").write_text(json.dumps({
        "K": K_rect.tolist(), "width": rect.shape[1], "height": rect.shape[0],
        "dist_model": cal["color"].get("model", None), "rectified": bool(
            not np.allclose(K_rect, [[cal["color"]["fx"], 0, cal["color"]["ppx"]],
                                     [0, cal["color"]["fy"], cal["color"]["ppy"]], [0, 0, 1]])),
    }, indent=2))
    vals = buf[hit]
    if len(vals):  # escena con puntos: rango p2-p98; vacía: overlay = color
        lo, hi = np.percentile(vals, [2, 98])
        norm = np.clip((buf - lo) / max(hi - lo, 1e-6), 0, 1)
        dm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        dm[~hit] = 0
        shown = cv2.inpaint(dm, (~hit).astype(np.uint8), 3, cv2.INPAINT_TELEA)  # solo display
        blend = (0.5 * rect + 0.5 * shown).astype(np.uint8)
        blend[~hit] = rect[~hit]
        zmed = f"{np.median(vals):.2f}m"
    else:
        shown = rect.copy()
        blend = rect.copy()
        zmed = "sin-puntos"
    side = np.concatenate([rect, shown, blend], axis=1)  # color | profundidad | mezcla
    cv2.imwrite(str(out / "overlay.png"), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[win] if len(pts) else np.zeros((0, 3)))
    pcd.colors = o3d.utility.Vector3dVector(cols[win].astype(float) / 255 if win.any()
                                            else np.zeros((0, 3)))
    o3d.io.write_point_cloud(str(out / "cloud_color.ply"), pcd)
    print(f"OK {out} puntos={win.sum()}/{len(pts)} en-RGB={100*hit.mean():.1f}% zmed={zmed}")


if __name__ == "__main__":
    main()
