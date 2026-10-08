"""Pose 6D estática de un mouse. Corre en el env `pose/` (FoundationPose).

Entradas: salida de project_to_rgb (RGB rectificado + Z + K_rect), malla
preparada y máscara manual (blanco=objeto). V1: un objeto.

Uso (desde la raíz del repo):
  uv run --project pose python pose/pose_static.py --rgbdir data/processed/<cap>_rgb --mesh data/mesh/mouse/<m>_prepared.obj --mask mask.png --out data/processed/<cap>_pose
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_model import (draw_overlay, draw_textured_overlay, estimate_pose, fit_metrics,  # noqa: E402
                        load_textured_mesh, make_estimator)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rgbdir", required=True, help="salida de scripts/project_to_rgb.py")
    p.add_argument("--mesh", required=True, help="OBJ preparado (metros, con MTL)")
    p.add_argument("--mask", required=True, help="PNG máscara manual, blanco=objeto")
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=None, help="dir FoundationPose con weights/ (def: FoundationPose/)")
    p.add_argument("--iters-init", type=int, default=5, help="refinamientos en init (paper: 5)")
    return p.parse_args()


def main():
    a = parse_args()
    rd, out = Path(a.rgbdir), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    fp = Path(a.weights) if a.weights else Path(__file__).resolve().parent / "FoundationPose"

    K = np.array(json.loads((rd / "K_rect.json").read_text())["K"], float)
    bgr = cv2.imread(str(rd / "color_rect.png"))
    if bgr is None:
        sys.exit(f"error: sin color_rect.png en {rd} (corre project_to_rgb del paso 1)")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(rd / "depth_on_rgb.npy").astype(np.float32)  # inf=inválido, se conserva
    m = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE)
    if m is None:
        sys.exit(f"error: máscara ilegible {a.mask} (V1 exige máscara manual)")
    if m.shape[:2] != rgb.shape[:2]:
        sys.exit(f"error: máscara {m.shape} ≠ RGB {rgb.shape[:2]}")
    mask = m > 127
    if not mask.any():
        sys.exit("error: máscara vacía")

    mesh = load_textured_mesh(a.mesh)
    est = make_estimator(mesh, fp, debug_dir=out, debug=0)
    pose, t_est, vram = estimate_pose(est, K, rgb, depth, mask, iters=a.iters_init)
    # ponytail: ranking del scorer guardado tal cual; no es probabilidad
    sc = getattr(est, "scores", None)
    scores = np.asarray(sc.detach().cpu().numpy()).ravel() if sc is not None else None

    H, W = rgb.shape[:2]
    bounds = np.asarray(mesh.bounds, float)  # caja en frame del objeto
    vis, sil = draw_overlay(rgb, K, pose, est, np.stack([bounds[0], bounds[1]]).reshape(2, 3))
    cv2.imwrite(str(out / "overlay_pose.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    tex = draw_textured_overlay(rgb, K, pose, est, np.stack([bounds[0], bounds[1]]).reshape(2, 3))
    cv2.imwrite(str(out / "overlay_textured.png"), cv2.cvtColor(tex, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "mask_used.png"), (mask * 255).astype(np.uint8))
    np.save(out / "depth_used.npy", depth)
    np.save(out / "pose_4x4.npy", pose)
    if scores is not None:
        np.save(out / "hypo_scores.npy", scores)
    prep_meta = Path(a.mesh).parent / "transform_original_to_prepared.json"
    (out / "meta_pose.json").write_text(json.dumps({
        "rgbdir": rd.name, "K_rect": K.tolist(),
        "mesh": {"prepared": Path(a.mesh).name,
                 "prep_transform": json.loads(prep_meta.read_text()) if prep_meta.is_file() else None,
                 "diameter_m": round(float(est.diameter), 4)},
        "mask": {"file": Path(a.mask).name, "px": int(mask.sum())},
        "iters_init": a.iters_init, "t_est_s": round(t_est, 3),
        "vram_alloc_mib": round(vram, 1),
        "fit": fit_metrics(depth, K, pose, est.diameter, est=est),
        "hypo_note": "hypo_scores = ranking del scorer, NO probabilidad/confianza",
        "conventions": "pose objeto->camara RGB rectificada, metros, RGB",
    }, indent=2))
    fm = fit_metrics(depth, K, pose, est.diameter, est=est)
    print(f"OK {out} t={t_est:.1f}s vram={vram:.0f}MiB valid_frac={fm['valid_frac']} "
          f"resid_med={fm['resid_med_m']}m")


if __name__ == "__main__":
    main()
