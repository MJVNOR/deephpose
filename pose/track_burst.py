"""Track por etapas sobre bundles (validación previa al vivo). Corre en env `pose/`.

frame0: register con máscara; resto: track_one con pose previa (sin global).
Marca LOST por evidencia geométrica (no por scores). Mide pico VRAM del track.

Uso (desde la raíz):
  uv run --project pose python pose/track_burst.py --burst data/raw/<sello>_burst10 --mesh data/mesh/mouse/<m>_prepared.obj --mask <mask_f00.png> --out data/processed/<sello>_track
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_model import (draw_overlay, estimate_pose, fit_metrics,  # noqa: E402
                        load_textured_mesh, make_estimator, track_pose)


def drot_deg(a, b):
    R = np.asarray(b).reshape(3, 3) @ np.asarray(a).reshape(3, 3).T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--burst", required=True)
    p.add_argument("--mesh", required=True)
    p.add_argument("--mask", required=True, help="máscara del frame0")
    p.add_argument("--out", required=True)
    p.add_argument("--track-iters", type=int, default=2, help="refinamientos por track (upstream: 2, paper: 1)")
    p.add_argument("--lost-frac", type=float, default=0.3)
    p.add_argument("--lost-resid", type=float, default=0.02)
    p.add_argument("--lost-dt", type=float, default=0.05)
    p.add_argument("--lost-drot", type=float, default=15.0)
    a = p.parse_args()
    b, out = Path(a.burst), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    bundles = sorted([d for d in b.iterdir() if d.is_dir() and d.name.endswith("_bundle")])
    m = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE)
    if m is None or not (m > 127).any():
        sys.exit(f"error: máscara inválida {a.mask}")

    mesh = load_textured_mesh(a.mesh)
    est = make_estimator(mesh, Path(__file__).resolve().parent / "FoundationPose",
                         debug_dir=out, debug=0)
    bounds = np.stack([np.asarray(mesh.bounds)[0], np.asarray(mesh.bounds)[1]]).reshape(2, 3)
    summary, prev = [], None
    for i, bd in enumerate(bundles):
        K = np.array(json.loads((bd / "K_rect.json").read_text())["K"], float)
        rgb = cv2.cvtColor(cv2.imread(str(bd / "color_rect.png")), cv2.COLOR_BGR2RGB)
        depth = np.load(bd / "depth_on_rgb.npy").astype(np.float32)
        fm = json.loads((bd / "meta.json").read_text())
        if i == 0:
            pose, dt, vram = estimate_pose(est, K, rgb, depth, m > 127, iters=5)
            state = "INIT"
        else:
            torch.cuda.reset_peak_memory_stats()
            pose, dt = track_pose(est, K, rgb, depth, iters=a.track_iters)
            vram = torch.cuda.max_memory_allocated() / 2**20
            state = "TRACK"
        fit = fit_metrics(depth, K, pose, est.diameter, est=est)
        row = {"frame": bd.name, "state": state, "t_s": round(dt, 3),
               "vram_alloc_mib": round(float(vram), 1),
               "valid_frac": fit["valid_frac"], "resid_med_m": fit["resid_med_m"],
               "timestamps_ms": fm["timestamps_ms"]}
        if prev is not None:
            row["dt_m"] = round(float(np.linalg.norm(pose[:3, 3] - prev[:3, 3])), 4)
            row["drot_deg"] = round(drot_deg(prev[:3, :3], pose[:3, :3]), 2)
            if (fit["valid_frac"] < a.lost_frac or (fit["resid_med_m"] or 0) > a.lost_resid
                    or row["dt_m"] > a.lost_dt or row["drot_deg"] > a.lost_drot):
                row["state"] = "LOST"
        np.save(out / f"{bd.name}_pose.npy", pose)
        vis, _ = draw_overlay(rgb, K, pose, est, bounds)
        cv2.imwrite(str(out / f"{bd.name}_overlay.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        summary.append(row)
        prev = pose
        print(f"{bd.name} {row['state']} t={dt:.1f}s frac={fit['valid_frac']} resid={fit['resid_med_m']}")
    (out / "track_summary.json").write_text(json.dumps({
        "burst": b.name, "track_iters": a.track_iters,
        "lost_thr": {"frac": a.lost_frac, "resid_m": a.lost_resid, "dt_m": a.lost_dt, "drot_deg": a.lost_drot},
        "frames": summary}, indent=2))
    lost = sum(1 for r in summary if r["state"] == "LOST")
    print(f"OK {out} frames={len(summary)} lost={lost}")


if __name__ == "__main__":
    main()
