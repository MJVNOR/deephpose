"""Evalúa FS contra la profundidad original D415: misma geometría y colormap.

Uso:
  uv run python scripts/evaluate.py --fs data/processed/<capt>_fs --capture data/raw/<capt>
Salidas: <cap>_eval/{metrics.json} + reports/figures/eval_<cap>.png.
La D415 es comparación, no ground truth.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from disparity_to_depth import load_capture_depth


def parse_args():
    from config import load
    C = load()["depth"]
    p = argparse.ArgumentParser()
    p.add_argument("--fs", required=True)
    p.add_argument("--capture", required=True)
    p.add_argument("--z-min", type=float, default=C["z_min"])
    p.add_argument("--z-max", type=float, default=C["z_max"])
    return p.parse_args()


def turbo(norm):
    return cv2.applyColorMap((np.clip(norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]


def main():
    t0 = time.perf_counter()
    a = parse_args()
    fs, cap = Path(a.fs), Path(a.capture)
    meta = json.loads((fs / "meta.json").read_text())
    cal = json.loads((cap / "calibration.json").read_text())

    depth_fs, valid_fs, _, _ = load_capture_depth(fs / "disp.npy", cap / "calibration.json",
                                                  meta["scale"], a.z_min, a.z_max)
    raw = np.load(cap / "depth_original.npy")
    depth_d4 = raw.astype(np.float64) * cal["depth_scale_m_per_unit"]
    if depth_d4.shape != depth_fs.shape:  # misma geometría antes de restar
        depth_d4 = cv2.resize(depth_d4, (depth_fs.shape[1], depth_fs.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        print("aviso: D415 remuestreada (nearest) a geometría FS")
    valid = valid_fs & np.isfinite(depth_d4) & (depth_d4 > 0)
    diff = np.abs(depth_fs.astype(np.float64) - depth_d4)
    vd = diff[valid]

    m = {"capture": cap.name, "joint_valid_pct": round(100 * valid.mean(), 2),
         "fs_valid_pct": round(100 * valid_fs.mean(), 2),
         "d415_valid_pct": round(100 * float(np.mean(depth_d4 > 0)), 2),
         "mae_m": round(float(vd.mean()), 4), "medae_m": round(float(np.median(vd)), 4),
         "bias_m": round(float((depth_fs.astype(np.float64) - depth_d4)[valid].mean()), 4),
         "t_load_s": meta["t_load_s"], "t_infer_s": meta["t_infer_s"],
         "vram_peak_mib": meta["vram_peak_mib"]}
    m["t_eval_s"] = round(time.perf_counter() - t0, 2)

    both = np.concatenate([depth_fs[valid], depth_d4[valid]])
    lo, hi = [round(float(x), 3) for x in np.percentile(both, [2, 98])]
    n = lambda d: (d - lo) / max(hi - lo, 1e-6)
    left = cv2.imread(str(cap / "left.png"), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(cap / "right.png"), cv2.IMREAD_GRAYSCALE)
    left = cv2.cvtColor(cv2.resize(left, (depth_fs.shape[1], depth_fs.shape[0])), cv2.COLOR_GRAY2RGB)
    right = cv2.cvtColor(cv2.resize(right, (depth_fs.shape[1], depth_fs.shape[0])), cv2.COLOR_GRAY2RGB)
    disp = np.load(fs / "disp.npy")
    dn = (disp - np.percentile(disp, 2)) / max(np.percentile(disp, 98) - np.percentile(disp, 2), 1e-6)
    dimg = turbo(n(np.where(valid, depth_fs, lo)))
    dimg[~valid] = 0
    rimg = turbo(n(np.where(valid, depth_d4, lo)))
    rimg[~valid] = 0
    dd = np.zeros_like(dimg)
    dd[valid] = cv2.applyColorMap((np.clip(vd / max(np.percentile(vd, 98), 1e-6), 0, 1) * 255
                                   ).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].reshape(-1, 3)
    fig = np.concatenate([left, right, turbo(np.clip(dn, 0, 1)), dimg, rimg, dd], axis=1)

    out = Path("data/processed") / f"{cap.name}_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    cv2.imwrite(str(out / "panel.png"), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"reports/figures/eval_{cap.name}.png", cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
    print(f"OK {cap.name} MAE={m['mae_m']}m med={m['medae_m']}m valid={m['joint_valid_pct']}%")


if __name__ == "__main__":
    main()
