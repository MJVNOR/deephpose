"""Inferencia FoundationStereo sobre un par guardado. Un par a la vez, AMP validada.

Uso:
  uv run python scripts/infer.py --capture data/raw/<captura>
Guarda disparidad (px, resolución de red) + meta.json con tiempos y VRAM.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "scripts")
from fs_model import infer_disp, load_model  # noqa: E402


def parse_args():
    from config import load
    C = load()
    p = argparse.ArgumentParser()
    p.add_argument("--capture", required=True)
    p.add_argument("--ckpt", default=C["model"]["ckpt"])
    p.add_argument("--out", default=None)
    p.add_argument("--valid-iters", type=int, default=C["model"]["valid_iters"])
    p.add_argument("--scale", type=float, default=C["model"]["scale"])
    return p.parse_args()


def main():
    a = parse_args()
    cap = Path(a.capture)
    out = Path(a.out) if a.out else Path("data/processed") / f"{cap.name}_fs"
    out.mkdir(parents=True, exist_ok=False)

    model, cfg, t_load = load_model(a.ckpt, a.valid_iters)
    img0 = cv2.imread(str(cap / "left.png"), cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(str(cap / "right.png"), cv2.IMREAD_UNCHANGED)
    if img0 is None or img1 is None:
        sys.exit(f"error: no se leen {cap}/left.png o right.png")
    infer_disp(model, cfg, img0, img1, a.scale, a.valid_iters)  # calentamiento
    disp, info = infer_disp(model, cfg, img0, img1, a.scale, a.valid_iters)

    np.save(out / "disp.npy", disp)
    (out / "meta.json").write_text(json.dumps({
        "capture": cap.name, "ckpt": a.ckpt, "valid_iters": a.valid_iters,
        "scale": a.scale, "hw": list(disp.shape), "t_load_s": round(t_load, 2),
        "t_infer_s": info["t_infer_s"], "vram_peak_mib": info["vram_peak_alloc_mib"],
        "amp_finite": bool(np.all(np.isfinite(disp))),
        "disp_valid_pct": round(100 * float(np.mean(disp > 0)), 1),
    } | {k: v for k, v in info.items() if k not in ("t_infer_s",)}, indent=2))
    print(f"OK {out} t_infer={info['t_infer_s']:.2f}s t_total={info['t_total_s']:.2f}s "
          f"vram_alloc={info['vram_alloc_mib']:.0f}MiB reserved={info['vram_reserved_mib']:.0f}MiB")


if __name__ == "__main__":
    main()
