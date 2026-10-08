"""Procesa una ráfaga: FS + proyección a RGB por frame. Reusa fs_model/disparity/project.

Uso:
  uv run python scripts/process_burst.py --burst data/raw/<sello>_burst10
Salidas: <burst>/fNN_bundle/{color_rect.png, depth_on_rgb.npy, K_rect.json, meta.json}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "scripts")
from config import load as load_cfg  # noqa: E402
from disparity_to_depth import check_stereo_cal, disp_to_depth  # noqa: E402
from fs_model import infer_disp, load_model  # noqa: E402
from project_to_rgb import project_ir_to_rgb  # noqa: E402


def main():
    C = load_cfg()
    p = argparse.ArgumentParser()
    p.add_argument("--burst", required=True)
    p.add_argument("--ckpt", default=C["model"]["ckpt"])
    p.add_argument("--valid-iters", type=int, default=C["model"]["valid_iters"])
    p.add_argument("--scale", type=float, default=C["model"]["scale"])
    p.add_argument("--z-min", type=float, default=C["depth"]["z_min"])
    p.add_argument("--z-max", type=float, default=C["depth"]["z_max"])
    a = p.parse_args()
    b = Path(a.burst)
    cal = json.loads((b / "calibration.json").read_text())
    _, B = check_stereo_cal(cal)
    model, cfg, _ = load_model(a.ckpt, a.valid_iters)
    frames = sorted([d for d in b.iterdir() if d.is_dir() and len(d.name) == 3 and d.name[0] == "f"])
    for fr in frames:
        t0 = time.perf_counter()
        l = cv2.imread(str(fr / "left.png"), cv2.IMREAD_UNCHANGED)
        r = cv2.imread(str(fr / "right.png"), cv2.IMREAD_UNCHANGED)
        disp, info = infer_disp(model, cfg, l, r, a.scale, a.valid_iters)
        sx = disp.shape[1] / cal["ir1"]["width"]  # focal efectiva real tras el resize
        depth, valid = disp_to_depth(disp, cal["ir1"]["fx"] * sx, B, a.z_min, a.z_max)
        K_ir = np.array([[cal["ir1"]["fx"] * sx, 0, cal["ir1"]["ppx"] * sx],
                         [0, cal["ir1"]["fy"] * sx, cal["ir1"]["ppy"] * sx], [0, 0, 1]])
        color = cv2.cvtColor(cv2.imread(str(fr / "color.png")), cv2.COLOR_BGR2RGB)
        ex = cal["extrinsics_ir1_to_color"]
        buf, hit, K_rect, rect, _ = project_ir_to_rgb(
            depth, valid, K_ir, ex["rotation"], ex["translation"], color, cal["color"])
        out = fr.parent / (fr.name + "_bundle")
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "color_rect.png"), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
        np.save(out / "depth_on_rgb.npy", buf)
        (out / "K_rect.json").write_text(json.dumps({"K": K_rect.tolist()}, indent=2))
        fm = json.loads((fr / "metadata.json").read_text())
        (out / "meta.json").write_text(json.dumps(
            {"frame": fr.name, "timestamps_ms": fm["timestamps_ms"],
             "frame_numbers": fm["frame_numbers"], "t_proc_s": round(time.perf_counter() - t0, 2),
             "t_infer_s": info["t_infer_s"]}, indent=2))
        print(f"OK {out.name} t={time.perf_counter()-t0:.1f}s hit={100*hit.mean():.1f}%")


if __name__ == "__main__":
    main()
