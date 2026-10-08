"""Ráfaga desatendida: N framesets en una sola sesión (un warmup) para track por etapas.

Cada frame guarda left/right/color/depth_original + metadata propia (ts, fn).
La calibración es común y se guarda una vez. Reutiliza intr_to_dict de capture.

Uso:
  uv run python scripts/capture_burst.py --frames 10 --interval 1.0 --emitter on
  (mueve el objeto entre frames; no lo toques durante cada captura)
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, "scripts")
from capture import intr_to_dict  # noqa: E402


def parse_args():
    from config import load
    C = load()["camera"]
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--interval", type=float, default=1.0, help="pausa entre frames (mueve el objeto)")
    p.add_argument("--width", type=int, default=C["width"])
    p.add_argument("--height", type=int, default=C["height"])
    p.add_argument("--fps", type=int, default=C["fps"])
    p.add_argument("--emitter", choices=["on", "off"], default=C["emitter"])
    p.add_argument("--warmup", type=int, default=C["warmup"])
    p.add_argument("--dt-max", type=float, default=C["dt_max_ms"])
    return p.parse_args()


def main():
    a = parse_args()
    if not list(rs.context().devices):
        sys.exit("error: sin cámara (conecta la D415 por USB 3.0 y cierra realsense-viewer)")
    cfg = rs.config()
    cfg.enable_stream(rs.stream.infrared, 1, a.width, a.height, rs.format.y8, a.fps)
    cfg.enable_stream(rs.stream.infrared, 2, a.width, a.height, rs.format.y8, a.fps)
    cfg.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
    cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    pipe = rs.pipeline()
    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        sys.exit(f"error: cámara ocupada o streams no soportados: {e}")
    try:
        sensor = profile.get_device().first_depth_sensor()
        want = 1.0 if a.emitter == "on" else 0.0
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, want)
        for _ in range(a.warmup):
            pipe.wait_for_frames()
        p1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        p2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        pc = profile.get_stream(rs.stream.color).as_video_stream_profile()
        pd = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ex = p1.get_extrinsics_to(p2)
        ex_rgb = p1.get_extrinsics_to(pc)
        cal = {"ir1": intr_to_dict(p1.get_intrinsics()), "ir2": intr_to_dict(p2.get_intrinsics()),
               "color": intr_to_dict(pc.get_intrinsics()),
               "depth_intrinsics": intr_to_dict(pd.get_intrinsics()),
               "extrinsics_ir1_to_ir2": {"rotation": list(ex.rotation), "translation": list(ex.translation)},
               "extrinsics_ir1_to_color": {"rotation": list(ex_rgb.rotation), "translation": list(ex_rgb.translation)},
               "baseline_m": float(ex.translation[0]),
               "depth_scale_m_per_unit": float(sensor.get_depth_scale())}
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = Path(a.out) / f"{stamp}_emitter-{a.emitter}_burst{len(range(a.frames))}"
        dst.mkdir(parents=True, exist_ok=False)
        (dst / "calibration.json").write_text(json.dumps(cal, indent=2))
        ok = 0
        for i in range(a.frames):
            fs = pipe.wait_for_frames()
            f1, f2, fd = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_depth_frame()
            fc = fs.get_color_frame()
            if not f1 or not f2 or not fd or not fc:
                print(f"aviso: frame {i} incompleto, salto", file=sys.stderr)
                continue
            ts = [f1.get_timestamp(), f2.get_timestamp(), fd.get_timestamp(), fc.get_timestamp()]
            if max(ts) - min(ts) > a.dt_max:
                print(f"aviso: frame {i} desincronizado, salto", file=sys.stderr)
                continue
            fr = dst / f"f{i:02d}"
            fr.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fr / "left.png"), np.asanyarray(f1.get_data()).copy())
            cv2.imwrite(str(fr / "right.png"), np.asanyarray(f2.get_data()).copy())
            cv2.imwrite(str(fr / "color.png"),
                        cv2.cvtColor(np.asanyarray(fc.get_data()).copy(), cv2.COLOR_RGB2BGR))
            np.save(fr / "depth_original.npy", np.asanyarray(fd.get_data()).copy())
            (fr / "metadata.json").write_text(json.dumps({
                "model": "D415", "timestamps_ms": {"ir1": ts[0], "ir2": ts[1], "depth": ts[2], "color": ts[3]},
                "frame_numbers": {"ir1": f1.get_frame_number(), "ir2": f2.get_frame_number(),
                                  "depth": fd.get_frame_number(), "color": fc.get_frame_number()},
                "emitter": a.emitter}, indent=2))
            ok += 1
            print(f"frame {i} dt={max(ts)-min(ts):.1f}ms")
            time.sleep(a.interval)
        print(f"OK {dst} frames={ok}/{a.frames}")
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
