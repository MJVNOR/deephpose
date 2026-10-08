"""Captura IR estéreo + profundidad original de la D415. Una carpeta por captura en data/raw/.

Uso:
  uv run python scripts/capture.py --emitter on
  uv run python scripts/capture.py --emitter off --width 640 --height 480 --fps 30
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    from config import load
    C = load()["camera"]
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw")
    p.add_argument("--width", type=int, default=C["width"])
    p.add_argument("--height", type=int, default=C["height"])
    p.add_argument("--fps", type=int, default=C["fps"])
    p.add_argument("--emitter", choices=["on", "off"], default=C["emitter"])
    p.add_argument("--warmup", type=int, default=C["warmup"], help="framesets a descartar (autoexposición)")
    p.add_argument("--dt-max", type=float, default=C["dt_max_ms"], help="desfase máximo en ms")
    return p.parse_args()


def intr_to_dict(i):
    m = getattr(i, "model", None)
    name = getattr(m, "name", None) or str(m)  # ponytail: modelo como texto; lectores viejos sin "model" asumen abajo
    return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
            "ppx": i.ppx, "ppy": i.ppy, "coeffs": list(i.coeffs), "model": name}


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
        sys.exit(f"error: combinación de streams no soportada o cámara ocupada: {e}")

    try:
        sensor = profile.get_device().first_depth_sensor()
        want = 1.0 if a.emitter == "on" else 0.0
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, want)
        actual = sensor.get_option(rs.option.emitter_enabled) if sensor.supports(rs.option.emitter_enabled) else -1.0

        for _ in range(a.warmup):  # deja estabilizar exposición
            pipe.wait_for_frames()

        fs = pipe.wait_for_frames()
        f1, f2, fd = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_depth_frame()
        fc = fs.get_color_frame()
        if not f1 or not f2 or not fd or not fc:
            sys.exit("error: captura incompleta (falta IR1/IR2/depth/color del mismo frameset)")
        t1, t2, td, tc = f1.get_timestamp(), f2.get_timestamp(), fd.get_timestamp(), fc.get_timestamp()
        if max(t1, t2, td, tc) - min(t1, t2, td, tc) > a.dt_max:
            sys.exit(f"error: frames desincronizados: {t1:.1f} {t2:.1f} {td:.1f} {tc:.1f} ms")

        img1, img2 = np.asanyarray(f1.get_data()), np.asanyarray(f2.get_data())
        depth = np.asanyarray(fd.get_data())  # uint16 original, sin tocar
        color = np.asanyarray(fc.get_data())  # RGB8

        p1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        p2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        pc = profile.get_stream(rs.stream.color).as_video_stream_profile()
        pd = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ex = p1.get_extrinsics_to(p2)
        ex_rgb = p1.get_extrinsics_to(pc)  # IR-izq -> RGB
        di = pd.get_intrinsics()
        i1 = p1.get_intrinsics()
        if abs(di.fx - i1.fx) > 1e-3 or abs(di.ppx - i1.ppx) > 1e-3:
            print("aviso: depth no comparte viewpoint con IR1; la proyección asume depth==IR1", file=sys.stderr)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dst = Path(a.out) / f"{stamp}_emitter-{a.emitter}"
        dst.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(str(dst / "left.png"), img1)
        cv2.imwrite(str(dst / "right.png"), img2)
        cv2.imwrite(str(dst / "color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        np.save(dst / "depth_original.npy", depth)
        (dst / "calibration.json").write_text(json.dumps({
            "ir1": intr_to_dict(p1.get_intrinsics()),
            "ir2": intr_to_dict(p2.get_intrinsics()),
            "color": intr_to_dict(pc.get_intrinsics()),
            "depth_intrinsics": intr_to_dict(di),
            "extrinsics_ir1_to_ir2": {"rotation": list(ex.rotation), "translation": list(ex.translation)},
            "extrinsics_ir1_to_color": {"rotation": list(ex_rgb.rotation), "translation": list(ex_rgb.translation)},
            "baseline_m": float(ex.translation[0]),
            "depth_scale_m_per_unit": float(sensor.get_depth_scale()),
        }, indent=2))
        (dst / "metadata.json").write_text(json.dumps({
            "model": "D415", "width": a.width, "height": a.height, "fps": a.fps,
            "timestamps_ms": {"ir1": t1, "ir2": t2, "depth": td, "color": tc},
            "frame_numbers": {"ir1": f1.get_frame_number(), "ir2": f2.get_frame_number(),
                              "depth": fd.get_frame_number(), "color": fc.get_frame_number()},
            "emitter_requested": a.emitter, "emitter_actual": actual, "warmup_frames": a.warmup,
        }, indent=2))
        print(f"OK {dst} emitter={a.emitter}({actual}) dt={max(t1,t2,td,tc)-min(t1,t2,td,tc):.1f}ms")
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
