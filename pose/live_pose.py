"""Pose 6D en vivo (env `pose/`). Orquesta: captura RS + worker FS + track.

Arquitectura: el stack FS vive en `scripts/fs_worker.py` (proceso aparte, env
principal) porque ambos repos exigen un `Utils` top-level distinto; el bucle
de captura aquí es porte adaptado de `scripts/live.py` (warmup/dt/copy).
`live.py` queda como comparador operativo.

Teclas: r=reinit con la máscara inicial, s=guardar frame+pose, q/ESC=salir.
Uso (desde la raíz):
  uv run --project pose python pose/live_pose.py --mask <mask_init.png> --mesh <m_prepared.obj> [--frames 30]
"""
import argparse
import datetime
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from disparity_to_depth import check_stereo_cal, disp_to_depth  # noqa: E402
from project_to_rgb import project_ir_to_rgb  # noqa: E402
from pose_model import (draw_overlay, estimate_pose, fit_metrics,  # noqa: E402
                        load_textured_mesh, make_estimator, track_pose)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", default=None, help="máscara del frame de init; si se omite se pide un bbox sobre el vivo congelado (ESPACIO congela, dibuja, ENTER)")
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", default="data/processed")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--emitter", choices=["on", "off"], default="on")
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--dt-max", type=float, default=33.0)
    p.add_argument("--z-min", type=float, default=0.2)
    p.add_argument("--z-max", type=float, default=10.0)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--valid-iters", type=int, default=16)
    p.add_argument("--track-iters", type=int, default=2)
    p.add_argument("--frames", type=int, default=0, help="0 = infinito")
    return p.parse_args()


def refine_mask_from_bbox(depth_on_rgb, bbox, dz=0.06):
    """bbox (x0,y0,x1,y1) -> máscara gruesa por continuidad de profundidad + mayor CC."""
    x0, y0, x1, y1 = [max(0, v) for v in bbox]
    x1, y1 = min(depth_on_rgb.shape[1], x1), min(depth_on_rgb.shape[0], y1)
    crop = depth_on_rgb[y0:y1, x0:x1]
    v = crop[np.isfinite(crop)]
    if len(v) == 0:
        return np.zeros_like(depth_on_rgb, bool)
    med = float(np.median(v))
    keep = np.isfinite(crop) & (np.abs(crop - med) < dz)
    keep = cv2.morphologyEx(keep.astype("uint8"), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
    if n < 2:
        return np.zeros_like(depth_on_rgb, bool)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.zeros(depth_on_rgb.shape[:2], bool)
    m[y0:y1, x0:x1] = lab == big
    return m


def main():
    a = parse_args()
    mask0 = None
    if a.mask:
        mg = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE)
        if mg is None or not (mg > 127).any():
            sys.exit(f"error: máscara inválida {a.mask}")
        mask0 = mg > 127
    if not list(rs.context().devices):
        sys.exit("error: sin cámara (conecta la D415 y cierra realsense-viewer)")

    worker = subprocess.Popen(
        ["uv", "run", "--project", str(ROOT), "python", "scripts/fs_worker.py"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, cwd=str(ROOT))

    def ask_worker(l_path, r_path, d_path):
        worker.stdin.write(f"infer {l_path} {r_path} {d_path} {a.scale} {a.valid_iters}\n")
        worker.stdin.flush()
        resp = worker.stdout.readline().strip().split()
        if not resp or resp[0] != "ok":
            sys.exit(f"error: worker FS: {' '.join(resp)}")
        return float(resp[2])

    cfg_rs = rs.config()
    cfg_rs.enable_stream(rs.stream.infrared, 1, a.width, a.height, rs.format.y8, a.fps)
    cfg_rs.enable_stream(rs.stream.infrared, 2, a.width, a.height, rs.format.y8, a.fps)
    cfg_rs.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
    cfg_rs.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    pipe, started, th, stop = rs.pipeline(), False, None, threading.Event()
    ring = ROOT / "data" / "processed" / "_live_ring"
    try:
        profile = pipe.start(cfg_rs)
        started = True
    except RuntimeError as e:
        worker.terminate()
        sys.exit(f"error: cámara ocupada o streams no soportados: {e}")
    try:
        p1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        p2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        pc = profile.get_stream(rs.stream.color).as_video_stream_profile()
        i1, i2, ic = p1.get_intrinsics(), p2.get_intrinsics(), pc.get_intrinsics()

        def _d(i):
            return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy,
                    "ppx": i.ppx, "ppy": i.ppy, "coeffs": list(i.coeffs),
                    "model": getattr(getattr(i, "model", None), "name", None) or str(getattr(i, "model", None))}
        ex_rgb = p1.get_extrinsics_to(pc)
        cal = {"color": _d(ic)}
        ex = p1.get_extrinsics_to(p2)
        scal = {"ir1": _d(i1), "ir2": _d(i2), "baseline_m": float(ex.translation[0])}
        _, B = check_stereo_cal(scal)
        mesh = load_textured_mesh(a.mesh)
        est = make_estimator(mesh, Path(__file__).resolve().parent / "FoundationPose",
                             debug_dir=ROOT / "data" / "processed" / "_live_debug", debug=0)
        bounds = np.stack([np.asarray(mesh.bounds)[0], np.asarray(mesh.bounds)[1]]).reshape(2, 3)

        q: queue.Queue = queue.Queue(maxsize=2)

        def grab():
            try:
                for _ in range(a.warmup):
                    pipe.wait_for_frames()
            except RuntimeError:
                pass
            while not stop.is_set():
                try:
                    fs = pipe.wait_for_frames(timeout_ms=2000)
                except RuntimeError:
                    continue
                f1, f2, fc = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_color_frame()
                if not f1 or not f2 or not fc:
                    continue
                ts = [f1.get_timestamp(), f2.get_timestamp(), fc.get_timestamp()]
                if max(ts) - min(ts) > a.dt_max:
                    continue
                try:
                    domain = str(f1.get_frame_timestamp_domain())
                except AttributeError:
                    domain = "unknown"
                item = {"l": np.asanyarray(f1.get_data()).copy(),
                        "r": np.asanyarray(f2.get_data()).copy(),
                        "c": np.asanyarray(fc.get_data()).copy(),
                        "t": time.perf_counter(), "ts": ts, "ts_domain": domain,
                        "fn": [f1.get_frame_number(), f2.get_frame_number(), fc.get_frame_number()]}
                try:
                    q.put_nowait(item)
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put_nowait(item)

        th = threading.Thread(target=grab, daemon=True)
        th.start()
        ring.mkdir(parents=True, exist_ok=True)
        win = "deephpose live_pose (r=reinit, s=guardar, q=salir)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        n, t_prev, need_init, pose = 0, time.perf_counter(), True, None
        t_lat = t_age = 0.0
        state = "INIT"
        while True:
            try:
                it = q.get(timeout=5)
            except queue.Empty:
                continue
            while not q.empty():
                it = q.get()
            t_frame = time.perf_counter()
            lp, rp, dp = ring / "l.png", ring / "r.png", ring / "d.npy"
            cv2.imwrite(str(lp), it["l"])
            cv2.imwrite(str(rp), it["r"])
            t_infer = ask_worker(lp, rp, dp)
            disp = np.load(dp)
            sx = disp.shape[1] / a.width
            depth, valid = disp_to_depth(disp, scal["ir1"]["fx"] * sx, B, a.z_min, a.z_max)
            K_ir = np.array([[scal["ir1"]["fx"] * sx, 0, scal["ir1"]["ppx"] * sx],
                             [0, scal["ir1"]["fy"] * sx, scal["ir1"]["ppy"] * sx], [0, 0, 1]])
            buf, hit, K_rect, rect, _ = project_ir_to_rgb(
                depth, valid, K_ir, list(ex_rgb.rotation), list(ex_rgb.translation),
                it["c"], cal["color"])
            if mask0 is None:  # init interactivo: bbox sobre el vivo congelado
                cv2.imshow(win, np.ascontiguousarray(cv2.cvtColor(rect, cv2.COLOR_RGB2BGR)))
                print("dibuja el bbox del raton sobre la imagen y pulsa ENTER (ESC=salir)", flush=True)
                x, y, w, h = cv2.selectROI(win, cv2.cvtColor(rect, cv2.COLOR_RGB2BGR), False)
                if w < 5 or h < 5:
                    continue
                mask0 = (refine_mask_from_bbox(buf, (x, y, x + w, y + h)) * 255).astype(np.uint8)
                if not (mask0 > 127).any():
                    print("máscara vacía, mueve el ratón y repite el bbox", flush=True)
                    mask0 = None
                    continue
                print(f"máscara init: {(mask0 > 127).sum()} px", flush=True)
            if need_init:
                pose, dt, vram = estimate_pose(est, K_rect, rect, buf, mask0 > 127, iters=5)
                if est.pose_last is None:  # register sin puntos válidos: no hay pose que dibujar
                    state = "INIT_FAIL"
                    cv2.imshow(win, np.ascontiguousarray(cv2.cvtColor(rect, cv2.COLOR_RGB2BGR)))
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27) or (a.frames and n >= a.frames):
                        break
                    continue
                state, need_init = "INIT", False
            else:
                torch.cuda.reset_peak_memory_stats()
                pose, dt = track_pose(est, K_rect, rect, buf, iters=a.track_iters)
                vram = torch.cuda.max_memory_allocated() / 2**20
                state = "TRACK"
            fit = fit_metrics(buf, K_rect, pose, est.diameter, est=est)
            if state == "TRACK" and (fit["valid_frac"] < 0.3 or (fit["resid_med_m"] or 0) > 0.02):
                state = "LOST"
            t_lat = time.perf_counter() - t_frame
            t_age = t_frame - it["t"]
            fps = 1 / max(time.perf_counter() - t_prev, 1e-6)
            t_prev = time.perf_counter()
            vis, _ = draw_overlay(rect, K_rect, pose, est, bounds)
            cv2.putText(vis, f"{state} lat={t_lat:.1f}s edad={t_age:.1f}s fps={fps:.1f} t_est={dt:.2f}s",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, np.ascontiguousarray(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)))
            n += 1
            print(f"f{n} {state} lat={t_lat:.1f}s edad={t_age:.1f}s fps={fps:.1f} "
                  f"t_pose={dt:.2f}s frac={fit['valid_frac']} resid={fit['resid_med_m']}", flush=True)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("r"):
                need_init = True
            if k == ord("s"):
                dst = Path(a.out) / f"live_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                dst.mkdir(parents=True)
                cv2.imwrite(str(dst / "color_rect.png"), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
                np.save(dst / "depth_on_rgb.npy", buf)
                np.save(dst / "pose_4x4.npy", pose)
                (dst / "meta.json").write_text(json.dumps(
                    {"K_rect": K_rect.tolist(), "state": state, "t_infer_s": t_infer,
                     "t_pose_s": round(dt, 3), "t_lat_s": round(t_lat, 3), "frame_age_s": round(t_age, 3),
                     "timestamps_ms": it["ts"], "ts_domain": it["ts_domain"],
                     "frame_numbers": it["fn"], "fit": fit}, indent=2))
                print(f"guardada {dst}")
            if k in (ord("q"), 27) or (a.frames and n >= a.frames):
                break
    finally:
        stop.set()
        if th is not None:
            th.join(timeout=5)
        if started:
            try:
                pipe.stop()
            except RuntimeError:
                pass
        try:
            worker.stdin.write("quit\n")
            worker.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        worker.terminate()
        cv2.destroyAllWindows()
    print("live_pose cerrado")


if __name__ == "__main__":
    main()
