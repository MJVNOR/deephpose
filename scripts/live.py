"""Captura continua D415 + inferencia FS en vivo. Cola limitada, descarta viejos.

Teclas: s = guardar captura (formato capture.py), q/ESC = salir.
Uso:
  uv run python scripts/live.py --emitter on [--frames 50]
"""
import argparse
import datetime
import json
import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

sys.path.insert(0, "scripts")
from capture import intr_to_dict  # noqa: E402
from disparity_to_depth import check_stereo_cal, disp_to_depth  # noqa: E402
from fs_model import infer_disp, load_model  # noqa: E402


def parse_args():
    from config import load
    C = load()
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=C["model"]["ckpt"])
    p.add_argument("--valid-iters", type=int, default=C["model"]["valid_iters"])
    p.add_argument("--scale", type=float, default=C["model"]["scale"])
    p.add_argument("--width", type=int, default=C["camera"]["width"])
    p.add_argument("--height", type=int, default=C["camera"]["height"])
    p.add_argument("--fps", type=int, default=C["camera"]["fps"])
    p.add_argument("--emitter", choices=["on", "off"], default=C["camera"]["emitter"])
    p.add_argument("--warmup", type=int, default=C["camera"]["warmup"])
    p.add_argument("--dt-max", type=float, default=C["camera"]["dt_max_ms"])
    p.add_argument("--z-min", type=float, default=C["depth"]["z_min"])
    p.add_argument("--z-max", type=float, default=C["depth"]["z_max"])
    p.add_argument("--view", choices=["both", "fs", "d415"], default=C["live"]["view"])
    p.add_argument("--zoom", type=float, default=C["live"]["zoom"], help="solo display, no toca datos")
    p.add_argument("--frames", type=int, default=C["live"]["frames"], help="0 = infinito (tecla q para salir)")
    return p.parse_args()


def turbo_norm(d, lo=0.2, hi=10.0):
    """Profundidad -> BGR uint8 listo para imshow (applyColorMap ya devuelve BGR)."""
    return cv2.applyColorMap(((np.clip(d, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8),
                             cv2.COLORMAP_TURBO)


def main():
    a = parse_args()
    if not list(rs.context().devices):
        sys.exit("error: sin cámara (conecta la D415 y cierra realsense-viewer)")
    try:
        model, cfg, _ = load_model(a.ckpt, a.valid_iters)
    except torch.cuda.OutOfMemoryError as e:
        sys.exit(f"error: sin memoria CUDA con scale={a.scale}, prueba --scale 0.5: {e}")
    except RuntimeError as e:
        sys.exit(f"error: fallo cargando modelo: {e}")

    cfg_rs = rs.config()
    cfg_rs.enable_stream(rs.stream.infrared, 1, a.width, a.height, rs.format.y8, a.fps)
    cfg_rs.enable_stream(rs.stream.infrared, 2, a.width, a.height, rs.format.y8, a.fps)
    cfg_rs.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
    cfg_rs.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    pipe = rs.pipeline()
    started, th = False, None
    stop = threading.Event()
    try:
        profile = pipe.start(cfg_rs)
        started = True
    except RuntimeError as e:
        sys.exit(f"error: cámara ocupada o streams no soportados: {e}")

    try:
        sensor = profile.get_device().first_depth_sensor()
        want = 1.0 if a.emitter == "on" else 0.0
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, want)
        actual = sensor.get_option(rs.option.emitter_enabled) if sensor.supports(rs.option.emitter_enabled) else -1.0
        scale_d = float(sensor.get_depth_scale())
        p1 = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        p2 = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        pc = profile.get_stream(rs.stream.color).as_video_stream_profile()
        pd = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ex, ex_rgb = p1.get_extrinsics_to(p2), p1.get_extrinsics_to(pc)

        cal = {"ir1": intr_to_dict(p1.get_intrinsics()), "ir2": intr_to_dict(p2.get_intrinsics()),
               "color": intr_to_dict(pc.get_intrinsics()),
               "depth_intrinsics": intr_to_dict(pd.get_intrinsics()),
               "extrinsics_ir1_to_ir2": {"rotation": list(ex.rotation), "translation": list(ex.translation)},
               "extrinsics_ir1_to_color": {"rotation": list(ex_rgb.rotation), "translation": list(ex_rgb.translation)},
               "baseline_m": float(ex.translation[0]), "depth_scale_m_per_unit": scale_d}
        di, i1 = pd.get_intrinsics(), p1.get_intrinsics()
        if abs(di.fx - i1.fx) > 1e-3 or abs(di.ppx - i1.ppx) > 1e-3:
            print("aviso: depth no comparte viewpoint con IR1; la proyección asume depth==IR1", file=sys.stderr)
        _, B = check_stereo_cal(cal)  # aborta aquí si la calibración no vale para Z=fx·B/d

        from config import load as _load
        q: queue.Queue = queue.Queue(maxsize=_load()["live"]["queue_size"])

        def grab():
            dropped = 0
            try:
                for _ in range(a.warmup):  # deja estabilizar exposición
                    pipe.wait_for_frames()
            except RuntimeError:
                pass  # la cámara a veces tarda tras abrir; el loop reintenta
            while not stop.is_set():
                try:
                    fs = pipe.wait_for_frames(timeout_ms=2000)
                except RuntimeError:
                    continue
                f1, f2, fd, fc = (fs.get_infrared_frame(1), fs.get_infrared_frame(2),
                                 fs.get_depth_frame(), fs.get_color_frame())
                if not f1 or not f2 or not fd or not fc:
                    continue
                # dominio de timestamps (normalmente hardware clock, ms)
                try:
                    domain = str(f1.get_frame_timestamp_domain())
                except AttributeError:
                    domain = "unknown"
                ts = [f1.get_timestamp(), f2.get_timestamp(), fd.get_timestamp(), fc.get_timestamp()]
                dt = max(ts) - min(ts)
                if dt > a.dt_max:  # frameset desincronizado: se descarta, no se mezcla
                    dropped += 1
                    continue
                item = {"l": np.asanyarray(f1.get_data()).copy(),
                        "r": np.asanyarray(f2.get_data()).copy(),  # ponytail: copy; la vista muere en el próximo wait_for_frames
                        "d": np.asanyarray(fd.get_data()).copy(), "c": np.asanyarray(fc.get_data()).copy(),
                        "t": time.perf_counter(), "ts": ts, "dt_ms": dt, "ts_domain": domain,
                        "fn": [f1.get_frame_number(), f2.get_frame_number(), fd.get_frame_number(), fc.get_frame_number()]}
                try:
                    q.put_nowait(item)
                except queue.Full:
                    try:
                        q.get_nowait()  # descarta el más viejo, el procesamiento va atrás
                    except queue.Empty:
                        pass
                    q.put_nowait(item)

        th = threading.Thread(target=grab, daemon=True)
        th.start()
        # calentamiento del modelo con el primer frameset
        w0 = q.get(timeout=15)["l"]
        infer_disp(model, cfg, w0, w0, a.scale, a.valid_iters)

        n, t_prev, t_inf = 0, time.perf_counter(), 0.0
        vlo, vhi = 0.2, 3.0  # rango display con memoria (suaviza parpadeo)
        win = "deephpose live (s=guardar, q=salir)"
        mouse = {"x": -1, "y": -1}
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)  # la imagen sigue al tamaño de la ventana
        cv2.setMouseCallback(win, lambda e, x, y, *_: mouse.update(x=x, y=y))
        while True:
            try:
                it = q.get(timeout=5)
            except queue.Empty:
                continue
            while not q.empty():  # salta al más reciente
                it = q.get()
            disp, info = infer_disp(model, cfg, it["l"], it["r"], a.scale, a.valid_iters)
            t_inf = info["t_infer_s"]
            sx = disp.shape[1] / cal["ir1"]["width"]  # focal efectiva real tras el resize
            z, _ = disp_to_depth(disp, cal["ir1"]["fx"] * sx, B, a.z_min, a.z_max)
            d4 = it["d"].astype(float) * scale_d
            if d4.shape != z.shape:  # misma geometría antes de comparar
                d4 = cv2.resize(d4, (z.shape[1], z.shape[0]), interpolation=cv2.INTER_NEAREST)
            jv = np.concatenate([z[np.isfinite(z)], d4[d4 > 0]])
            if len(jv):  # auto-rango p2-p98 compartido, con memoria anti-parpadeo
                lo, hi = np.percentile(jv, [2, 98])
                vlo, vhi = 0.9 * vlo + 0.1 * lo, 0.9 * vhi + 0.1 * hi
            panels = {"d415": turbo_norm(np.where(d4 > 0, d4, np.inf), vlo, vhi),
                      "fs": turbo_norm(np.where(np.isfinite(z), z, np.inf), vlo, vhi)}
            panels["d415"][d4 <= 0] = 0  # inválidos en negro, no en azul
            panels["fs"][~np.isfinite(z)] = 0
            if a.view == "both":
                vis = np.concatenate([panels["d415"], panels["fs"]], axis=1)
                tag = "D415 | FS"
            else:
                vis = panels[a.view]
                tag = "FS" if a.view == "fs" else "D415"
            fps = 1 / max(time.perf_counter() - t_prev, 1e-6)
            t_prev = time.perf_counter()
            try:
                _, _, ww, wh = cv2.getWindowImageRect(win)
            except cv2.error:
                break  # ventana cerrada con la X
            if ww < 10 or wh < 10:
                ww, wh = int(vis.shape[1] * a.zoom), int(vis.shape[0] * a.zoom)
            show = cv2.resize(vis, (ww, wh), interpolation=cv2.INTER_NEAREST)
            Hs, Ws = wh, ww
            mx, my = mouse["x"], mouse["y"]  # distancia bajo el cursor
            if 0 <= mx < Ws and 0 <= my < Hs:
                dx, dy = int(mx * vis.shape[1] / Ws), int(my * vis.shape[0] / Hs)
                if a.view == "both":
                    src, xx = (d4, dx) if dx < z.shape[1] else (z, dx - z.shape[1])
                else:
                    src, xx = (z, dx) if a.view == "fs" else (d4, dx)
                v = src[dy, xx] if 0 <= xx < src.shape[1] and 0 <= dy < src.shape[0] else np.nan
                txt = "sin dato" if not (np.isfinite(v) and v > 0) else (
                    f"{v:.2f} m" if v >= 1 else f"{v * 100:.1f} cm")
                cv2.circle(show, (mx, my), 5, (255, 255, 255), 1)
                cv2.putText(show, txt, (min(mx + 12, Ws - 130), max(my - 12, 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(show, f"{tag} t={t_inf:.2f}s fps={fps:.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(win, np.ascontiguousarray(show))
            n += 1
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dst = Path("data/raw") / f"{stamp}_emitter-{a.emitter}_live"
                dst.mkdir(parents=True)
                cv2.imwrite(str(dst / "left.png"), it["l"])
                cv2.imwrite(str(dst / "right.png"), it["r"])
                cv2.imwrite(str(dst / "color.png"), cv2.cvtColor(it["c"], cv2.COLOR_RGB2BGR))
                np.save(dst / "depth_original.npy", it["d"])
                (dst / "calibration.json").write_text(json.dumps(cal, indent=2))
                (dst / "metadata.json").write_text(json.dumps(
                    {"model": "D415", "width": a.width, "height": a.height, "fps": a.fps,
                     "timestamps_ms": {"ir1": it["ts"][0], "ir2": it["ts"][1],
                                       "depth": it["ts"][2], "color": it["ts"][3]},
                     "frame_numbers": {"ir1": it["fn"][0], "ir2": it["fn"][1],
                                       "depth": it["fn"][2], "color": it["fn"][3]},
                     "ts_domain": it["ts_domain"], "dt_ms": round(it["dt_ms"], 1),
                     "emitter_requested": a.emitter, "emitter_actual": actual,
                     "warmup_frames": a.warmup, "t_infer_s": round(t_inf, 3)}, indent=2))
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
        cv2.destroyAllWindows()
    print("live cerrado, cámara liberada")


if __name__ == "__main__":
    main()
