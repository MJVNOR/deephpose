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
import concurrent.futures
import datetime
import time as _time

_T0 = _time.perf_counter()


def ts():
    return f"{_time.perf_counter()-_T0:7.1f}s"
import json
import os
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
                        load_textured_mesh, make_estimator, track_pose, render_depth, render_textured, prepare_fast_pose)
from fs_transport import FSClient
from display_frame import select_display_frame
from sam_initialization import run_sam_attempt
from recovery import RecoverySearch, fit_is_valid
from sam_mask import unload_sam, prepare_sam, configure_sam  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask", default=None, help="máscara PNG para el primer init (se reutiliza con r)")
    p.add_argument("--mask-text", default=None, help="prompt SAM (p.ej. 'computer mouse'): máscara automática + re-init solo")
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
    p.add_argument("--rec", action="store_true", help="graba streams a data/bags/live_<ts>.db3 (post: reproduce con --bag)")
    p.add_argument("--bag", default=None, help="reproduce un .db3/.bag en vez de la cámara (tests repetibles, sin D415)")
    p.add_argument('--headless', action='store_true', help='Sin ventana; requiere máscara o inicialización guardada')
    p.add_argument('--metrics-jsonl', type=Path, help='Telemetría por pose, crea un archivo nuevo')
    p.add_argument('--benchmark-init', type=Path, help='Replay fijo: pausa bag durante carga y warmup, usa su primer bundle')
    p.add_argument('--initial-state', type=Path, help='Estado inicial guardado por benchmark_replay (requiere --benchmark-init)')
    p.add_argument('--legacy-costs', action='store_true', help='Solo A/B: IPC por disco y segundo render como la versión anterior')
    p.add_argument('--snapshots',type=Path,help='Guarda muestras visuales cada 20 poses (usar fuera del benchmark de velocidad)')
    p.add_argument('--sorted-zbuffer',action='store_true',help='Solo A/B: proyección con los ordenamientos anteriores')
    p.add_argument('--reference-volumes',action='store_true',help='Solo A/B: volúmenes estéreo originales de FoundationStereo')
    p.add_argument('--reference-cpu',action='store_true',help='Solo A/B: proyección detallada y dibujo de ejes anterior')
    p.add_argument('--reference-tracking',action='store_true',help='Solo A/B: tracking original sin reutilización de rejilla/caché')
    p.add_argument('--reference-init',action='store_true',help='Solo A/B: registro original y FS continuo durante INIT')
    p.add_argument('--fast-init',action='store_true',help='Experimental: SAM precargado residente y registro 42 orientaciones/8 finalistas; puede cambiar la pose')
    p.add_argument('--sam-backend',choices=['sam3.1','sam3','efficientsam3'],default='sam3.1',help='SAM 3.1 predeterminado; SAM3 y EfficientSAM3 opcionales')
    p.add_argument('--sam-threshold',type=float,default=None,help='Umbral: SAM 3.1/SAM3=0.5, EfficientSAM3=0.1')
    p.add_argument('--overlay', choices=['mesh', 'silhouette'], default='mesh', help='Malla 3D superpuesta (default) o silueta roja')
    p.add_argument('--overlay-alpha', type=float, default=.65, help='Opacidad del modelo entre 0 y 1 (default: 0.65)')
    p.add_argument('--no-auto-recover', action='store_true', help='Desactiva la recuperación automática con --mask-text')
    p.add_argument('--recovery-lost-frames', type=int, default=3, help='Fallos consecutivos antes de buscar con SAM (default: 3)')
    p.add_argument('--recovery-retry-s', type=float, default=1.0, help='Espera entre búsquedas fallidas (default: 1 s)')
    args = p.parse_args()
    if not 0 <= args.overlay_alpha <= 1:
        p.error('--overlay-alpha debe estar entre 0 y 1')
    return args


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
    a.sam_threshold = configure_sam(a.sam_backend, a.sam_threshold)
    recovery = RecoverySearch(a.recovery_lost_frames, a.recovery_retry_s)
    auto_recover = bool(a.mask_text and not a.no_auto_recover)
    if a.fast_init and a.reference_init:
        sys.exit('error: --fast-init y --reference-init son incompatibles')
    if a.snapshots:
        a.snapshots.mkdir(parents=True,exist_ok=False)
    if bool(a.benchmark_init) != bool(a.initial_state) or (a.benchmark_init and not a.bag):
        sys.exit('error: --benchmark-init y --initial-state requieren --bag y se usan juntos')
    if a.headless and not (a.mask or a.mask_text or a.initial_state):
        sys.exit('error: headless requiere máscara o estado inicial')
    mask_file = None
    if a.mask:
        mg = cv2.imread(str(a.mask), cv2.IMREAD_GRAYSCALE)
        if mg is None or not (mg > 127).any():
            sys.exit(f"error: máscara inválida {a.mask}")
        mask_file = (mg > 127).astype(np.uint8) * 255
    mask0 = None
    is_bag = bool(a.bag)
    if not is_bag and not list(rs.context().devices):
        sys.exit("error: sin cámara (conecta la D415 y cierra realsense-viewer)")

    print("deephpose: arrancando worker FS...", flush=True)
    os.environ['DEEPH_FS_REFERENCE_VOLUMES'] = '1' if a.reference_volumes else '0'
    from fs_transport import LegacyFSClient
    worker = LegacyFSClient(ROOT) if a.legacy_costs else FSClient(ROOT)

    cfg_rs = rs.config()
    cfg_rs.enable_stream(rs.stream.infrared, 1, a.width, a.height, rs.format.y8, a.fps)
    cfg_rs.enable_stream(rs.stream.infrared, 2, a.width, a.height, rs.format.y8, a.fps)
    cfg_rs.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, a.fps)
    cfg_rs.enable_stream(rs.stream.color, a.width, a.height, rs.format.rgb8, a.fps)
    if is_bag:
        cfg_rs.enable_device_from_file(str(a.bag), repeat_playback=False)
    elif a.rec:
        bagdir = ROOT / "data" / "bags"
        bagdir.mkdir(parents=True, exist_ok=True)
        bagpath = bagdir / f"live_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db3"  # ponytail: el SDK nuevo graba .db3 (SQLite), no .bag
        cfg_rs.enable_record_to_file(str(bagpath))
        print(f"grabando {bagpath} (30s: 5 quieto + lento + 1 swipe)", flush=True)
    pipe, started, th, th2, th3, stop = rs.pipeline(), False, None, None, None, threading.Event()
    pool = None  # se crea tras cargar modelos; el finally lo tolera si algo falla antes
    metrics_file = None
    try:
        profile = pipe.start(cfg_rs)
        started = True
        if is_bag:
            playback = profile.get_device().as_playback()
            playback.pause()
    except RuntimeError as e:
        worker.close()
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
        print("deephpose: cargando pose (scorer+refiner+malla)...", flush=True)
        mesh = load_textured_mesh(a.mesh)
        est = make_estimator(mesh, Path(__file__).resolve().parent / "FoundationPose",
                             debug_dir=ROOT / "data" / "processed" / "_live_debug", debug=0)
        sam_prepare_s = 0.0
        pose_prepare_s = 0.0
        if a.mask_text:
            print(f'deephpose: backend {a.sam_backend}, umbral {a.sam_threshold:g}', flush=True)
        if a.mask_text and a.fast_init:
            print('deephpose: preparando SAM residente (carga inicial, fuera del tiempo de INIT)...', flush=True)
            sam_prepare_s = prepare_sam(a.mask_text, (a.height, a.width))
            print(f'deephpose: SAM preparado en {sam_prepare_s:.2f}s; búsqueda rápida experimental activa', flush=True)
        elif a.mask_text:
            print("deephpose: SAM bajo demanda (carga en el primer init)", flush=True)
        if a.fast_init:
            print('deephpose: preparando kernels del registro rápido...', flush=True)
            warm_K = np.array([[ic.fx, 0, ic.ppx], [0, ic.fy, ic.ppy], [0, 0, 1]])
            pose_prepare_s = prepare_fast_pose(est, warm_K, (a.height, a.width))
            print(f'deephpose: registro preparado en {pose_prepare_s:.2f}s', flush=True)
        print("deephpose: listo, abre la ventana y arranca (mantén el ratón quieto hasta TRACK)", flush=True)
        bounds = np.stack([np.asarray(mesh.bounds)[0], np.asarray(mesh.bounds)[1]]).reshape(2, 3)
        initial_pose = None
        if a.benchmark_init:
            from prepare_replay import sha256
            benchmark = json.loads((a.benchmark_init/'manifest.json').read_text())
            initial_meta = json.loads((a.initial_state/'manifest.json').read_text())
            if initial_meta['manifest_sha256'] != sha256(a.benchmark_init/'manifest.json') or benchmark['bag_sha256'] != sha256(a.bag):
                raise ValueError('Benchmark initial state or recording changed')
            if initial_meta['mesh_sha256'] != sha256(a.mesh) or initial_meta['frame_ids'][0] != benchmark['frames'][0]['frame']:
                raise ValueError('Initial state mesh/frame mismatch')
            first = benchmark['frames'][0]
            with np.load(a.benchmark_init/first['bundle']) as b:
                init_rgb,init_depth,init_K = b['rgb'],b['depth'],b['K']
            initial_pose = np.load(a.initial_state/'initial_pose.npy')
            centered = initial_pose @ np.linalg.inv(est.get_tf_to_centered_mesh().detach().cpu().numpy())
            with np.load(a.benchmark_init/first['raw']) as b:
                for _ in range(2):
                    worker.infer(b['left'],b['right'],a.scale,a.valid_iters)
            for _ in range(3):
                est.pose_last = torch.as_tensor(centered,device='cuda',dtype=torch.float32)
                track_pose(est,init_K,init_rgb,init_depth,iters=a.track_iters,reference=a.reference_tracking)
            est.pose_last = torch.as_tensor(centered,device='cuda',dtype=torch.float32)
            # Seek while paused: model startup consumes no source frames.
            playback = profile.get_device().as_playback()
            playback.set_real_time(True)
            playback.seek(datetime.timedelta(0))
        if a.metrics_jsonl:
            a.metrics_jsonl.parent.mkdir(parents=True,exist_ok=True)
            metrics_file = a.metrics_jsonl.open('x',encoding='utf-8')
            from prepare_replay import sha256
            a.metrics_jsonl.with_suffix('.manifest.json').write_text(json.dumps({
                'args':vars(a),'mode':'real-time source, latest-frame queues, initialization excluded' if a.benchmark_init else 'live',
                'display_mode':'synchronized_rgb_pose',
                'recovery_source_sha256':sha256(ROOT/'pose/recovery.py'),
                'display_box_source_sha256':sha256(ROOT/'pose/mesh_box.py'),
                'sources':{str(p):sha256(p) for p in [Path(__file__),ROOT/'pose/pose_model.py',ROOT/'pose/initialization_fast.py',ROOT/'pose/mesh_initialization.py',ROOT/'pose/tracking_fast.py',ROOT/'pose/overlay_fast.py',ROOT/'pose/display_frame.py',ROOT/'pose/sam_initialization.py',ROOT/'pose/sam_mask.py',ROOT/'pose/efficient_sam_client.py',ROOT/'pose/sam31_client.py',ROOT/'experiments/sam31/backend.py',ROOT/'experiments/sam31/worker.py',ROOT/'experiments/sam31/compat.py',ROOT/'experiments/efficientsam3/backend.py',ROOT/'experiments/efficientsam3/worker.py',ROOT/'scripts/fs_model.py',ROOT/'scripts/fs_volumes.py',ROOT/'scripts/fs_transport.py',ROOT/'scripts/fs_worker.py',ROOT/'scripts/project_to_rgb.py']}
            },indent=2,default=str))

        q: queue.Queue = queue.Queue(maxsize=2)
        # Un solo hilo bombea la ventana; el contenido cambia con cada pose completa.
        # el main publica BGR en view y lee teclas por keyq (dict ops atómicas por GIL)
        view = {"raw": None, "vis": None, "seq": 0, "live_status": 'Preparando profundidad...'}
        keyq: queue.Queue = queue.Queue()
        freeze = threading.Event()  # selectROI modal: el display no pinta mientras elige bbox
        init_gpu_busy = threading.Event()
        sam_gpu_busy = threading.Event()
        fs_gpu_lock = threading.Lock()

        def sam31_recovery(rgb, text):
            # On 8 GiB GPUs the three simultaneous activation sets cause paging.
            # Capture/UI continue; let in-flight FS finish before SAM allocates.
            sam_gpu_busy.set()
            try:
                with fs_gpu_lock:
                    return run_sam_attempt(rgb, text)
            finally:
                sam_gpu_busy.clear()

        if a.sam_backend == 'sam3.1':
            recovery.segment = sam31_recovery
        init_progress = {'phase':'preparando', 'hypotheses':0}

        def set_init_phase(phase, hypotheses):
            init_progress.update(phase=phase, hypotheses=hypotheses)
        pause_for_init = threading.Event()
        bag_init_paused = threading.Event()
        if is_bag and initial_pose is None:
            pause_for_init.set()

        def poll_key():
            try:
                return keyq.get_nowait()
            except queue.Empty:
                return None

        def grab():
            try:
                if a.benchmark_init:
                    while True:
                        warm = pipe.wait_for_frames()
                        infrared = warm.get_infrared_frame(1)
                        if infrared and infrared.get_frame_number() >= first['frame']:
                            break
                else:
                    for _ in range(a.warmup):
                        pipe.wait_for_frames()
            except RuntimeError:
                pass
            fails = 0
            while not stop.is_set():
                try:
                    fs = pipe.wait_for_frames(timeout_ms=2000)
                    fails = 0
                except RuntimeError:
                    if is_bag and playback.current_status() == rs.playback_status.paused:
                        fails = 0
                        continue
                    fails += 1
                    if is_bag and (profile.get_device().as_playback().current_status() == rs.playback_status.stopped or fails >= 5):
                        stop.set()
                        return
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
                if is_bag and pause_for_init.is_set():
                    playback.pause()  # first input remains fixed while FS/SAM/register load
                    bag_init_paused.set()
                    pause_for_init.clear()
                try:
                    q.put_nowait(item)
                except queue.Full:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    q.put_nowait(item)
                view["raw"] = cv2.cvtColor(item["c"], cv2.COLOR_RGB2BGR)  # crudo 30fps aunque el track vaya a 2.5

        if is_bag and not a.benchmark_init:
            playback.set_real_time(True)
            playback.seek(datetime.timedelta(0))
        th = threading.Thread(target=grab, daemon=True)
        th.start()
        if is_bag:
            playback.resume()
        pq: queue.Queue = queue.Queue(maxsize=1)  # bundles listos para track (siempre el más nuevo)

        def infer_loop():
            while not stop.is_set():
                if sam_gpu_busy.is_set():
                    stop.wait(0.01)
                    continue
                if not a.reference_init and init_gpu_busy.is_set():
                    stop.wait(0.05)
                    continue
                try:
                    it = q.get(timeout=1)
                except queue.Empty:
                    continue
                while not q.empty():  # salta al más reciente
                    it = q.get()
                if not a.reference_init and init_gpu_busy.is_set():
                    continue  # a newer capture will be queued when initialization finishes
                t_frame = time.perf_counter()
                try:
                    with fs_gpu_lock:
                        if sam_gpu_busy.is_set():
                            continue
                        disp,info = worker.infer(it['l'],it['r'],a.scale,a.valid_iters)
                    t_infer = info['t_infer_s']
                except Exception as exc:
                    print(f'error: worker FS: {exc}',flush=True)
                    stop.set()
                    break
                t_worker_end = time.perf_counter()
                if disp.shape != (a.height, a.width):  # FS a menor escala: interpola disp a res. completa
                    sx = disp.shape[1] / a.width  # disp en px de red; a px de res. completa (si no, Z sale 1/sx × lejos)
                    disp = cv2.resize(disp, (a.width, a.height), interpolation=cv2.INTER_LINEAR) / sx
                depth, valid = disp_to_depth(disp, scal["ir1"]["fx"], B, a.z_min, a.z_max)
                t_depth_end = time.perf_counter()
                K_ir = np.array([[scal["ir1"]["fx"], 0, scal["ir1"]["ppx"]],
                                 [0, scal["ir1"]["fy"], scal["ir1"]["ppy"]], [0, 0, 1]])
                buf, hit, K_rect, rect, _ = project_ir_to_rgb(
                    depth, valid, K_ir, list(ex_rgb.rotation), list(ex_rgb.translation),
                    it["c"], cal["color"], sorted_zbuffer=a.sorted_zbuffer,
                    return_details=a.reference_cpu)
                t_project_end = time.perf_counter()
                bd = {"rect": rect, "buf": buf, "K_rect": K_rect, "t": it["t"], "t_frame": t_frame,
                      "t_infer": t_infer, "ts": it["ts"], "ts_domain": it["ts_domain"], "fn": it["fn"],
                      "ready":t_project_end,"stages":{"capture_queue_s":t_frame-it['t'],
                      "worker_roundtrip_s":t_worker_end-t_frame,"depth_s":t_depth_end-t_worker_end,
                      "projection_s":t_project_end-t_depth_end,"fs_details":info}}
                try:
                    pq.put_nowait(bd)
                except queue.Full:
                    try:
                        pq.get_nowait()  # descarta el bundle viejo: el track siempre va a lo último
                    except queue.Empty:
                        pass
                    pq.put_nowait(bd)

        th2 = threading.Thread(target=infer_loop, daemon=True)
        th2.start()
        win = "deephpose live_pose (b=bbox, r=reinit, s=guardar, q=salir)"

        def show_loop():
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "arrancando (warmup + modelos)...", (12, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(win, blank)
            lastkey, lastt = None, 0.0
            while not stop.is_set():
                if freeze.is_set():  # selectROI modal en el main: no pelear por la ventana
                    cv2.waitKey(30)
                    continue
                # Completed overlay and rectified RGB belong to the same source frame.
                live_status = view['live_status']
                frame = select_display_frame(None if live_status else view['vis'],view['raw'],blank)
                if live_status:
                    frame = frame.copy()
                    cv2.putText(frame, live_status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                .65, (255, 255, 255), 2)
                k = cv2.waitKey(30)  # 30fps + bombeo de mensajes: adiós Not Responding
                if k != -1:
                    lastkey, lastt = k & 0xFF, time.perf_counter()
                    keyq.put(lastkey)
                if lastkey is not None and time.perf_counter() - lastt < 1.5:  # eco: confirma que la tecla entró
                    frame = frame.copy()
                    cv2.putText(frame, f"tecla: {chr(lastkey) if 32 <= lastkey < 127 else lastkey}",
                                (12, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(win, frame)

        if not a.headless:
            th3 = threading.Thread(target=show_loop, daemon=True)
            th3.start()
        n, t_prev, need_init, pose = 0, time.perf_counter(), initial_pose is None, initial_pose
        t_init0 = time.perf_counter()
        t_lat = t_age = 0.0
        state = "INIT"
        sam_due = True  # no detection retries after a delay; errors wait for r
        pending = None
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        sam_pending = None  # SAM en fondo: la llamada bloquea ~8s y sin bombeo el GUI congela
        sam_bd = init_bd = None
        sam_status = None
        sam_retry_at = float('inf')
        init_started_at = None
        sam_elapsed_s = 0.0
        init_elapsed_s = 0.0
        awaiting_validation = False
        catchup_after_frame = None
        catchup_not_before = 0.0
        recovering = False
        register_elapsed_s = 0.0
        while True:
            if a.sam_backend == 'sam3.1' and recovery.future is not None and not recovery.future.done():
                if poll_key() in (ord('q'), 27) or stop.is_set():
                    break
                time.sleep(0.005)
                continue
            try:
                bd = pq.get(timeout=0.005 if need_init and (sam_pending is not None or pending is not None) else 0.2)
            except queue.Empty:
                time.sleep(0.005)
                if stop.is_set():  # worker caído y cola vacía: no hay nada que trackear
                    break
                if need_init and init_bd is not None:
                    bd = init_bd  # poll async init even when paused playback produces no new bundles
                else:
                    if poll_key() in (ord('q'),27):
                        break
                    continue
            if need_init and (sam_pending is not None or pending is not None):
                bd = init_bd  # keep the exact SAM/register image stable while working
            rect, buf, K_rect = bd["rect"], bd["buf"], bd["K_rect"]
            t_pose_start = time.perf_counter()
            t_frame, t_infer = bd["t_frame"], bd["t_infer"]
            if mask0 is None and need_init:  # orden: SAM fresco > fichero > bbox con tecla b
                init_bd = init_bd or bd
                if a.mask_text and sam_pending is None and (sam_due or time.perf_counter() >= sam_retry_at):
                    sam_bd = bd
                    init_bd = bd
                    init_started_at = time.perf_counter()
                    # SAM is independent of stereo capture/inference. Only the
                    # pose registration stage may reserve the GPU below.
                    sam_pending = pool.submit(run_sam_attempt, rect, a.mask_text)
                    sam_due = False
                    sam_retry_at = float('inf')
                    sam_status = None
                    t_init0 = time.perf_counter()
                if sam_pending is not None:
                    view['live_status'] = 'Buscando objeto con SAM... (q=salir)'
                    show = np.ascontiguousarray(cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
                    cv2.putText(show, f"INIT máscara SAM... {time.perf_counter()-t_init0:.0f}s (q=salir)",
                                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    view["vis"], view["seq"] = show, view["seq"] + 1
                    if poll_key() in (ord("q"), 27):
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    if not sam_pending.done():
                        continue
                    result = sam_pending.result()
                    sam_elapsed_s = result.elapsed_s
                    sm, sbox, sscore = result.mask, result.box, result.score
                    sam_pending = None
                    if result.error or (sm is not None and not a.fast_init and not auto_recover):
                        unload_sam()  # release VRAM before pose registration or error recovery
                    if sm is not None:
                        init_bd = sam_bd  # mask, RGB and depth must come from the same frame
                        mask0 = (sm * 255).astype(np.uint8)
                        print(f"[{ts()}] máscara SAM: {mask0.sum()//255} px box={sbox} score={sscore:.2f}", flush=True)
                    else:
                        init_gpu_busy.clear()
                        sam_status = 'SAM error: reintentando (ver terminal)' if result.error else 'SAM sin deteccion: buscando...'
                        print(f"[{ts()}] {sam_status}; texto={a.mask_text!r}; {result.error or 'ninguna mascara supero el umbral'}", flush=True)
                        if a.headless and mask_file is None and not auto_recover:
                            raise RuntimeError(result.error or f'SAM no detecto {a.mask_text!r}')
                        if not result.error or auto_recover:
                            sam_retry_at = time.perf_counter() + 2.0
                        if is_bag and bag_init_paused.is_set():
                            bag_init_paused.clear()
                            playback.resume()
                if mask0 is None and mask_file is not None:
                    mask0 = mask_file
                if mask0 is None:  # bucle vivo: muestra frame, 'b' abre el selector una vez
                    if a.headless and not auto_recover:
                        raise RuntimeError('No se obtuvo máscara de inicialización en modo headless')
                    show = np.ascontiguousarray(cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
                    cv2.putText(show, sam_status or "pulsa b y dibuja el bbox (q=salir)", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    if sam_status:
                        cv2.putText(show, 'r=reintentar SAM | b=bbox manual | q=salir', (12, 56),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    view["vis"], view["seq"] = show, view["seq"] + 1
                    view['live_status'] = sam_status or 'pulsa b y dibuja el bbox (q=salir)'
                    k = poll_key()
                    if k in (ord("q"), 27):
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    if k == ord('r') and a.mask_text:
                        sam_due = True
                        init_bd = None
                    if k == ord("b"):
                        init_gpu_busy.clear()
                        if recovery.future is None or recovery.future.done():
                            unload_sam()
                        freeze.set()
                        try:
                            x, y, w, h = cv2.selectROI(win, cv2.cvtColor(rect, cv2.COLOR_RGB2BGR), False)
                        finally:
                            freeze.clear()
                        if w >= 5 and h >= 5:
                            mask0 = (refine_mask_from_bbox(buf, (x, y, x + w, y + h)) * 255).astype(np.uint8)
                            if (mask0 > 127).any():
                                init_bd = bd
                                print(f"máscara init: {(mask0 > 127).sum()} px", flush=True)
                            else:
                                mask0 = None
                    continue
            if need_init:
                if pending is None and mask0 is not None:
                    t_init0 = time.perf_counter()  # el register va a un hilo; el GUI sigue pintando
                    if init_started_at is None:
                        init_started_at = t_init0
                    init_gpu_busy.set()
                    set_init_phase('preparando',0)
                    init_bd = init_bd or bd
                    pending = pool.submit(estimate_pose, est, init_bd['K_rect'].copy(), init_bd['rect'].copy(),
                                          init_bd['buf'].copy(), mask0 > 127, 5,
                                          reference=a.reference_init, progress=set_init_phase, fast_init=a.fast_init)
                show = np.ascontiguousarray(cv2.cvtColor(init_bd['rect'], cv2.COLOR_RGB2BGR))
                label = f"INIT {init_progress['phase']} {init_progress['hypotheses'] or ''}... {time.perf_counter()-t_init0:.0f}s"
                view['live_status'] = label
                cv2.putText(show, label,
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                view["vis"], view["seq"] = show, view["seq"] + 1
                k = poll_key()
                if k in (ord("q"), 27):
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                if not pending.done():
                    continue
                try:
                    pose, dt, vram = pending.result()
                    if est.pose_last is None:
                        raise RuntimeError('Registro sin pose: comprueba la máscara y la profundidad')
                except Exception as e:
                    init_gpu_busy.clear()
                    print(f"error: init falló: {e}", flush=True)
                    if a.headless and not auto_recover:
                        raise RuntimeError('Falló el registro inicial en modo headless') from e
                    pending = None
                    if auto_recover:
                        mask0 = None
                        init_bd = sam_bd = None
                        init_started_at = None
                        sam_retry_at = time.perf_counter() + a.recovery_retry_s
                        sam_status = 'Registro fallido: buscando otra mascara...'
                        if is_bag and bag_init_paused.is_set():
                            bag_init_paused.clear()
                            playback.resume()
                        continue
                if pending is None or est.pose_last is None:  # sin pose que dibujar
                    init_gpu_busy.clear()
                    state = "INIT_FAIL"
                    k = poll_key()
                    if k in (ord("q"), 27):
                        break
                    continue
                pending = None
                init_gpu_busy.clear()
                register_elapsed_s = dt
                init_elapsed_s = time.perf_counter()-init_started_at
                print(f'[{ts()}] INIT completo: SAM={sam_elapsed_s:.3f}s registro={dt:.3f}s total={init_elapsed_s:.3f}s fallback={est.init_stats["fallback"]}', flush=True)
                state, need_init = "INIT", False
                bd = init_bd
                rect,buf,K_rect = bd['rect'],bd['buf'],bd['K_rect']
                t_frame,t_infer = bd['t_frame'],bd['t_infer']
                init_bd = None
                # A mask/registration belongs to its source frame. Never display
                # it on the current camera image: track a newer bundle first.
                if not (is_bag and bag_init_paused.is_set()):
                    awaiting_validation = True
                    catchup_after_frame = bd['fn'][0]
                    catchup_not_before = time.perf_counter()
                    view['live_status'] = 'Actualizando pose al movimiento actual...'
                    continue
            else:
                if awaiting_validation and (bd['fn'][0] <= catchup_after_frame or bd['t'] < catchup_not_before):
                    continue
                torch.cuda.reset_peak_memory_stats()
                pose, dt = track_pose(est, K_rect, rect, buf, iters=a.track_iters,reference=a.reference_tracking)
                vram = torch.cuda.max_memory_allocated() / 2**20
                state = "TRACK"
            t_pose_end = time.perf_counter()
            color_render = None
            if a.overlay == 'mesh':
                rendered, color_render = render_textured(est, K_rect, *buf.shape, return_depth=True)
            else:
                rendered = render_depth(est,K_rect,*buf.shape)
            t_render_end = time.perf_counter()
            fit = fit_metrics(buf, K_rect, pose, est.diameter, est=est, depth_render=rendered)
            t_fit_end = time.perf_counter()
            valid_pose = fit_is_valid(fit)
            if not valid_pose:
                state = 'LOST'
            elif awaiting_validation:
                state = 'RECOVERED' if recovering else 'INIT'
                print(f'[{ts()}] {state}: validada en frame {bd["fn"][0]} (registro en {catchup_after_frame})', flush=True)
                recovering = False
                recovery.reset()
            awaiting_validation = awaiting_validation and not valid_pose
            candidate = None
            if auto_recover:
                attempts_before = recovery.attempts
                completed_before = recovery.completed
                candidate = recovery.update(valid_pose, bd, time.perf_counter(), pool, a.mask_text)
                if recovery.completed != completed_before and recovery.last_error:
                    print(f'[{ts()}] RECOVERY: {recovery.last_error}; se reintentara', flush=True)
                    if recovery.future is None:
                        unload_sam()  # next attempt recreates a failed worker/model
                if recovery.attempts != attempts_before:
                    print(f'[{ts()}] RECOVERY: SAM en segundo plano, intento {recovery.attempts}', flush=True)
                if candidate is not None:
                    init_bd, result = candidate
                    mask0 = (result.mask * 255).astype(np.uint8)
                    sam_elapsed_s = result.elapsed_s
                    init_started_at = time.perf_counter() - result.elapsed_s
                    need_init = recovering = True
                    recovery.reset()
                    print(f'[{ts()}] RECOVERY: mascara del frame {init_bd["fn"][0]}; registrando', flush=True)
            t_lat = time.perf_counter() - t_frame
            t_age = t_frame - bd["t"]
            fps = 1 / max(time.perf_counter() - t_prev, 1e-6)
            t_prev = time.perf_counter()
            vis, _ = draw_overlay(rect, K_rect, pose, est, bounds,
                                  depth_render=None if a.legacy_costs else rendered, fast=not a.reference_cpu,
                                  alpha=a.overlay_alpha, color_render=color_render)
            t_overlay_end = time.perf_counter()
            hud = f"{state} lat={t_lat:.1f}s edad={t_age:.1f}s fps={fps:.1f} t_est={dt:.2f}s"
            cv2.putText(vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            view["vis"], view["seq"] = np.ascontiguousarray(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)), view["seq"] + 1
            view['live_status'] = ('Recuperando objeto: SAM + pose en segundo plano...'
                                   if not valid_pose and auto_recover else None)
            if is_bag and bag_init_paused.is_set():
                bag_init_paused.clear()
                playback.resume()
            n += 1
            if metrics_file:
                metrics_file.write(json.dumps({'n':n,'state':state,'wall_s':time.perf_counter(),
                    'frame_numbers':bd['fn'],'timestamps_ms':bd['ts'],'fs_model_s':t_infer,'pose_s':dt,
                    'capture_to_publish_s':time.perf_counter()-bd['t'],'fps_hud':fps,'fit':fit,
                    'stages':dict(bd['stages'],bundle_queue_s=t_pose_start-bd['ready'],
                        pose_wall_s=t_pose_end-t_pose_start,render_s=t_render_end-t_pose_end,
                        fit_s=t_fit_end-t_render_end,overlay_s=t_overlay_end-t_fit_end),
                    'pose':np.asarray(pose).reshape(4,4).tolist(),
                    'valid_iters':a.valid_iters,'track_iters':a.track_iters,'legacy_costs':a.legacy_costs,
                    'scale':a.scale,'display_mode':'live_raw_recovery' if not valid_pose and auto_recover else 'synchronized_rgb_pose',
                    'reference_volumes':bd['stages']['fs_details'].get('reference_volumes',True),
                    'reference_cpu':a.reference_cpu,
                    'reference_tracking':a.reference_tracking,
                    'reference_init':a.reference_init,
                    'fast_init':a.fast_init,
                    'sam_backend':a.sam_backend,'sam_threshold':a.sam_threshold,
                    'recovery':dict(enabled=auto_recover, attempts=recovery.attempts,
                                    searching=recovery.future is not None, registering=need_init,
                                    validated_from_frame=catchup_after_frame if state in ('INIT','RECOVERED') else None,
                                    last_error=recovery.last_error),
                    'initialization':dict(sam_s=sam_elapsed_s, register_s=register_elapsed_s, total_s=init_elapsed_s,
                                          sam_prepare_s=sam_prepare_s, pose_prepare_s=pose_prepare_s,
                                          **est.init_stats) if state in ('INIT','RECOVERED') and hasattr(est, 'init_stats') else None,
                    'fp_reserved_mib':torch.cuda.memory_reserved()/2**20,
                    'sorted_zbuffer':a.sorted_zbuffer})+'\n')
                metrics_file.flush()
            if a.snapshots and (n == 1 or n % 20 == 0):
                cv2.imwrite(str(a.snapshots/f'{n:05d}_frame_{bd["fn"][0]:06d}.jpg'),view['vis'])
            print(f"f{n} {state} lat={t_lat:.1f}s edad={t_age:.1f}s fps={fps:.1f} "
                  f"t_pose={dt:.2f}s frac={fit['valid_frac']} resid={fit['resid_med_m']}", flush=True)
            k = poll_key()
            if k == ord("r"):
                recovery.reset()
                recovering = True
                awaiting_validation = False
                need_init, mask0, sam_due = True, None, True  # re-adquiere (SAM > fichero > bbox)
                init_bd = sam_bd = None
                init_started_at = None
                sam_elapsed_s = 0.0
                sam_status = None
                sam_retry_at = float('inf')
                view['vis'] = None
            if k == ord("s"):
                dst = Path(a.out) / f"live_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                dst.mkdir(parents=True, exist_ok=True)  # ponytail: dos `s` en el mismo segundo colisionaban
                cv2.imwrite(str(dst / "color_rect.png"), cv2.cvtColor(rect, cv2.COLOR_RGB2BGR))
                np.save(dst / "depth_on_rgb.npy", buf)
                np.save(dst / "pose_4x4.npy", pose)
                (dst / "meta.json").write_text(json.dumps(
                    {"K_rect": K_rect.tolist(), "state": state, "t_infer_s": t_infer,
                     "t_pose_s": round(dt, 3), "t_lat_s": round(t_lat, 3), "frame_age_s": round(t_age, 3),
                     "timestamps_ms": bd["ts"], "ts_domain": bd["ts_domain"],
                     "frame_numbers": bd["fn"], "fit": fit}, indent=2))
                print(f"guardada {dst}")
            if k in (ord("q"), 27) or (a.frames and n >= a.frames):
                break
    finally:
        stop.set()
        worker.close()  # unblock pending pipe reads before joining inference thread
        if pool is not None:
            # Do not unload a model while an in-flight SAM/register uses it.
            pool.shutdown(wait=True, cancel_futures=True)
        unload_sam()
        for t in (th, th2, th3):
            if t is not None:
                t.join(timeout=5)
        if started:
            try:
                pipe.stop()
            except RuntimeError:
                pass
        if metrics_file:
            metrics_file.close()
        cv2.destroyAllWindows()
    print("live_pose cerrado")


if __name__ == "__main__":
    main()
