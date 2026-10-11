"""Extract a fixed RealSense sequence and cache FS bundles in the main env.

Playback runs non-real-time; inference happens AFTER capture finishes, so
model loading cannot consume the recording. Every requested ID is verified. Existing
outputs are never overwritten. The immutable manifest is the replay input.
"""
import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from capture import intr_to_dict
from replay_selection import selected_frame, verify_sequence


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--start-frame', type=int, default=60)
    p.add_argument('--stride', type=int, default=15)
    p.add_argument('--scale', type=float, default=0.5)
    p.add_argument('--valid-iters', type=int, default=16)
    p.add_argument('--dt-max', type=float, default=33.)
    a = p.parse_args()
    selected_frame(a.start_frame, a.start_frame, a.stride)
    if not a.bag.is_file() or not 0 < a.scale <= 1 or a.valid_iters < 1:
        p.error('Invalid bag, scale or iteration count')
    a.out.mkdir(parents=True, exist_ok=False)
    raw = a.out / 'raw'
    raw.mkdir()
    cfg, pipe = rs.config(), rs.pipeline()
    cfg.enable_device_from_file(str(a.bag.resolve()), repeat_playback=False)
    for stream, index, fmt in [(rs.stream.infrared, 1, rs.format.y8),
                               (rs.stream.infrared, 2, rs.format.y8),
                               (rs.stream.depth, 0, rs.format.z16),
                               (rs.stream.color, 0, rs.format.rgb8)]:
        cfg.enable_stream(stream, index, 640, 480, fmt, 30)
    resolved = cfg.resolve(pipe)
    ir1 = resolved.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
    ir2 = resolved.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
    rgb = resolved.get_stream(rs.stream.color).as_video_stream_profile()
    ex = ir1.get_extrinsics_to(rgb)
    cal = {'ir1': intr_to_dict(ir1.get_intrinsics()), 'ir2': intr_to_dict(ir2.get_intrinsics()),
           'color': intr_to_dict(rgb.get_intrinsics()),
           'baseline_m': ir1.get_extrinsics_to(ir2).translation[0],
           'extrinsics_ir1_to_color': {'rotation': list(ex.rotation), 'translation': list(ex.translation)}}
    rows, errors, seen = [], [], set()
    lock = threading.Lock()
    received = threading.Event()

    def capture(frame):
        if not received.is_set():
            print(f'First playback frameset: {frame.is_frameset()}', flush=True)
        received.set()
        try:
            if not frame.is_frameset():
                return
            fs = frame.as_frameset()
            left, right, color = fs.get_infrared_frame(1), fs.get_infrared_frame(2), fs.get_color_frame()
            if not left or not right or not color:
                return
            n = left.get_frame_number()
            if not selected_frame(n, a.start_frame, a.stride):
                return
            timestamps = [f.get_timestamp() for f in (left, right, color)]
            if max(timestamps) - min(timestamps) > a.dt_max or right.get_frame_number() != n:
                return  # verification below fails if a requested frame was not synchronized
            with lock:
                if n in seen:
                    return
                seen.add(n)
                path = raw / f'{n:06d}.npz'
                np.savez(path, left=np.asanyarray(left.get_data()), right=np.asanyarray(right.get_data()),
                         rgb=np.asanyarray(color.get_data()))
                rows.append({'frame': n, 'frame_numbers': [f.get_frame_number() for f in (left, right, color)],
                             'timestamps_ms': timestamps, 'raw': str(path.relative_to(a.out))})
        except Exception as exc:
            errors.append(repr(exc))

    started = False
    try:
        profile = pipe.start(cfg)
        started = True
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        deadline = time.monotonic() + 180
        while True:
            if errors or time.monotonic() > deadline:
                raise RuntimeError(errors or 'Playback extraction timed out')
            try:
                frameset = pipe.wait_for_frames(timeout_ms=2000)
            except RuntimeError:
                if playback.current_status() == rs.playback_status.stopped:
                    break
                raise
            capture(frameset)
    finally:
        if started:
            pipe.stop()
    if errors:
        raise RuntimeError(errors)
    rows.sort(key=lambda row: row['frame'])
    verify_sequence([row['frame'] for row in rows], a.start_frame, a.stride)
    print(f'Extracted {len(rows)} fixed frames: {rows[0]["frame"]}..{rows[-1]["frame"]}', flush=True)
    (a.out / 'capture.json').write_text(json.dumps({'calibration': cal, 'frames': rows}, indent=2))

    from config import load
    from fs_model import load_model, infer_disp
    from disparity_to_depth import check_stereo_cal, disp_to_depth
    from project_to_rgb import project_ir_to_rgb
    ckpt = load()['model']['ckpt']
    model, model_cfg, _ = load_model(ckpt, a.valid_iters)
    _, baseline = check_stereo_cal(cal)
    K = np.array([[cal['ir1']['fx'], 0, cal['ir1']['ppx']],
                  [0, cal['ir1']['fy'], cal['ir1']['ppy']], [0, 0, 1]])
    with np.load(a.out / rows[0]['raw']) as first:
        for _ in range(2):
            infer_disp(model, model_cfg, first['left'], first['right'], a.scale, a.valid_iters)
    for i, row in enumerate(rows):
        with np.load(a.out / row['raw']) as data:
            left, right, color = data['left'], data['right'], data['rgb']
        t0 = time.perf_counter()
        disp, info = infer_disp(model, model_cfg, left, right, a.scale, a.valid_iters)
        scale_x = disp.shape[1] / left.shape[1]
        if disp.shape != left.shape:
            disp = cv2.resize(disp, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_LINEAR) / scale_x
        depth, valid = disp_to_depth(disp, K[0, 0], baseline)
        buf, _, kr, rect, _ = project_ir_to_rgb(depth, valid, K, ex.rotation, ex.translation, color, cal['color'])
        row['fs_and_projection_s'] = time.perf_counter() - t0
        row['fs_info'] = info
        path = a.out / f'bundle_{row["frame"]:06d}.npz'
        np.savez(path, rgb=rect, depth=buf, K=kr)
        row['bundle'] = path.name
        row['sha256'] = sha256(path)
        if i % 20 == 0:
            print(f'FS bundle {i+1}/{len(rows)}', flush=True)
    manifest = {'mode': 'fixed-frame offline replay; no dropping during inference',
                'bag': str(a.bag.resolve()), 'bag_sha256': sha256(a.bag),
                'start_frame': a.start_frame, 'stride': a.stride, 'scale': a.scale,
                'valid_iters': a.valid_iters, 'checkpoint_sha256': sha256(ckpt),
                'source_sha256': {f: sha256(f) for f in ['scripts/prepare_replay.py', 'scripts/fs_model.py', 'scripts/project_to_rgb.py']},
                'calibration': cal, 'frames': rows}
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Complete: {a.out / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
