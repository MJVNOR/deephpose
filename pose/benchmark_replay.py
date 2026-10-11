"""Compare tracking iterations on identical cached bundles and initial pose.

Run from the repository root in the pose env. This is offline tracking cost,
not end-to-end live FPS. No frame is skipped while loading or initializing.
"""
import argparse
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lost(fit):
    residual = fit['resid_med_m']
    return fit['valid_frac'] < .3 or residual is None or not np.isfinite(residual) or residual > .02


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mask-text', default='computer mouse')
    p.add_argument('--track-iters', nargs='+', type=int, default=[1, 2, 5])
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--initial-state', type=Path, help='Reuse saved initial_pose.npy and mask_init.png; isolate depth/tracking changes')
    a = p.parse_args()
    if a.repeats < 1 or any(i < 1 for i in a.track_iters):
        p.error('Iterations and repeats must be positive')
    manifest = json.loads((a.replay / 'manifest.json').read_text())
    frames = manifest['frames']
    if len(frames) < 2:
        p.error('Replay must contain at least two frames')
    # Verify all inputs before allocating GPU models. Load once: exclude disk I/O
    # from timing and ensure every pass uses exactly the same arrays.
    data = []
    for row in frames:
        path = a.replay / row['bundle']
        if digest(path) != row['sha256']:
            raise ValueError(f'Bundle changed: {path}')
        with np.load(path) as bundle:
            data.append((bundle['rgb'], bundle['depth'], bundle['K']))
    a.out.mkdir(parents=True, exist_ok=False)
    from pose_model import load_textured_mesh, make_estimator, estimate_pose, track_pose, fit_metrics, draw_overlay
    from sam_mask import sam_mask, unload_sam
    logging.getLogger().setLevel(logging.WARNING)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    mesh = load_textured_mesh(a.mesh)
    est = make_estimator(mesh, Path(__file__).resolve().parent / 'FoundationPose', a.out / 'debug', debug=0)
    logging.getLogger().setLevel(logging.WARNING)
    rgb, depth, K = data[0]
    if a.initial_state:
        old = json.loads((a.initial_state/'manifest.json').read_text())
        expected_source = manifest.get('source_manifest_sha256',digest(a.replay/'manifest.json'))
        if old['manifest_sha256'] != expected_source:
            raise ValueError('Initial state belongs to a different source replay')
        if old['frame_ids'][0] != frames[0]['frame'] or old['mesh_sha256'] != digest(a.mesh):
            raise ValueError('Initial state frame or mesh does not match')
        mask = cv2.imread(str(a.initial_state/'mask_init.png'),0)
        if mask is None or mask.shape != depth.shape:
            raise ValueError('Invalid initial mask')
        mask = mask > 127
        sam_s,box,score = 0.,old['sam_box'],old['sam_score']
    else:
        print('SAM on fixed initial frame', frames[0]['frame'], flush=True)
        t0 = time.perf_counter()
        mask, box, score = sam_mask(rgb, a.mask_text)
        sam_s = time.perf_counter() - t0
        unload_sam()
    if mask is None or not mask.any():
        raise RuntimeError('SAM found no object on fixed frame; no automatic frame substitution')
    cv2.imwrite(str(a.out / 'mask_init.png'), mask.astype(np.uint8) * 255)
    if a.initial_state:
        initial_pose = np.load(a.initial_state/'initial_pose.npy')
        if initial_pose.shape != (4,4) or not np.isfinite(initial_pose).all():
            raise ValueError('Invalid initial pose')
        centered = initial_pose @ np.linalg.inv(est.get_tf_to_centered_mesh().detach().cpu().numpy())
        est.pose_last = torch.as_tensor(centered,device='cuda',dtype=torch.float32)
        init_s = 0.
    else:
        initial_pose, init_s, _ = estimate_pose(est, K, rgb, depth, mask, iters=5)
    if est.pose_last is None:
        raise RuntimeError('Registration failed')
    initial_centered = est.pose_last.detach().clone()
    initial_fit = fit_metrics(depth, K, initial_pose, est.diameter, est=est)
    if lost(initial_fit):
        raise RuntimeError(f'Initial registration fails fit thresholds: {initial_fit}')
    np.save(a.out / 'initial_pose.npy', initial_pose)
    bounds = np.asarray(mesh.bounds)
    vis, _ = draw_overlay(rgb, K, initial_pose, est, bounds)
    cv2.imwrite(str(a.out / 'initial_overlay.jpg'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    meta = {'mode': 'offline tracking; FS cached; no real-time drops; identical initial pose',
            'initial_state': str(a.initial_state) if a.initial_state else None,
            'manifest_sha256': digest(a.replay / 'manifest.json'), 'mesh_sha256': digest(a.mesh),
            'source_sha256': {str(f): digest(f) for f in [Path(__file__), Path('pose/pose_model.py')]},
            'frame_ids': [r['frame'] for r in frames], 'mask_text': a.mask_text,
            'sam_s': sam_s, 'sam_box': box, 'sam_score': score, 'init_s': init_s,
            'initial_fit': initial_fit, 'track_iters': a.track_iters, 'repeats': a.repeats,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'thresholds': {'valid_frac_min': .3, 'resid_max_m': .02},
            'warmup': '3 tracks on first frame per variant, restoring pose before warmup and measured sequence',
            'numerics': 'seed 0; fixed inputs/state, no guarantee of bitwise deterministic CUDA kernels'}
    (a.out / 'manifest.json').write_text(json.dumps(meta, indent=2))
    summaries = []
    for repeat in range(a.repeats):
        order = a.track_iters if repeat % 2 == 0 else list(reversed(a.track_iters))
        for iters in order:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            for _ in range(3):
                est.pose_last = initial_centered.clone()
                track_pose(est, K, rgb, depth, iters=iters)
            est.pose_last = initial_centered.clone()
            variant = a.out / f'iters{iters}_repeat{repeat+1}'
            variant.mkdir()
            rows = []
            with (variant / 'frames.jsonl').open('w') as output:
                for i, ((frame_rgb, frame_depth, frame_K), source) in enumerate(zip(data[1:], frames[1:]), 1):
                    pose, seconds = track_pose(est, frame_K, frame_rgb, frame_depth, iters=iters)
                    metric_start = time.perf_counter()
                    fit = fit_metrics(frame_depth, frame_K, pose, est.diameter, est=est)
                    metrics_s = time.perf_counter() - metric_start
                    row = {'frame': source['frame'], 'timestamps_ms': source['timestamps_ms'],
                           'pose_s': seconds, 'metrics_s': metrics_s,
                           'state': 'LOST' if lost(fit) else 'TRACK', 'fit': fit, 'pose': pose.tolist()}
                    # Save the same frames in every variant; visualization excluded from timings.
                    if i % 20 == 0 or i == len(frames) - 1:
                        vis, _ = draw_overlay(frame_rgb, frame_K, pose, est, bounds)
                        cv2.imwrite(str(variant / f'frame_{source["frame"]:06d}.jpg'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                    output.write(json.dumps(row) + '\n')
                    rows.append(row)
                    if i % 40 == 0:
                        print(f'iters={iters} repeat={repeat+1}: {i}/{len(frames)-1}', flush=True)
            times = np.array([r['pose_s'] for r in rows])
            residuals = [r['fit']['resid_med_m'] for r in rows if r['fit']['resid_med_m'] is not None]
            summary = {'iters': iters, 'repeat': repeat+1, 'updates': len(rows),
                       'lost': sum(r['state'] == 'LOST' for r in rows),
                       'pose_ms_p50': float(np.median(times)*1000), 'pose_ms_p95': float(np.percentile(times,95)*1000),
                       'pose_only_hz': float(len(times)/times.sum()),
                       'residual_mm_p50': float(np.median(residuals)*1000) if residuals else None,
                       'residual_mm_p95': float(np.percentile(residuals,95)*1000) if residuals else None,
                       'first_lost_frame': next((r['frame'] for r in rows if r['state']=='LOST'),None)}
            (variant / 'summary.json').write_text(json.dumps(summary, indent=2))
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (a.out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print('Complete:', a.out, flush=True)


if __name__ == '__main__':
    main()
