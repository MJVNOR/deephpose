"""Measure resident SAM plus fast registration with an idle, loaded FS worker."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import torch
from pose_model import make_estimator, load_textured_mesh, estimate_pose, track_pose, fit_metrics, prepare_fast_pose
from sam_mask import prepare_sam, sam_mask, configure_sam, unload_sam
sys.path.insert(0, str(Path('scripts').resolve()))
from fs_transport import FSClient


def main():
    configure_sam('sam3')  # Preserve the historical fast-init comparison.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    source = Path('data/processed/replay_fs_sweep_20261010/fs4')
    manifest = json.loads((source/'manifest.json').read_text())
    mesh = load_textured_mesh('data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    est = make_estimator(mesh, Path('pose/FoundationPose'), a.out/'debug')
    logging.getLogger().setLevel(logging.WARNING)
    prepared_s = prepare_sam('computer mouse')
    print('sam_prepare_s', prepared_s, flush=True)
    with np.load(source/manifest['frames'][0]['bundle']) as b:
        pose_prepared_s = prepare_fast_pose(est, b['K'], b['depth'].shape)
    worker = FSClient(Path.cwd())
    rows = []
    def save():
        (a.out/'summary.json').write_text(json.dumps(dict(sam_prepare_s=prepared_s,
            pose_prepare_s=pose_prepared_s, rows=rows), indent=2))
    try:
        with np.load('data/processed/replay_20261010_fixed_v3/raw/000060.npz') as raw:
            for _ in range(3):
                worker.infer(raw['left'], raw['right'], .5, 4)
        for index in (0, 24, 48, 80, 112, 160):
            entry = manifest['frames'][index]
            with np.load(source/entry['bundle']) as b:
                rgb, depth, K = b['rgb'], b['depth'], b['K']
            fast_poses = []
            frame_rows = []
            for repeat in range(2):
                torch.cuda.synchronize(); start = time.perf_counter()
                mask, box, score = sam_mask(rgb, 'computer mouse')
                torch.cuda.synchronize(); sam_seconds = time.perf_counter()-start
                if mask is None:
                    row = dict(frame=entry['frame'], repeat=repeat, detected=False, sam_s=sam_seconds)
                    rows.append(row); print(row, flush=True); save()
                    break
                pose, seconds, _ = estimate_pose(est, K, rgb, depth, mask, 5, fast_init=True)
                total = time.perf_counter()-start
                row = dict(frame=entry['frame'], repeat=repeat, detected=True, sam_s=sam_seconds, register_s=seconds,
                    total_s=total, score=score, init_stats=est.init_stats,
                    reserved_mib=torch.cuda.memory_reserved()/2**20, pose=pose.tolist())
                rows.append(row); fast_poses.append(pose.copy())
                frame_rows.append(row)
                last_mask = mask
                print({k:v for k,v in row.items() if k!='pose'}, flush=True)
                save()
            if not fast_poses:
                continue
            centered_fast_pose = est.pose_last.clone()
            reference, reference_s, _ = estimate_pose(est, K, rgb, depth, last_mask, 5)
            for row, pose in zip(frame_rows, fast_poses):
                delta = pose[:3,:3] @ reference[:3,:3].T
                row.update(reference_s=reference_s,
                    angle_deg=float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1)))),
                    translation_mm=float(np.linalg.norm(pose[:3,3]-reference[:3,3])*1000))
            est.pose_last = centered_fast_pose
            tracking = []
            for next_entry in manifest['frames'][index+1:index+6]:
                with np.load(source/next_entry['bundle']) as b:
                    pose, seconds = track_pose(est, b['K'], b['rgb'], b['depth'], 5)
                    fit = fit_metrics(b['depth'], b['K'], pose, est.diameter, est=est)
                    tracking.append(dict(frame=next_entry['frame'], seconds=seconds, fit=fit))
            frame_rows[-1]['following_tracking'] = tracking
            save()
    finally:
        worker.close()
        unload_sam()


if __name__ == '__main__':
    main()
