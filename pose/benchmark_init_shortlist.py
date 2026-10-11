"""Compare experimental global search against full registration using saved SAM masks."""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch
from pose_model import make_estimator, load_textured_mesh, estimate_pose, depth_to_fp, fit_metrics, render_depth
from initialization_fast import register_with_shortlist
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    source = Path('data/processed/replay_fs_sweep_20261010/fs4')
    manifest = json.loads((source/'manifest.json').read_text())
    mesh = load_textured_mesh('data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    est = make_estimator(mesh, Path('pose/FoundationPose'), a.out/'debug')
    logging.getLogger().setLevel(logging.WARNING)
    rows = []
    for index in (0, 80, 160):
        entry = manifest['frames'][index]
        with np.load(source/entry['bundle']) as b:
            rgb, depth, K = b['rgb'], b['depth'], b['K']
        mask = np.load(f"reports/sam_one_second_profile_20261010/mask_{entry['frame']}.npy")
        reference = None
        for hypotheses in (252, 84, 42, 84, 252):
            torch.cuda.synchronize(); start = time.perf_counter()
            if hypotheses == 252:
                pose, _, _ = estimate_pose(est, K, rgb, depth, mask, 5)
            else:
                pose = register_with_shortlist(est, hypotheses=hypotheses, finalists=8,
                    K=np.asarray(K, float), rgb=rgb, depth=depth_to_fp(depth), ob_mask=mask, iteration=5)
            torch.cuda.synchronize(); seconds = time.perf_counter()-start
            if reference is None:
                reference = pose.copy()
            delta = pose[:3,:3] @ reference[:3,:3].T
            angle = float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2, -1, 1))))
            rendered = render_depth(est, K, *depth.shape)
            sil = rendered > .001
            row = dict(frame=entry['frame'], hypotheses=hypotheses, seconds=seconds,
                angle_deg=angle, translation_mm=float(np.linalg.norm(pose[:3,3]-reference[:3,3])*1000),
                mask_iou=float((sil & mask).sum()/max(1,(sil | mask).sum())),
                fit=fit_metrics(depth, K, pose, est.diameter, est=est, depth_render=rendered), pose=pose.tolist())
            rows.append(row); print({k:v for k,v in row.items() if k!='pose'}, flush=True)
            (a.out/'summary.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
