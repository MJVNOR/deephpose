"""Render old/new display boxes using one unchanged saved pose and RGB-D."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pose_model import load_textured_mesh, make_estimator, render_depth, draw_overlay


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / 'reports/base_aligned_box_20261011'
    out.mkdir(exist_ok=True)
    source = root / 'data/processed/replay_fs_sweep_20261010/fs4'
    entry = json.loads((source/'manifest.json').read_text())['frames'][0]
    with np.load(source/entry['bundle']) as b:
        rgb, depth, K = b['rgb'], b['depth'], b['K']
    pose = np.asarray(json.loads((root/'reports/efficientsam3_pose_20261011/summary.json').read_text())[0]['reference_pose'])
    mesh = load_textured_mesh(root/'data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    est = make_estimator(mesh, root/'pose/FoundationPose', out/'debug')
    est.pose_last = torch.as_tensor(pose @ np.linalg.inv(est.get_tf_to_centered_mesh().cpu().numpy()), device='cuda', dtype=torch.float32)
    rendered = render_depth(est, K, *depth.shape)
    box = est.display_box
    est.display_box = None
    before, _ = draw_overlay(rgb, K, pose, est, mesh.bounds, depth_render=rendered, fast=True)
    est.display_box = box
    after, _ = draw_overlay(rgb, K, pose, est, mesh.bounds, depth_render=rendered, fast=True)
    for label, img in [('Antes: ejes OBJ', before), ('Ahora: plano de la base', after)]:
        cv2.putText(img, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    cv2.imwrite(str(out/'comparison.jpg'), cv2.cvtColor(np.concatenate([before,after],axis=1),cv2.COLOR_RGB2BGR))
    metrics = dict(to_object=box.to_object.tolist(), bounds=box.bounds.tolist(),
                   extents_mm=((box.bounds[1]-box.bounds[0])*1000).tolist(),
                   planar_area_fraction=box.planar_area_fraction,
                   tilt_from_obj_y_deg=float(np.degrees(np.arccos(abs(box.to_object[1,2])))))
    (out/'geometry.json').write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == '__main__':
    main()
