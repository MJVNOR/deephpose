"""Preview the live CAD overlay on a saved RGB/pose pair."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from pose_model import load_textured_mesh, make_estimator, render_depth, render_textured, draw_overlay


def main():
    root = Path(__file__).resolve().parents[1]
    out = root/'reports/mesh_overlay_20261011'
    out.mkdir(exist_ok=True)
    source = root/'data/processed/replay_fs_sweep_20261010/fs4'
    entry = json.loads((source/'manifest.json').read_text())['frames'][0]
    with np.load(source/entry['bundle']) as b:
        rgb, depth, K = b['rgb'], b['depth'], b['K']
    pose = np.asarray(json.loads((root/'reports/efficientsam3_pose_20261011/summary.json').read_text())[0]['reference_pose'])
    mesh = load_textured_mesh(root/'data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    est = make_estimator(mesh, root/'pose/FoundationPose', out/'debug')
    est.pose_last = torch.as_tensor(pose @ np.linalg.inv(est.get_tf_to_centered_mesh().cpu().numpy()), device='cuda', dtype=torch.float32)
    rendered, colors = render_textured(est, K, *depth.shape, return_depth=True)
    np.testing.assert_allclose(rendered, render_depth(est, K, *depth.shape), atol=1e-6)
    preview, sil = draw_overlay(rgb, K, pose, est, mesh.bounds, depth_render=rendered,
                               color_render=colors, alpha=.65, fast=True)
    assert sil.any() and np.any(colors[sil])
    cv2.imwrite(str(out/'overlay.jpg'), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out/'comparison.jpg'), cv2.cvtColor(np.concatenate([rgb, preview], axis=1), cv2.COLOR_RGB2BGR))
    print(f'PASS: shared depth matches; {sil.sum()} CAD pixels. Preview: {out}')


if __name__ == '__main__':
    main()
