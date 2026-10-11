"""Paired registration comparison, including simulated continuous FS competition."""
import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from pose_model import load_textured_mesh,make_estimator,estimate_pose,render_depth
sys.path.insert(0,str(Path('scripts').resolve()))
from fs_transport import FSClient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source=Path('data/processed/replay_fs_sweep_20261010/fs4')
    manifest=json.loads((source/'manifest.json').read_text())
    mesh=load_textured_mesh('data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    est=make_estimator(mesh,Path('pose/FoundationPose'),a.out/'debug')
    logging.getLogger().setLevel(logging.WARNING)
    pose_rows={r['frame']:np.asarray(r['pose']) for r in map(json.loads,Path('reports/fs_pose_comparison_20261010/fs4/iters5_repeat1/frames.jsonl').read_text().splitlines())}
    rows=[]
    for index in [0,80,160]:
        entry=manifest['frames'][index]
        with np.load(source/entry['bundle']) as b:rgb,depth,K=b['rgb'],b['depth'],b['K']
        if index==0:
            mask=cv2.imread('reports/replay_comparison_20261010_v2/mask_init.png',0)>127
        else:
            centered=pose_rows[entry['frame']]@np.linalg.inv(est.get_tf_to_centered_mesh().cpu().numpy())
            est.pose_last=torch.as_tensor(centered,device='cuda',dtype=torch.float32)
            mask=render_depth(est,K,*depth.shape)>0.001
        reference_arrays=None
        for reference in [True,False,False,True]:
            pose,seconds,_=estimate_pose(est,K,rgb,depth,mask,5,reference=reference)
            arrays=[pose,est.poses.detach().cpu().numpy(),est.scores.detach().cpu().numpy()]
            if reference_arrays is None:reference_arrays=[v.copy() for v in arrays]
            for original,current in zip(reference_arrays,arrays):np.testing.assert_array_equal(original,current)
            row=dict(frame=entry['frame'],reference=reference,seconds=seconds,exact=True,hypotheses=len(est.poses))
            rows.append(row);print(row,flush=True)
    with np.load(source/manifest['frames'][0]['bundle']) as b:rgb,depth,K=b['rgb'],b['depth'],b['K']
    mask=cv2.imread('reports/replay_comparison_20261010_v2/mask_init.png',0)>127
    with np.load('data/processed/replay_20261010_fixed_v3/raw/000060.npz') as raw:left,right=raw['left'],raw['right']
    worker=FSClient(Path.cwd());stop=threading.Event();allow=threading.Event();idle=threading.Event();idle.set()
    errors=[];thread=None;concurrent=[]
    try:
        for _ in range(3):worker.infer(left,right,.5,4)
        def fs_loop():
            try:
                while not stop.is_set():
                    if not allow.wait(.05):continue
                    idle.clear()
                    worker.infer(left,right,.5,4)
                    idle.set()
            except Exception as exc:errors.append(str(exc))
            finally:idle.set()
        thread=threading.Thread(target=fs_loop,daemon=True);thread.start()
        baseline_pose=None
        for reference in [True,False,False,True]:
            allow.set();time.sleep(.5)
            if not reference:allow.clear()  # current FS request may finish, as in the live pipeline
            pose,seconds,_=estimate_pose(est,K,rgb,depth,mask,5,reference=reference)
            if baseline_pose is None:baseline_pose=pose.copy()
            np.testing.assert_array_equal(baseline_pose,pose)
            row=dict(reference=reference,seconds=seconds,exact_pose=True)
            concurrent.append(row);print('Concurrent',row,flush=True)
            allow.clear();idle.wait(10)
        if errors:raise RuntimeError(errors)
    finally:
        stop.set();allow.set()
        if thread:thread.join(timeout=10)
        worker.close()
    result={'isolated':rows,'concurrent_fs':concurrent,
            'note':'FS competition uses repeated recorded IR input, not a physical camera. Optimized runs pause further FS requests but let the in-flight one finish. Fixed SAM mask excludes SAM latency. Other masks are CAD projections, not ground truth.'}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
