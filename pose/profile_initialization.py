"""Measure real registration phases with the fixed SAM mask, without camera motion."""
import argparse
import json
import logging
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from pose_model import load_textured_mesh,make_estimator,estimate_pose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source=Path('data/processed/replay_fs_sweep_20261010/fs4')
    meta=json.loads((source/'manifest.json').read_text())
    with np.load(source/meta['frames'][0]['bundle']) as b:rgb,depth,K=b['rgb'],b['depth'],b['K']
    mask=cv2.imread('reports/replay_comparison_20261010_v2/mask_init.png',0)>127
    mesh=load_textured_mesh('data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    est=make_estimator(mesh,Path('pose/FoundationPose'),a.out/'debug')
    logging.getLogger().setLevel(logging.WARNING)
    stages={}
    def timed(label,method):
        def call(*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter()
            result=method(*args,**kwargs)
            torch.cuda.synchronize();stages[label]=time.perf_counter()-start
            return result
        return call
    est.refiner.predict=timed('refine_s',est.refiner.predict)
    est.scorer.predict=timed('score_s',est.scorer.predict)
    rows=[]
    for repeat in range(3):
        stages.clear();pose,seconds,vram=estimate_pose(est,K,rgb,depth,mask,5,reference=True)
        rows.append(dict(repeat=repeat,total_s=seconds,**stages,pose=pose.tolist(),hypotheses=len(est.poses)))
        print(rows[-1],flush=True)
    (a.out/'summary.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
