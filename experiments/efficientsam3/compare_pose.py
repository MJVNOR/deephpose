"""Evaluate the effect of changed masks using identical FoundationPose settings."""
import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'pose'))
from pose_model import make_estimator,load_textured_mesh,estimate_pose,fit_metrics,track_pose


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data/processed/replay_fs_sweep_20261010/fs4'
    manifest=json.loads((source/'manifest.json').read_text())
    candidate=ROOT/'reports/efficientsam3_ev_m_thr01_20261011'
    reference=ROOT/'reports/sam3_reference_20261011'
    est=make_estimator(load_textured_mesh(ROOT/'data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj'),
                       ROOT/'pose/FoundationPose',a.out/'debug')
    logging.getLogger().setLevel(logging.WARNING)
    rows=[]
    for index in range(0,161,8):
        entry=manifest['frames'][index]; name=f"mask_{entry['frame']}.npy"
        if not (candidate/name).exists() or not (reference/name).exists():
            rows.append(dict(frame=entry['frame'],compared=False));continue
        with np.load(source/entry['bundle']) as b:
            rgb,depth,K=b['rgb'],b['depth'],b['K']
        poses=[]; measurements=[]
        for folder in (reference,candidate):
            mask=np.load(folder/name)
            pose,seconds,_=estimate_pose(est,K,rgb,depth,mask,5,fast_init=True)
            poses.append(pose.copy())
            measurements.append(dict(seconds=seconds,fit=fit_metrics(depth,K,pose,est.diameter,est=est),init_stats=est.init_stats))
        delta=poses[1][:3,:3]@poses[0][:3,:3].T
        row=dict(frame=entry['frame'],compared=True,reference=measurements[0],candidate=measurements[1],
            translation_mm=float(np.linalg.norm(poses[1][:3,3]-poses[0][:3,3])*1000),
            angle_deg=float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1)))),
            reference_pose=poses[0].tolist(),candidate_pose=poses[1].tolist(),tracking=[])
        for following in manifest['frames'][index+1:index+4]:
            with np.load(source/following['bundle']) as b:
                pose,seconds=track_pose(est,b['K'],b['rgb'],b['depth'],5)
                row['tracking'].append(dict(frame=following['frame'],fit=fit_metrics(b['depth'],b['K'],pose,est.diameter,est=est)))
        rows.append(row);print({k:v for k,v in row.items() if 'pose' not in k and k!='tracking'},flush=True)
        (a.out/'summary.json').write_text(json.dumps(rows,indent=2))
    (a.out/'summary.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
