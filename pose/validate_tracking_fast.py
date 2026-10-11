"""ABBA tracking on fixed FS4 bundles, common state and exact pose comparison."""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import torch
from pose_model import load_textured_mesh,make_estimator,track_pose,fit_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    replay=Path('data/processed/replay_fs_sweep_20261010/fs4')
    manifest=json.loads((replay/'manifest.json').read_text())
    data=[]
    for frame in manifest['frames']:
        with np.load(replay/frame['bundle']) as b:data.append((b['rgb'],b['depth'],b['K']))
    mesh=load_textured_mesh('data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    est=make_estimator(mesh,Path('pose/FoundationPose'),a.out/'debug')
    logging.getLogger().setLevel(logging.WARNING)
    initial=np.load('reports/replay_comparison_20261010_v2/initial_pose.npy')
    centered=torch.as_tensor(initial@np.linalg.inv(est.get_tf_to_centered_mesh().cpu().numpy()),device='cuda',dtype=torch.float32)
    summaries=[];poses={}
    for label,reference in [('reference1',True),('candidate1',False),('candidate2',False),('reference2',True)]:
        torch.manual_seed(0)
        rgb,depth,K=data[0]
        for _ in range(3):
            est.pose_last=centered.clone();track_pose(est,K,rgb,depth,5,reference=reference)
        est.pose_last=centered.clone();rows=[];outputs=[]
        for frame,(rgb,depth,K) in zip(manifest['frames'][1:],data[1:]):
            pose,seconds=track_pose(est,K,rgb,depth,5,reference=reference)
            fit=fit_metrics(depth,K,pose,est.diameter,est=est)
            lost=fit['valid_frac']<.3 or fit['resid_med_m'] is None or fit['resid_med_m']>.02
            rows.append({'frame':frame['frame'],'track_ms':seconds*1000,'lost':lost,'fit':fit,
                         'reserved_mib':torch.cuda.memory_reserved()/2**20,'pose':pose.tolist()})
            outputs.append(pose)
        poses[label]=np.asarray(outputs)
        (a.out/f'{label}.json').write_text(json.dumps(rows,indent=2))
        result={'label':label,'frames':len(rows),'lost':sum(r['lost'] for r in rows),
                'track_ms_p50':float(np.median([r['track_ms'] for r in rows])),
                'track_ms_p95':float(np.percentile([r['track_ms'] for r in rows],95)),
                'max_reserved_mib':max(r['reserved_mib'] for r in rows)}
        summaries.append(result);print(result,flush=True)
    differences={name:float(np.max(np.abs(value-poses['reference1']))) for name,value in poses.items()}
    result={'runs':summaries,'max_pose_element_difference':differences,'iterations':5,
            'note':'Same 160 tracking frames and initial state; independent sequences, not teacher forcing.'}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    if any(value!=0 for value in differences.values()):raise RuntimeError('Pose differences require investigation')


if __name__=='__main__':main()
