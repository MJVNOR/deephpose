"""Isolated FP tracking profile with CPU timings and CUDA-event intervals."""
import argparse
import cProfile
import json
import logging
import pstats
from pathlib import Path
import numpy as np
import torch
from pose_model import load_textured_mesh,make_estimator,track_pose
import estimater as estimator_module
import learning.training.predict_pose_refine as refine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--paired-only',action='store_true',help='Alternate warmed reference/candidate without profiling hooks')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    replay=Path('data/processed/replay_fs_sweep_20261010/fs4')
    manifest=json.loads((replay/'manifest.json').read_text())
    with np.load(replay/manifest['frames'][0]['bundle']) as data:
        rgb,depth,K=data['rgb'],data['depth'],data['K']
    mesh=load_textured_mesh('data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    est=make_estimator(mesh,Path('pose/FoundationPose'),a.out/'debug')
    logging.getLogger().setLevel(logging.WARNING)
    pose=np.load('reports/replay_comparison_20261010_v2/initial_pose.npy')
    centered=torch.as_tensor(pose@np.linalg.inv(est.get_tf_to_centered_mesh().cpu().numpy()),device='cuda',dtype=torch.float32)
    for _ in range(3):
        est.pose_last=centered.clone();track_pose(est,K,rgb,depth,5,reference=True)
    if a.paired_only:
        samples={True:[],False:[]}
        for repeat in range(20):
            for reference in ([True,False] if repeat%2==0 else [False,True]):
                est.pose_last=centered.clone();track_pose(est,K,rgb,depth,5,reference=reference)
                est.pose_last=centered.clone()
                _,seconds=track_pose(est,K,rgb,depth,5,reference=reference)
                samples[reference].append(seconds*1000)
        result={('reference' if key else 'optimized'):{'p50_ms':float(np.median(values)),
            'p95_ms':float(np.percentile(values,95)),'samples_ms':values} for key,values in samples.items()}
        (a.out/'paired.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
        return
    records=[];patches=[]
    def wrap(obj,name,label):
        original=getattr(obj,name)
        def timed(*args,**kwargs):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();result=original(*args,**kwargs);end.record()
            records.append((label,start,end));return result
        patches.append((obj,name,original));setattr(obj,name,timed)
    for name in ('erode_depth','bilateral_filter_depth','depth2xyzmap_batch'):
        wrap(estimator_module,name,name)
    for name in ('make_crop_data_batch','compute_crop_window_tf_batch','nvdiffrast_render'):
        wrap(refine,name,name)
    wrap(refine.kornia.geometry.transform,'warp_perspective','warp_perspective')
    wrap(est.refiner.dataset,'transform_batch','transform_batch')
    wrap(est.refiner.model,'forward','network_forward')
    wrap(torch.cuda,'empty_cache','empty_cache')
    samples=[];totals=[];prof=cProfile.Profile()
    try:
        prof.enable()
        for _ in range(10):
            records.clear();est.pose_last=centered.clone()
            _,seconds=track_pose(est,K,rgb,depth,5,reference=True);totals.append(seconds*1000)
            sample={}
            for label,start,end in records:sample[label]=sample.get(label,0)+start.elapsed_time(end)
            samples.append(sample)
        prof.disable()
    finally:
        for obj,name,original in reversed(patches):setattr(obj,name,original)
    prof.dump_stats(str(a.out/'tracking.prof'))
    with (a.out/'tracking.txt').open('w') as f:pstats.Stats(prof,stream=f).sort_stats('cumtime').print_stats(45)
    result={'iterations':5,'calls':10,'total_ms_p50':float(np.median(totals)),
            'stages_ms_p50':{k:float(np.median([s[k] for s in samples])) for k in samples[0]},
            'note':'CUDA-event elapsed intervals include stream gaps; nested stages overlap. Isolated, not concurrent throughput.'}
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)


if __name__=='__main__':main()
