"""Profile RGB projection on real replay inputs, separately from throughput runs."""
import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path
import cv2
import numpy as np
from config import load
from fs_model import load_model,infer_disp
from disparity_to_depth import check_stereo_cal,disp_to_depth
from project_to_rgb import project_ir_to_rgb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.replay/'manifest.json').read_text())
    cal=manifest['calibration']
    _,baseline=check_stereo_cal(cal)
    ir=cal['ir1']
    K=np.array([[ir['fx'],0,ir['ppx']],[0,ir['fy'],ir['ppy']],[0,0,1]])
    ex=cal['extrinsics_ir1_to_color']
    model,cfg,_=load_model(load()['model']['ckpt'],4)
    inputs=[]
    for index in [0,len(manifest['frames'])//2,-1]:
        row=manifest['frames'][index]
        with np.load(a.replay/row['raw']) as raw:
            left,right,rgb=raw['left'],raw['right'],raw['rgb']
        disp,_=infer_disp(model,cfg,left,right,.5,4)
        disp=cv2.resize(disp,(left.shape[1],left.shape[0]))/(disp.shape[1]/left.shape[1])
        depth,valid=disp_to_depth(disp,ir['fx'],baseline)
        inputs.append((depth,valid,K,ex['rotation'],ex['translation'],rgb,cal['color']))
    for sample in inputs:
        project_ir_to_rgb(*sample)
    profiler=cProfile.Profile()
    timings=[]
    profiler.enable()
    for _ in range(10):
        for sample in inputs:
            start=time.perf_counter()
            project_ir_to_rgb(*sample)
            timings.append(time.perf_counter()-start)
    profiler.disable()
    profiler.dump_stats(str(a.out/'projection.prof'))
    with (a.out/'projection.txt').open('w') as f:
        pstats.Stats(profiler,stream=f).sort_stats('cumulative').print_stats(30)
        pstats.Stats(profiler,stream=f).sort_stats('tottime').print_stats(20)
    (a.out/'summary.json').write_text(json.dumps({'calls':len(timings),
        'frames':[manifest['frames'][i]['frame'] for i in [0,len(manifest['frames'])//2,-1]],
        'wall_ms_p50':float(np.median(timings)*1000),'wall_ms_p95':float(np.percentile(timings,95)*1000),
        'note':'CPU profile over three real inputs, FS4 scale .5; isolated, not concurrent throughput.'},indent=2))


if __name__=='__main__':
    main()
