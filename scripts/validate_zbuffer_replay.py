"""Exact whole-projection comparison on real stereo frames, alternating A/B order."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from config import load
from fs_model import load_model,infer_disp
from disparity_to_depth import check_stereo_cal,disp_to_depth
from project_to_rgb import project_ir_to_rgb
from prepare_replay import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.replay/'manifest.json').read_text())
    cal=manifest['calibration']; ir=cal['ir1']; ex=cal['extrinsics_ir1_to_color']
    _,baseline=check_stereo_cal(cal)
    K=np.array([[ir['fx'],0,ir['ppx']],[0,ir['fy'],ir['ppy']],[0,0,1]])
    model,cfg,_=load_model(load()['model']['ckpt'],4)
    timings={True:[],False:[]}
    with (a.out/'frames.jsonl').open('x') as output:
        for i,row in enumerate(manifest['frames']):
            with np.load(a.replay/row['raw']) as raw:
                left,right,rgb=raw['left'],raw['right'],raw['rgb']
            disp,_=infer_disp(model,cfg,left,right,.5,4)
            disp=cv2.resize(disp,(left.shape[1],left.shape[0]))/(disp.shape[1]/left.shape[1])
            depth,valid=disp_to_depth(disp,ir['fx'],baseline)
            inputs=(depth,valid,K,ex['rotation'],ex['translation'],rgb,cal['color'])
            if i==0:
                for sorted_mode in (True,False):
                    project_ir_to_rgb(*inputs,sorted_zbuffer=sorted_mode)
            results={};elapsed={}
            for sorted_mode in ((True,False) if i%2==0 else (False,True)):
                start=time.perf_counter()
                results[sorted_mode]=project_ir_to_rgb(*inputs,sorted_zbuffer=sorted_mode)
                elapsed[sorted_mode]=time.perf_counter()-start
                timings[sorted_mode].append(elapsed[sorted_mode])
            for old,new in zip(results[True][:4],results[False][:4]):
                np.testing.assert_array_equal(old,new)
            for key in results[True][4]:
                np.testing.assert_array_equal(results[True][4][key],results[False][4][key])
            output.write(json.dumps({'frame':row['frame'],'exact':True,
                'sorted_ms':elapsed[True]*1000,'reduced_ms':elapsed[False]*1000})+'\n')
            output.flush()
            if i%20==0: print('Verified',i+1,flush=True)
    summary={'frames':len(manifest['frames']),'exact':True,
        'fields':['depth','validity','K','RGB','u','v','zc','pts','arg'],
        'manifest_sha256':sha256(a.replay/'manifest.json'),
        'source_sha256':sha256('scripts/project_to_rgb.py'),
        'timings_ms':{('sorted' if mode else 'reduced'):{'p50':float(np.median(values)*1000),
             'p95':float(np.percentile(values,95)*1000)} for mode,values in timings.items()}}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
