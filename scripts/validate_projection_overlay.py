"""Exact live-output equivalence and paired CPU timings on 161 real inputs."""
import argparse
import ast
import json
import time
import sys
from pathlib import Path
import cv2
import numpy as np
from config import load
from fs_model import load_model, infer_disp
from disparity_to_depth import check_stereo_cal, disp_to_depth
from project_to_rgb import project_ir_to_rgb
sys.path.insert(0, str(Path('pose').resolve()))
from overlay_fast import draw_opaque_axes


def reference_axes():
    tree = ast.parse(Path('pose/FoundationPose/Utils.py').read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('draw_xyz_axis','project_3d_to_2d')]
    env = {'np':np,'cv2':cv2}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'Utils.py','exec'),env)
    return env['draw_xyz_axis']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    mesh=Path('data/mesh/mouse/910-007575_MX_MASTER_4_MAC_Space_Black_prepared.obj')
    vertices=np.asarray([list(map(float,line.split()[1:4])) for line in mesh.read_text().splitlines() if line.startswith('v ')])
    center=(vertices.min(0)+vertices.max(0))/2
    manifest=json.loads((a.replay/'manifest.json').read_text())
    poses={r['frame']:np.asarray(r['pose']) for r in map(json.loads,Path('reports/fs_pose_comparison_20261010/fs4/iters5_repeat1/frames.jsonl').read_text().splitlines())}
    cal=manifest['calibration']; ir=cal['ir1']; ex=cal['extrinsics_ir1_to_color']
    _,baseline=check_stereo_cal(cal)
    K=np.array([[ir['fx'],0,ir['ppx']],[0,ir['fy'],ir['ppy']],[0,0,1]])
    model,cfg,_=load_model(load()['model']['ckpt'],4)
    old_axes=reference_axes();rows=[]
    for i,row in enumerate(manifest['frames']):
        with np.load(a.replay/row['raw']) as raw:
            left,right,rgb=raw['left'],raw['right'],raw['rgb']
        disp,_=infer_disp(model,cfg,left,right,.5,4)
        disp=cv2.resize(disp,(left.shape[1],left.shape[0]))/(disp.shape[1]/left.shape[1])
        depth,valid=disp_to_depth(disp,ir['fx'],baseline)
        args=(depth,valid,K,ex['rotation'],ex['translation'],rgb,cal['color'])
        results={}; times={}
        for detailed in ([True,False] if i%2==0 else [False,True]):
            t=time.perf_counter(); results[detailed]=project_ir_to_rgb(*args,return_details=detailed)
            times['projection_reference_ms' if detailed else 'projection_fast_ms']=(time.perf_counter()-t)*1000
        for old,new in zip(results[True][:4],results[False][:4]): np.testing.assert_array_equal(old,new)
        pose=poses.get(row['frame'],np.load('reports/replay_comparison_20261010_v2/initial_pose.npy'))
        pose=pose.copy();pose[:3,3]+=pose[:3,:3]@center
        rect,Krect=results[False][3],results[False][2]
        axes={}
        for reference in ([True,False] if i%2==0 else [False,True]):
            image=rect.copy();t=time.perf_counter()
            axes[reference]=(old_axes(image,pose,K=Krect,is_input_rgb=True) if reference else draw_opaque_axes(image,pose,Krect))
            times['axes_reference_ms' if reference else 'axes_fast_ms']=(time.perf_counter()-t)*1000
        np.testing.assert_array_equal(axes[True],axes[False])
        if i==0:
            cv2.imwrite(str(a.out/'axes.jpg'),cv2.cvtColor(axes[False],cv2.COLOR_RGB2BGR))
        rows.append(dict(frame=row['frame'],exact=True,**times))
        if i%20==0: print(rows[-1],flush=True)
    result={'frames':len(rows),'exact':True,'fields':['depth','validity','K','RGB','axis_pixels'],
            'timings_ms_p50':{k:float(np.median([r[k] for r in rows])) for k in times}}
    (a.out/'frames.json').write_text(json.dumps(rows,indent=2))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
