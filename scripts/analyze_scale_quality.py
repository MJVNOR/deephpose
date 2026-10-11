"""Summarize depth and fixed-frame pose differences; no ground truth assumed."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def stats(values):
    return {key:float(np.percentile(values,p)) for key,p in [('p50',50),('p95',95),('max',100)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundles',type=Path,required=True)
    p.add_argument('--poses',type=Path,required=True)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    depth=json.loads((a.bundles/'depth_comparison.json').read_text())
    rows=depth['frames']
    result={'note':depth['note'],'depth_frames':len(rows),'depth_difference':{}}
    for region in ['global','object_bbox','object_depth_edges']:
        result['depth_difference'][region]={field:stats([r[region][field] for r in rows if r[region][field] is not None]) for field in ['p50_mm','p95_mm']}
    result['roi_validity_disagreement']=stats([r['roi_validity_disagreement'] for r in rows])
    result['offline_runs']={s:json.loads((a.poses/s/'summary.json').read_text()) for s in ['scale0.5','scale0.4']}
    result['fs_model_ms']={s:stats([r['fs_info']['t_infer_s']*1000 for r in json.loads((a.bundles/s/'manifest.json').read_text())['frames']]) for s in ['scale0.5','scale0.4']}
    poses={s:{r['frame']:np.asarray(r['pose']) for r in map(json.loads,(a.poses/s/'iters5_repeat1/frames.jsonl').read_text().splitlines())} for s in ['scale0.5','scale0.4']}
    if poses['scale0.5'].keys()!=poses['scale0.4'].keys(): raise ValueError('Frame mismatch')
    with a.mesh.open() as f:
        vertices=np.asarray([list(map(float,s.split()[1:4])) for s in f if s.startswith('v ')])
    center=(vertices.min(0)+vertices.max(0))/2
    differences=[]
    for frame,ref in poses['scale0.5'].items():
        new=poses['scale0.4'][frame]
        angle=np.degrees(np.arccos(np.clip((np.trace(ref[:3,:3].T@new[:3,:3])-1)/2,-1,1)))
        distance=np.linalg.norm(ref[:3,:3]@center+ref[:3,3]-new[:3,:3]@center-new[:3,3])*1000
        differences.append({'frame':frame,'angle_deg':float(angle),'center_mm':float(distance)})
    result['pose_difference']={k:stats([r[k] for r in differences]) for k in ['angle_deg','center_mm']}
    result['largest_pose_differences']=sorted(differences,key=lambda r:r['angle_deg'],reverse=True)[:10]
    result['repeat_max_abs_difference']={}
    for s in poses:
        repeat={r['frame']:np.asarray(r['pose']) for r in map(json.loads,(a.poses/s/'iters5_repeat2/frames.jsonl').read_text().splitlines())}
        result['repeat_max_abs_difference'][s]=float(max(np.max(np.abs(P-repeat[f])) for f,P in poses[s].items()))
    # Matched source images, reference/new depth and absolute difference (clipped at 20mm).
    for frame in [1260,2250]:
        panels=[];ds={}
        for scale in [.5,.4]:
            with np.load(a.bundles/f'scale{scale}'/f'bundle_{frame:06d}.npz') as b:
                rgb,buf=b['rgb'],b['depth']
            ds[scale]=buf
            finite=np.isfinite(buf)
            values=np.zeros_like(buf);values[finite]=np.clip((buf[finite]-.2)/.8,0,1)
            canvas=cv2.applyColorMap((values*255).astype(np.uint8),cv2.COLORMAP_TURBO);canvas[~finite]=0
            cv2.putText(canvas,f'scale {scale} | depth 0.2-1m',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            panels.append(canvas)
        common=np.isfinite(ds[.5]) & np.isfinite(ds[.4]);diff=np.zeros_like(ds[.5])
        diff[common]=np.clip(np.abs(ds[.5][common]-ds[.4][common])/.02,0,1)
        heat=cv2.applyColorMap((diff*255).astype(np.uint8),cv2.COLORMAP_TURBO);heat[~common]=0
        cv2.putText(heat,'Difference: blue=0 red>=20mm',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
        cv2.imwrite(str(a.out/f'depth_frame_{frame}.jpg'),np.vstack([np.hstack([cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),panels[0]]),np.hstack([panels[1],heat])]))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
