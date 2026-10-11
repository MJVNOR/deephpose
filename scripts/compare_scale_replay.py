"""Paired FS4 scale .5/.4 over fixed stereo frames, with global/ROI/edge differences."""
import argparse
import copy
import json
import time
from pathlib import Path
import cv2
import numpy as np
from prepare_replay import sha256
from fs_model import load_model,infer_disp
from config import load
from disparity_to_depth import check_stereo_cal,disp_to_depth
from project_to_rgb import project_ir_to_rgb


def difference(a,b,mask):
    values=np.abs(a[mask]-b[mask])*1000
    return {'pixels':int(len(values)),'p50_mm':float(np.median(values)) if len(values) else None,
            'p95_mm':float(np.percentile(values,95)) if len(values) else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pose-reference',type=Path,required=True)
    p.add_argument('--initial-state',type=Path,required=True)
    p.add_argument('--mesh',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    base=json.loads((a.source/'manifest.json').read_text())
    manifests={}
    for scale in [.5,.4]:
        folder=a.out/f'scale{scale}';folder.mkdir()
        m=copy.deepcopy(base);m.update(scale=scale,valid_iters=4,
            source_manifest_sha256=sha256(a.source/'manifest.json'),
            source_sha256={f:sha256(f) for f in [__file__,'scripts/fs_model.py','scripts/project_to_rgb.py']},
            mode='same stereo frames, alternating scale order, warmed configurations')
        manifests[scale]=m
    poses={r['frame']:np.asarray(r['pose']) for r in map(json.loads,a.pose_reference.read_text().splitlines())}
    poses[base['frames'][0]['frame']]=np.load(a.initial_state/'initial_pose.npy')
    with a.mesh.open() as f:
        vertices=np.asarray([list(map(float,s.split()[1:4])) for s in f if s.startswith('v ')])
    lo,hi=vertices.min(0),vertices.max(0)
    corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
    cal=base['calibration'];ir=cal['ir1'];ex=cal['extrinsics_ir1_to_color']
    _,B=check_stereo_cal(cal)
    K=np.array([[ir['fx'],0,ir['ppx']],[0,ir['fy'],ir['ppy']],[0,0,1]])
    model,cfg,_=load_model(load()['model']['ckpt'],4)
    with np.load(a.source/base['frames'][0]['raw']) as raw:
        for scale in [.5,.4]:
            for _ in range(2): infer_disp(model,cfg,raw['left'],raw['right'],scale,4)
    comparisons=[]
    for i,source in enumerate(base['frames']):
        rawpath=(a.source/source['raw']).resolve()
        with np.load(rawpath) as raw: left,right,rgb=raw['left'],raw['right'],raw['rgb']
        depths={}
        for scale in ([.5,.4] if i%2==0 else [.4,.5]):
            start=time.perf_counter()
            disp,info=infer_disp(model,cfg,left,right,scale,4)
            disp=cv2.resize(disp,(left.shape[1],left.shape[0]),interpolation=cv2.INTER_LINEAR)/(disp.shape[1]/left.shape[1])
            depth,valid=disp_to_depth(disp,K[0,0],B)
            buf,_,kr,rect,_=project_ir_to_rgb(depth,valid,K,ex['rotation'],ex['translation'],rgb,cal['color'])
            row=manifests[scale]['frames'][i]
            row.update(fs_and_projection_s=time.perf_counter()-start,fs_info=info,raw=str(rawpath),raw_sha256=sha256(rawpath))
            target=a.out/f'scale{scale}'/row['bundle']; np.savez(target,rgb=rect,depth=buf,K=kr)
            row['sha256']=sha256(target); depths[scale]=buf
        ref,new=depths[.5],depths[.4]; finite=np.isfinite(ref);common=finite & np.isfinite(new)
        roi=np.zeros_like(common);P=poses[source['frame']]
        points=(P[:3,:3]@corners.T).T+P[:3,3]; projected=(kr@points.T).T
        if np.all(points[:,2]>0):
            xy=projected[:,:2]/projected[:,2:]
            x0,y0=np.floor(xy.min(0)).astype(int)-2;x1,y1=np.ceil(xy.max(0)).astype(int)+3
            x0,x1=np.clip([x0,x1],0,ref.shape[1]);y0,y1=np.clip([y0,y1],0,ref.shape[0])
            roi[y0:y1,x0:x1]=True
        edges=np.zeros_like(common)
        horizontal=finite[:,1:] & finite[:,:-1]
        vertical=finite[1:] & finite[:-1]
        delta_x=np.zeros_like(ref[:,1:]);delta_y=np.zeros_like(ref[1:])
        np.subtract(ref[:,1:],ref[:,:-1],out=delta_x,where=horizontal)
        np.subtract(ref[1:],ref[:-1],out=delta_y,where=vertical)
        edges[:,1:] |= horizontal & (np.abs(delta_x)>.005)
        edges[1:] |= vertical & (np.abs(delta_y)>.005)
        edges=cv2.dilate(edges.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        comparisons.append({'frame':source['frame'],'global':difference(new,ref,common),
            'object_bbox':difference(new,ref,common & roi),'object_depth_edges':difference(new,ref,common & roi & edges),
            'roi_pixels':int(roi.sum()),'roi_validity_disagreement':float(((finite!=np.isfinite(new)) & roi).sum()/max(1,roi.sum())),
            'valid_fraction':{str(s):float(np.isfinite(d).mean()) for s,d in depths.items()}})
        if i%20==0: print('Compared',i+1,flush=True)
    for scale,m in manifests.items(): (a.out/f'scale{scale}'/'manifest.json').write_text(json.dumps(m,indent=2))
    (a.out/'depth_comparison.json').write_text(json.dumps({'note':'Differences against scale .5, not absolute error. ROI is projected CAD bounding box from saved .5 poses; depth edges are >5mm adjacent finite differences, dilated 1px.','frames':comparisons},indent=2))


if __name__=='__main__': main()
