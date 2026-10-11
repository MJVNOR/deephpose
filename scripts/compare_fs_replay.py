"""Recompute FS 4/8/16 on the exact raw pairs from a frozen replay."""
import argparse
import copy
import json
import time
from pathlib import Path
import cv2
import numpy as np
from prepare_replay import sha256
from fs_model import load_model, infer_disp
from disparity_to_depth import check_stereo_cal, disp_to_depth
from project_to_rgb import project_ir_to_rgb
from config import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    base = json.loads((a.source/'manifest.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for iters in [4, 8, 16]:
        dst = a.out/f'fs{iters}'
        dst.mkdir()
        m = copy.deepcopy(base)
        m['valid_iters'] = iters
        m['source_manifest_sha256'] = sha256(a.source/'manifest.json')
        m['source_sha256'] = {f: sha256(f) for f in [__file__, 'scripts/fs_model.py', 'scripts/project_to_rgb.py']}
        m['mode'] = 'same raw pairs; FS order rotated per frame; warmed configurations'
        manifests[iters] = m
    cal = base['calibration']
    _, B = check_stereo_cal(cal)
    ir, ex = cal['ir1'], cal['extrinsics_ir1_to_color']
    K = np.array([[ir['fx'],0,ir['ppx']],[0,ir['fy'],ir['ppy']],[0,0,1]])
    model,cfg,_ = load_model(load()['model']['ckpt'],16)
    with np.load(a.source/base['frames'][0]['raw']) as raw:
        for iters in [4,8,16]:
            for _ in range(2):
                infer_disp(model,cfg,raw['left'],raw['right'],base['scale'],iters)
    for i, source in enumerate(base['frames']):
        rawpath = (a.source/source['raw']).resolve()
        with np.load(rawpath) as raw:
            left,right,color = raw['left'],raw['right'],raw['rgb']
        order = [4,8,16]
        order = order[i%3:] + order[:i%3]
        depths = {}
        for iters in order:
            t0 = time.perf_counter()
            d,info = infer_disp(model,cfg,left,right,base['scale'],iters)
            sx = d.shape[1]/left.shape[1]
            if d.shape != left.shape:
                d = cv2.resize(d,(left.shape[1],left.shape[0]),interpolation=cv2.INTER_LINEAR)/sx
            depth,valid = disp_to_depth(d,K[0,0],B)
            buf,_,kr,rect,_ = project_ir_to_rgb(depth,valid,K,ex['rotation'],ex['translation'],color,cal['color'])
            row = manifests[iters]['frames'][i]
            row['fs_and_projection_s'] = time.perf_counter()-t0
            row['fs_info'] = info
            row['raw'] = str(rawpath)
            row['raw_sha256'] = sha256(rawpath)
            target = a.out/f'fs{iters}'/row['bundle']
            np.savez(target,rgb=rect,depth=buf,K=kr)
            row['sha256'] = sha256(target)
            depths[iters] = buf
        for iters in [4,8,16]:
            ok = np.isfinite(depths[iters]) & np.isfinite(depths[16])
            row = manifests[iters]['frames'][i]
            row['depth_vs16_mm_p50'] = float(np.median(np.abs(depths[iters][ok]-depths[16][ok]))*1000) if ok.any() else None
            row['depth_valid_frac'] = float(np.isfinite(depths[iters]).mean())
        if i%20==0:
            print(f'Compared FS 4/8/16: {i+1}/{len(base["frames"])}',flush=True)
    for iters,m in manifests.items():
        (a.out/f'fs{iters}'/'manifest.json').write_text(json.dumps(m,indent=2))
    print('Complete:',a.out,flush=True)


if __name__ == '__main__':
    main()
