"""Aggregate paired FS runs and compare offline poses at identical source frames."""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_realtime import summarize


def percentiles(values):
    return {name:float(np.percentile(values,p)) for name,p in [('p50',50),('p95',95),('max',100)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--offline',type=Path)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--group-by',choices=['fs_iters','sorted_zbuffer','scale','reference_volumes','reference_cpu','reference_tracking'],default='fs_iters')
    args=parser.parse_args()
    completed=json.loads((args.directory/'summary.json').read_text())
    if len(completed)!=4:
        raise ValueError('ABBA runner has not completed all four runs')
    runs=[summarize(p) for p in sorted(args.directory.glob('*.jsonl'))]
    if len(runs)!=4:
        raise ValueError('Expected four completed ABBA runs')
    result={'runs':runs,'groups':{}}
    for variant in sorted({r[args.group_by] for r in runs}):
        subset=[r for r in runs if r[args.group_by]==variant]
        rows=[]
        for r in subset:
            rows.extend([json.loads(s) for s in (args.directory/(r['run']+'.jsonl')).read_text().splitlines()][1:])
        label=('sorted' if variant else 'reduced') if args.group_by=='sorted_zbuffer' else str(variant)
        result['groups'][label]={
            'runs':len(subset),'updates':sum(r['updates'] for r in subset),
            'lost':sum(r['lost'] for r in subset),
            'pose_hz':sum(r['updates']-1 for r in subset)/sum(r['measured_wall_s'] for r in subset),
            'latency_ms':percentiles([r['capture_to_publish_s']*1000 for r in rows]),
            'stages_ms':{k:percentiles([r['stages'][k]*1000 for r in rows])
                         for k,v in rows[0]['stages'].items() if isinstance(v,(float,int))}}
    if args.offline:
        center=None
        if args.mesh:
            with args.mesh.open() as f:
                vertices=np.asarray([list(map(float,line.split()[1:4])) for line in f if line.startswith('v ')])
            center=(vertices.min(0)+vertices.max(0))/2
        def load(iters):
            path=args.offline/f'fs{iters}'/'iters5_repeat1'/'frames.jsonl'
            return {r['frame']:r for r in map(json.loads,path.read_text().splitlines())}
        ref,candidate=load(8),load(4)
        if ref.keys()!=candidate.keys():
            raise ValueError('Offline comparison needs identical frames')
        differences=[]
        for frame in ref:
            a,b=np.asarray(ref[frame]['pose']),np.asarray(candidate[frame]['pose'])
            angle=np.degrees(np.arccos(np.clip((np.trace(a[:3,:3].T@b[:3,:3])-1)/2,-1,1)))
            differences.append({'frame':frame,'origin_translation_mm':float(np.linalg.norm(a[:3,3]-b[:3,3])*1000),
                                'rotation_deg':float(angle)})
            if center is not None:
                differences[-1]['center_translation_mm']=float(np.linalg.norm(
                    a[:3,:3]@center+a[:3,3]-b[:3,:3]@center-b[:3,3])*1000)
        result['offline_pose_difference_fs4_vs_fs8']={
            'note':'Difference between estimates, not ground-truth accuracy; translation uses original OBJ origin.',
            'frames':len(differences),
            'origin_translation_mm':percentiles([r['origin_translation_mm'] for r in differences]),
            'rotation_deg':percentiles([r['rotation_deg'] for r in differences]),
            'largest_rotation_differences':sorted(differences,key=lambda r:r['rotation_deg'],reverse=True)[:5]}
        if center is not None:
            result['offline_pose_difference_fs4_vs_fs8']['center_translation_mm']=percentiles(
                [r['center_translation_mm'] for r in differences])
    (args.directory/'validation_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
