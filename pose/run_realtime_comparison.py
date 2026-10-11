"""Sequential paired full-bag A/B runs; no overlapping GPU benchmarks."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bag',required=True)
    p.add_argument('--mesh',required=True)
    p.add_argument('--benchmark-init',required=True)
    p.add_argument('--initial-state',required=True)
    p.add_argument('--candidate',type=int,default=8)
    p.add_argument('--reference',type=int,default=16)
    p.add_argument('--optimized-only',action='store_true',help='ABBA con ambos caminos optimizados')
    p.add_argument('--compare-zbuffer',action='store_true',help='ABBA ordenamiento/reducción con las mismas iteraciones FS')
    p.add_argument('--compare-scales',action='store_true',help='ABBA de resolución con FS fijo')
    p.add_argument('--compare-volumes',action='store_true',help='ABBA de volúmenes estéreo con FS y escala fijos')
    p.add_argument('--compare-cpu',action='store_true',help='ABBA de proyección y overlay con FS y escala fijos')
    p.add_argument('--compare-tracking',action='store_true',help='ABBA del tracking con cinco iteraciones')
    p.add_argument('--reference-scale',type=float,default=.5)
    p.add_argument('--candidate-scale',type=float,default=.4)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if sum([a.compare_zbuffer,a.optimized_only,a.compare_scales,a.compare_volumes,a.compare_cpu,a.compare_tracking])>1:
        p.error('Selecciona un solo modo de comparación')
    if not 0<a.reference_scale<=1 or not 0<a.candidate_scale<=1:
        p.error('Las escalas deben estar en (0,1]')
    a.out.mkdir(parents=True,exist_ok=False)
    plans=[('baseline_r1',a.reference,True),('optimized_r1',a.candidate,False),
           ('optimized_r2',a.candidate,False),('baseline_r2',a.reference,True),
           ('candidate_legacy',a.candidate,True)]
    if a.optimized_only:
        plans=[('reference_r1',a.reference,False),('candidate_r1',a.candidate,False),
               ('candidate_r2',a.candidate,False),('reference_r2',a.reference,False)]
    if a.compare_zbuffer:
        plans=[('sorted_r1',a.candidate,False),('reduced_r1',a.candidate,False),
               ('reduced_r2',a.candidate,False),('sorted_r2',a.candidate,False)]
    if a.compare_scales or a.compare_volumes or a.compare_cpu or a.compare_tracking:
        plans=[('reference_r1',a.candidate,False),('candidate_r1',a.candidate,False),
               ('candidate_r2',a.candidate,False),('reference_r2',a.candidate,False)]
    results=[]
    for label,iters,legacy in plans:
        scale=(a.candidate_scale if label.startswith('candidate_') else a.reference_scale) if a.compare_scales else .5
        command=[sys.executable,'pose/live_pose.py','--bag',a.bag,'--mesh',a.mesh,
                 '--scale',str(scale),'--valid-iters',str(iters),'--track-iters','5',
                 '--headless','--benchmark-init',a.benchmark_init,'--initial-state',a.initial_state,
                 '--metrics-jsonl',str(a.out/f'{label}.jsonl')]
        if legacy:
            command.append('--legacy-costs')
        if a.compare_zbuffer and label.startswith('sorted_'):
            command.append('--sorted-zbuffer')
        if a.compare_volumes and label.startswith('reference_'):
            command.append('--reference-volumes')
        if a.compare_cpu and label.startswith('reference_'):
            command.append('--reference-cpu')
        if a.compare_tracking and label.startswith('reference_'):
            command.append('--reference-tracking')
        print('Starting',label,flush=True)
        with (a.out/f'{label}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=[json.loads(s) for s in (a.out/f'{label}.jsonl').read_text().splitlines()]
        if len(rows)<2:
            raise RuntimeError(f'{label}: incomplete replay')
        result={'label':label,'valid_iters':iters,'scale':scale,'legacy':legacy,'updates':len(rows),
                'reference_volumes':rows[0].get('reference_volumes',True),
                'reference_cpu':rows[0].get('reference_cpu',True),
                'reference_tracking':rows[0].get('reference_tracking',True),
                'lost':sum(r['state']=='LOST' for r in rows),
                'pose_hz':(len(rows)-1)/(rows[-1]['wall_s']-rows[0]['wall_s']),
                'first_frame':rows[0]['frame_numbers'][0],'last_frame':rows[-1]['frame_numbers'][0]}
        results.append(result)
        (a.out/'summary.json').write_text(json.dumps(results,indent=2))
        print('Completed',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
