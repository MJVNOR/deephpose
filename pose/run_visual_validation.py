"""Replay a second recording with GUI and sampled overlays; independent SAM initialization."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bag',required=True)
    p.add_argument('--mesh',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--iters',type=int,nargs='+',default=[8,4])
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    for iters in a.iters:
        label=f'fs{iters}'
        command=[sys.executable,'pose/live_pose.py','--bag',a.bag,'--mesh',a.mesh,
            '--scale','.5','--valid-iters',str(iters),'--track-iters','5',
            '--mask-text','computer mouse','--metrics-jsonl',str(a.out/f'{label}.jsonl'),
            '--snapshots',str(a.out/f'{label}_frames')]
        print('Starting',label,flush=True)
        with (a.out/f'{label}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=list(map(json.loads,(a.out/f'{label}.jsonl').read_text().splitlines()))
        print('Completed',label,'updates',len(rows),'lost',sum(r['state']=='LOST' for r in rows),flush=True)


if __name__=='__main__':
    main()
