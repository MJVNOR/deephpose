"""Run pose tracking on three already prepared FS variants, one GPU process at a time."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mesh',required=True)
    p.add_argument('--initial-state',required=True)
    p.add_argument('--variants',nargs='+',default=['fs16','fs8','fs4'])
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    for name in a.variants:
        with (a.out/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,'pose/benchmark_replay.py','--replay',str(a.source/name),
                '--mesh',a.mesh,'--out',str(a.out/name),'--initial-state',a.initial_state,
                '--track-iters','5','--repeats','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(f'Completed {name} / track5, two repeats',flush=True)


if __name__=='__main__':
    main()
