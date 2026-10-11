"""CPU-only comparison of saved pose boxes on identical RGB frames."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--offline',type=Path,required=True)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--frames',type=int,nargs='+',default=[1290,1335,1710])
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    vertices=[]
    with a.mesh.open() as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float,line.split()[1:4])))
    vertices=np.asarray(vertices)
    lo,hi=vertices.min(0),vertices.max(0)
    corners=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)]
    poses={iters:{r['frame']:np.asarray(r['pose']) for r in map(json.loads,
        (a.offline/f'fs{iters}/iters5_repeat1/frames.jsonl').read_text().splitlines())} for iters in (4,8)}
    for frame in a.frames:
        with np.load(a.replay/f'bundle_{frame:06d}.npz') as bundle:
            rgb,K=bundle['rgb'],bundle['K']
        panels=[]
        for iters in (4,8):
            P=poses[iters][frame]
            points=(P[:3,:3]@corners.T).T+P[:3,3]
            projected=(K@points.T).T
            xy=np.rint(projected[:,:2]/projected[:,2:]).astype(int)
            canvas=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            for i,j in edges:
                cv2.line(canvas,tuple(xy[i]),tuple(xy[j]),(0,255,0),2)
            cv2.putText(canvas,f'FS{iters} | frame {frame}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            panels.append(canvas)
        cv2.imwrite(str(a.out/f'frame_{frame:06d}.jpg'),np.hstack(panels))


if __name__=='__main__':
    main()
