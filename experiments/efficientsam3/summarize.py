"""Summarize local benchmarks and render diagnostic mask comparisons."""
import json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
report=ROOT/'reports/efficientsam3_comparison_20261011'
report.mkdir(parents=True,exist_ok=True)
names=['sam3_reference_20261011','efficientsam3_ev_m_20261011','efficientsam3_ev_m_thr02_20261011',
       'efficientsam3_ev_m_thr01_20261011','efficientsam3_tv_m_20261011','efficientsam3_tv_m_thr01_20261011']
summary=[]
for name in names:
    data=json.loads((ROOT/'reports'/name/'summary.json').read_text())
    times=[r['seconds'] for r in data['rows']]
    summary.append(dict(run=name,parameters=data['parameters'],mean_s=float(np.mean(times)),
        median_s=float(np.median(times)),p95_s=float(np.percentile(times,95)),
        detected_frames=sum(r['detected'] for r in data['rows'][::2]),frames=len(data['rows'][::2]),
        peak_allocated_mib=max(r['peak_mib'] for r in data['rows']),
        allocated_mib=max(r['allocated_mib'] for r in data['rows']),model_load_s=data['model_load_s']))
reference=ROOT/'reports/sam3_reference_20261011'
candidate=ROOT/'reports/efficientsam3_ev_m_thr01_20261011'
agreement=[]
for path in sorted(candidate.glob('mask_*.npy')):
    if (reference/path.name).exists():
        a,b=np.load(path),np.load(reference/path.name)
        agreement.append(dict(frame=int(path.stem.split('_')[1]),iou=float((a&b).sum()/max(1,(a|b).sum()))))
output=dict(runs=summary,mask_agreement=agreement)
(report/'summary.json').write_text(json.dumps(output,indent=2))
print(json.dumps(output,indent=2))
source=ROOT/'data/processed/replay_fs_sweep_20261010/fs4'
manifest=json.loads((source/'manifest.json').read_text())
entries={e['frame']:e for e in manifest['frames']}
panels=[]
for frame in [60,1140,1260,1740,1980,2460]:
    with np.load(source/entries[frame]['bundle']) as b:
        rgb=b['rgb']
    row=[]
    for label,folder in [('RGB',None),('SAM3 threshold .5',reference),('Efficient EV threshold .1',candidate)]:
        shown=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        subtitle=f'frame {frame}'
        if folder is not None:
            path=folder/f'mask_{frame}.npy'
            if path.exists():
                mask=np.load(path)
                shown[mask]=(.55*shown[mask]+.45*np.array([0,255,0])).astype(np.uint8)
            else:
                subtitle+=' - NO DETECTION'
        tile=np.zeros((280,320,3),np.uint8)
        tile[40:]=cv2.resize(shown,(320,240))
        cv2.putText(tile,label,(8,16),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
        cv2.putText(tile,subtitle,(8,33),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1)
        row.append(tile)
    panels.append(np.hstack(row))
cv2.imwrite(str(report/'masks.jpg'),np.vstack(panels))
