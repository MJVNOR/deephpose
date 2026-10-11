"""Inspect low scores without turning diagnostic masks into accepted detections."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from contextlib import nullcontext
from backend import EfficientMask, ROOT

p=argparse.ArgumentParser()
p.add_argument('--variant', choices=['ev','tv'], default='ev')
p.add_argument('--out',type=Path,required=True)
a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
model=EfficientMask(a.variant)
source=ROOT/'data/processed/replay_fs_sweep_20261010/fs4'
manifest=json.loads((source/'manifest.json').read_text())
rows=[]
with np.load(source/manifest['frames'][0]['bundle']) as b:
    rgb=b['rgb']
reference=np.load(ROOT/'reports/sam3_reference_20261011/mask_60.npy')
model.processor.confidence_threshold=0.0
for precision in ('bf16','fp32'):
    with torch.inference_mode(), (torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()):
        state=model.processor.set_image(Image.fromarray(rgb))
        for text in ('computer mouse','mouse','wireless mouse','black computer mouse','remote control','object'):
            model.processor.reset_all_prompts(state)
            state=model.processor.set_text_prompt(text,state)
            masks=state['masks'][:,0].cpu().numpy()
            scores=state['scores'].float().cpu().numpy()
            ious=(masks & reference).sum((1,2))/np.maximum(1,(masks | reference).sum((1,2)))
            winner=int(scores.argmax()); best=int(ious.argmax())
            row=dict(variant=a.variant,precision=precision,text=text,max_score=float(scores[winner]),
                top_iou=float(ious[winner]),best_iou=float(ious[best]),best_iou_score=float(scores[best]),
                top_pixels=int(masks[winner].sum()))
            np.save(a.out/f'{precision}_{text.replace(" ","_")}.npy',masks[winner])
            rows.append(row);print(row,flush=True)
            (a.out/'summary.json').write_text(json.dumps(rows,indent=2))
