"""Run with the selected backend's own Python environment; output comparable masks."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['sam3', 'efficient'], required=True)
    p.add_argument('--variant', choices=['ev','tv'], default='ev')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    if a.backend == 'efficient':
        from backend import EfficientMask
        infer = EfficientMask(a.variant)
        parameters = infer.parameters
        with infer.checkpoint.open('rb') as handle:
            checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    else:
        sys.path.insert(0, str(ROOT/'pose'))
        from sam_mask import sam_mask, load_sam, configure_sam
        configure_sam('sam3')
        model, _ = load_sam()
        parameters = sum(p.numel() for p in model.parameters())
        infer = sam_mask
        checkpoint_hash = 'facebook/sam3 local HF cache'
    torch.cuda.synchronize()
    result = dict(backend=a.backend, variant=a.variant if a.backend=='efficient' else None, threshold=a.threshold, model_load_s=time.perf_counter()-start,
        parameters=parameters, python=platform.python_version(), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), checkpoint_sha256=checkpoint_hash, rows=[],
        upstream_commit=subprocess.check_output(['git','-C',str(ROOT/'third_party/EfficientSAM3'),'rev-parse','HEAD'],text=True).strip())
    print({k:v for k,v in result.items() if k!='rows'}, flush=True)
    source = ROOT/'data/processed/replay_fs_sweep_20261010/fs4'
    manifest = json.loads((source/'manifest.json').read_text())
    with np.load(source/manifest['frames'][0]['bundle']) as b:
        for _ in range(2):
            infer(b['rgb'], 'computer mouse', thr=a.threshold)
    for index in range(0,161,8):
        entry = manifest['frames'][index]
        with np.load(source/entry['bundle']) as b:
            rgb = b['rgb']
        for repeat in range(2):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            mask, box, score = infer(rgb, 'computer mouse', thr=a.threshold)
            torch.cuda.synchronize()
            row = dict(frame=entry['frame'], index=index, repeat=repeat, seconds=time.perf_counter()-start,
                detected=mask is not None, pixels=int(mask.sum()) if mask is not None else 0,
                box=box, score=score, allocated_mib=torch.cuda.memory_allocated()/2**20,
                peak_mib=torch.cuda.max_memory_allocated()/2**20,
                reserved_mib=torch.cuda.memory_reserved()/2**20)
            if mask is not None:
                np.save(a.out/f"mask_{entry['frame']}.npy", mask)
            result['rows'].append(row); print(row, flush=True)
            (a.out/'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
