"""Measure SAM loading separately from warm masks on three recorded RGB frames."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from sam_mask import load_sam, sam_mask, configure_sam


def main():
    configure_sam('sam3')  # Historical SAM3 image baseline, not the live default.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    load_sam()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    print('load_seconds', load_seconds, flush=True)
    source = Path('data/processed/replay_fs_sweep_20261010/fs4')
    manifest = json.loads((source/'manifest.json').read_text())
    rows = []
    for index in (0, 0, 80, 160, 0):
        entry = manifest['frames'][index]
        with np.load(source/entry['bundle']) as b:
            rgb = b['rgb']
        torch.cuda.synchronize()
        start = time.perf_counter()
        mask, box, score = sam_mask(rgb, 'computer mouse')
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        row = dict(frame=entry['frame'], seconds=seconds, box=box, score=score,
                   mask_pixels=int(mask.sum()) if mask is not None else 0,
                   reserved_mib=torch.cuda.memory_reserved()/2**20)
        if mask is not None:
            np.save(a.out/f"mask_{entry['frame']}.npy", mask)
        rows.append(row)
        print(row, flush=True)
        (a.out/'summary.json').write_text(json.dumps(dict(load_seconds=load_seconds, rows=rows), indent=2))


if __name__ == '__main__':
    main()
