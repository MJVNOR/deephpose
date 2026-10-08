"""Worker FS persistente (env principal): evita reimportar torch/modelo por frame.

Protocolo por líneas en stdin/stdout (todo UTF-8, flush siempre):
  IN:  `infer <left.png> <right.png> <out_disp.npy> <scale> <valid_iters>`
  OUT: `ok <out_disp.npy> <t_infer_s>`  |  `err <mensaje>`
  IN:  `quit` -> sale 0.
Resuelve el choque de `Utils` top-level entre FoundationStereo y FoundationPose:
cada stack vive en su proceso.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "scripts")
from fs_model import infer_disp, load_model  # noqa: E402


def main():
    model = cfg = None
    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "quit":
            return
        if parts[0] != "infer" or len(parts) != 6:
            print("err comando: infer <l> <r> <out> <scale> <iters>", flush=True)
            continue
        _, sl, sr, sout, sscale, siters = parts
        try:
            if model is None:
                from config import load
                C = load()
                model, cfg, _ = load_model(C["model"]["ckpt"], int(siters))
            l = cv2.imread(sl, cv2.IMREAD_UNCHANGED)
            r = cv2.imread(sr, cv2.IMREAD_UNCHANGED)
            if l is None or r is None:
                print("err no se leen pares IR", flush=True)
                continue
            d, info = infer_disp(model, cfg, l, r, float(sscale), int(siters))
            np.save(sout, d)
            print(f"ok {sout} {info['t_infer_s']}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"err OOM {e}", flush=True)
        except Exception as e:
            print(f"err {type(e).__name__} {e}", flush=True)


if __name__ == "__main__":
    main()
