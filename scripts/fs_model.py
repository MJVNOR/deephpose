"""Núcleo compartido infer.py/live.py: carga del modelo e inferencia de disparidad."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "third_party/FoundationStereo")
from core.foundation_stereo import FoundationStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402


def load_model(ckpt_path, valid_iters):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        sys.exit(f"error: pesos no encontrados: {ckpt_path}")
    cfg = OmegaConf.load(ckpt_path.parent / "cfg.yaml")
    cfg.setdefault("vit_size", "vits")
    cfg.valid_iters = valid_iters
    torch.autograd.set_grad_enabled(False)
    t0 = time.perf_counter()
    model = FoundationStereo(cfg)
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model"])  # pesos oficiales NVIDIA
    if not torch.cuda.is_available():
        sys.exit("error: sin GPU (torch.cuda no disponible)")
    model.cuda().eval()
    return model, cfg, time.perf_counter() - t0


def to_3ch(img, scale):
    if img.ndim == 2:  # IR mono -> 3 canales replicados, intensidades intactas
        img = np.stack([img, img, img], axis=-1)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img


def infer_disp(model, cfg, img0, img1, scale=1.0, valid_iters=16):
    """img*: HxWx3 uint8. Devuelve (disp HxW f32, t_infer_s, vram_peak_mib)."""
    img0, img1 = to_3ch(img0, scale), to_3ch(img1, scale)
    H, W = img0.shape[:2]
    t = lambda im: torch.as_tensor(im).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t(img0).shape, divis_by=32, force_square=False)
    a, b = padder.pad(t(img0), t(img1))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=bool(cfg.get("mixed_precision", True))):
        t1 = time.perf_counter()
        disp = model(a, b, iters=valid_iters, test_mode=True)
    dt = time.perf_counter() - t1
    peak = torch.cuda.max_memory_allocated() / 2**20
    d = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W)
    if not np.all(np.isfinite(d)):
        print("aviso: AMP produjo no-finitos", file=sys.stderr)
    return d, dt, peak
