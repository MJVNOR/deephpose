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
    """img*: HxW uint8 (mono o 3ch). Devuelve (disp HxW f32, info dict).

    info: t_h2d_s (numpy->cuda+pad), t_infer_s (solo modelo, CUDA events),
    t_d2h_s (unpad+cuda->cpu+numpy), t_total_s (pared con sync),
    vram_alloc_mib (tensores ahora), vram_reserved_mib (pool ahora),
    vram_peak_alloc_mib (pico desde reset: pesos residentes + activaciones),
    vram_peak_reserved_mib (pico reservado: lo anterior + caché del allocator,
    no memoria de otros procesos).
    """
    t_all = time.perf_counter()
    img0, img1 = to_3ch(img0, scale), to_3ch(img1, scale)
    H, W = img0.shape[:2]
    mk = lambda im: torch.as_tensor(im).cuda().float()[None].permute(0, 3, 1, 2)
    t0 = time.perf_counter()
    padder = InputPadder(mk(img0).shape, divis_by=32, force_square=False)
    a, b = padder.pad(mk(img0), mk(img1))
    torch.cuda.synchronize()
    t_h2d = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=bool(cfg.get("mixed_precision", True))):
        ev0.record()
        disp = model(a, b, iters=valid_iters, test_mode=True)
        ev1.record()
    ev1.synchronize()  # ponytail: sin sync el tiempo GPU infra-mide
    t_infer = ev0.elapsed_time(ev1) / 1e3
    peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    peak_res = torch.cuda.max_memory_reserved() / 2**20
    t2 = time.perf_counter()
    d = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W)
    t_d2h = time.perf_counter() - t2
    info = {"t_h2d_s": round(t_h2d, 3), "t_infer_s": round(t_infer, 3),
            "t_d2h_s": round(t_d2h, 3), "t_total_s": round(time.perf_counter() - t_all, 3),
            "vram_alloc_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
            "vram_reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
            "vram_peak_alloc_mib": round(peak_alloc, 1),
            "vram_peak_reserved_mib": round(peak_res, 1)}
    if not np.all(np.isfinite(d)):
        print("aviso: AMP produjo no-finitos", file=sys.stderr)
    return d, info
