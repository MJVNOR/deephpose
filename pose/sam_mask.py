"""Máscara por texto: SAM 3.1 predeterminado; SAM3 y EfficientSAM3 opcionales.

Uso: sam_mask(rgb_uint8_HxWx3, "computer mouse") -> (mask bool, box xyxy, score).
Scores de SAM son ranking, no confianza (misma regla que el scorer).
"""
import os

import numpy as np
import torch
from PIL import Image

_SAM = None
_EFFICIENT = None
_SAM31 = None
_BACKEND = 'sam3.1'
_THRESHOLD = .5


def configure_sam(backend='sam3.1', threshold=None):
    global _BACKEND, _THRESHOLD
    if backend not in ('sam3.1', 'sam3', 'efficientsam3'):
        raise ValueError('Unknown SAM backend')
    selected_threshold = (.1 if backend == 'efficientsam3' else .5) if threshold is None else threshold
    if not 0 < selected_threshold <= 1:
        raise ValueError('SAM threshold must be in (0, 1]')
    if backend != _BACKEND:
        unload_sam()
    _BACKEND, _THRESHOLD = backend, selected_threshold
    return _THRESHOLD


def prepare_sam(text, shape=(480, 640)):
    """Pay loading and first CUDA kernels before interactive initialization."""
    import time
    start = time.perf_counter()
    blank = np.zeros((*shape, 3), dtype=np.uint8)
    sam_mask(blank, text)
    sam_mask(blank, text)
    torch.cuda.synchronize()
    return time.perf_counter()-start


def unload_sam():
    """Descarga SAM y devuelve su VRAM (patrón bajo demanda: pico 5.4GB en vez de 7.8GB)."""
    global _SAM, _EFFICIENT, _SAM31
    if _SAM31 is not None:
        _SAM31.close()
        _SAM31 = None
    if _EFFICIENT is not None:
        _EFFICIENT.close()
        _EFFICIENT = None
    _SAM = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def load_sam(device="cuda", dtype=torch.bfloat16):
    """Carga perezosa singleton desde caché local (sin red ni token).

    El token solo hizo falta para descargar; en ejecución se fuerza
    HF_HUB_OFFLINE=1. Si la caché falta, el error lo dice explícitamente.
    """
    global _SAM
    if _SAM is not None:
        return _SAM
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import Sam3Model, Sam3Processor
        model = Sam3Model.from_pretrained("facebook/sam3", dtype=dtype).to(device).eval()
        proc = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar SAM3 desde la caché local: {e}") from e
    _SAM = (model, proc)
    return _SAM


def sam_mask(rgb, text, thr=None):
    """Mejor máscara para el texto, o (None, None, None) si no hay nada."""
    global _EFFICIENT, _SAM31
    thr = _THRESHOLD if thr is None else thr
    if _BACKEND == 'sam3.1':
        if _SAM31 is None:
            from sam31_client import SAM31Client
            _SAM31 = SAM31Client()
        return _SAM31.segment(rgb, text, thr)
    if _BACKEND == 'efficientsam3':
        if _EFFICIENT is None:
            from efficient_sam_client import EfficientSAMClient
            _EFFICIENT = EfficientSAMClient()
        return _EFFICIENT.segment(rgb, text, thr)
    model, proc = load_sam()
    img = Image.fromarray(np.ascontiguousarray(rgb).astype(np.uint8)).convert("RGB")
    with torch.no_grad():
        inp = proc(images=img, text=text, return_tensors="pt").to("cuda")
        out = model(**inp)
        res = proc.post_process_instance_segmentation(
            out, threshold=thr, mask_threshold=0.5,
            target_sizes=inp.get("original_sizes").tolist())[0]
    if len(res["masks"]) == 0:
        return None, None, None
    i = int(np.asarray(res["scores"].detach().cpu().float()).argmax())
    mask = np.asarray(res["masks"][i].detach().cpu()).astype(bool)
    box = np.asarray(res["boxes"][i].detach().cpu().float()).round(0).astype(int).tolist()
    score = float(np.asarray(res["scores"][i].detach().cpu().float()))
    return mask, box, score
