"""Wrapper fino de FoundationPose model-based. Corre en el env `pose/`.

Convenciones: RGB uint8 HxWx3, profundidad metros float32 (inf=inválido),
K 3x3 del RGB rectificado, pose 4x4 objeto→cámara. La apariencia original
(texturas) se usa para estimar y superponer el CAD; el rojo es una alternativa visual.
El ranking de hipótesis del scorer NO es una probabilidad.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

_CODE = Path(__file__).resolve().parent / "FoundationPose"
sys.path.insert(0, str(_CODE))
from estimater import FoundationPose  # noqa: E402
from Utils import draw_posed_3d_box, draw_xyz_axis  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402
from mesh_initialization import bounded_reset_object  # noqa: E402


class MemoryBoundFoundationPose(FoundationPose):
    reset_object = bounded_reset_object(FoundationPose.reset_object)


def sample_diffuse_to_vertex_colors(part):
    """Parte con TextureVisuals -> colores por vértice desde su map_Kd."""
    vis = part.visual
    img = np.asarray(vis.material.image.convert("RGB"), dtype=np.float32)
    H, W = img.shape[:2]
    uv = np.asarray(vis.uv, dtype=np.float64)
    xs = np.clip((uv[:, 0] * (W - 1)).round().astype(int), 0, W - 1)
    ys = np.clip(((1.0 - uv[:, 1]) * (H - 1)).round().astype(int), 0, H - 1)
    part.visual = trimesh.visual.ColorVisuals(img[ys, xs].astype(np.uint8))
    return part


def load_textured_mesh(obj_path):
    """OBJ multitela -> un solo Trimesh con vertex_colors (cara/UV intactos en origen)."""
    scene = trimesh.load(obj_path, process=True)
    parts = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
    colored = []
    for g in parts:
        if isinstance(getattr(g, "visual", None), trimesh.visual.texture.TextureVisuals):
            g = sample_diffuse_to_vertex_colors(g)
        elif getattr(getattr(g, "visual", None), "vertex_colors", None) is None:
            g.visual = trimesh.visual.ColorVisuals(
                np.full((len(g.vertices), 3), 128, np.uint8))
        colored.append(g)
    mesh = trimesh.util.concatenate(colored)
    assert mesh.vertex_normals is not None and len(mesh.vertex_normals) == len(mesh.vertices)
    return mesh


def depth_to_fp(depth_m):
    """Borde API: inf/NaN->0 (FP usa <0.001 como inválido). La máscara original se conserva aparte."""
    d = np.asanyarray(depth_m, dtype=np.float32)
    out = np.where(np.isfinite(d), d, 0).astype(np.float32)
    return out


def make_estimator(mesh, weights_dir, debug_dir, debug=0):
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = MemoryBoundFoundationPose(model_pts=np.asarray(mesh.vertices), model_normals=np.asarray(mesh.vertex_normals),
                         mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx,
                         debug=debug, debug_dir=str(debug_dir))
    from mesh_box import base_aligned_box
    est.display_box = base_aligned_box(mesh)
    return est


def prepare_fast_pose(est, K, shape):
    """Warm real registration kernels on synthetic input; retain no dummy pose."""
    from initialization_fast import register_with_shortlist
    missing = object()
    names = ('pose_last', 'poses', 'scores', 'best_id', 'H', 'W', 'K', 'ob_id', 'ob_mask')
    previous = {name: getattr(est, name, missing) for name in names}
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    mask[height//3:2*height//3, width//3:2*width//3] = True
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        register_with_shortlist(est, hypotheses=42, finalists=8, K=np.asarray(K, float),
            rgb=np.zeros((height, width, 3), dtype=np.uint8), depth=np.full(shape, .5, dtype=np.float32),
            ob_mask=mask, iteration=5)
        torch.cuda.synchronize()
        return time.perf_counter()-start
    finally:
        for name, value in previous.items():
            if value is missing:
                if hasattr(est, name):
                    delattr(est, name)
            else:
                setattr(est, name, value)


def estimate_pose(est, K, rgb, depth_m, mask, iters=5, reference=False, progress=None, fast_init=False):
    """Init global: coarse + refine(iters) + scorer. Devuelve (pose 4x4, segundos, vram_mib)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    from initialization_fast import register_with_shared_crops, register_with_shortlist, shortlist_quality
    if fast_init and reference:
        raise ValueError('fast_init and reference cannot be combined')
    options = dict(K=np.asarray(K, float), rgb=np.ascontiguousarray(rgb),
                   depth=depth_to_fp(depth_m), ob_mask=np.asarray(mask).astype(bool), iteration=int(iters))
    est.init_stats = dict(fast=fast_init, fallback=False)
    if fast_init:
        pose = register_with_shortlist(est, hypotheses=42, finalists=8, progress=progress, **options)
        if est.pose_last is None:
            raise RuntimeError('Initial registration produced no pose')
        rendered = render_depth(est, K, *depth_m.shape)
        quality = shortlist_quality(depth_m, rendered, options['ob_mask'])
        est.init_stats.update(quality=quality, hypotheses=42, finalists=8)
        if not quality['accepted']:
            est.init_stats['fallback'] = True
            if progress:
                progress('busqueda completa', 252)
            pose = register_with_shared_crops(est, progress=progress, **options)
    else:
        pose = register_with_shared_crops(est, reference=reference, progress=progress, **options)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    vram = torch.cuda.memory_allocated() / 2**20
    return np.asarray(pose, float).reshape(4, 4), dt, vram


def track_pose(est, K, rgb, depth_m, iters=1, reference=False):
    """Seguimiento: solo refina desde pose_last, sin búsqueda global."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if reference:
        pose = est.track_one(rgb=np.ascontiguousarray(rgb), depth=depth_to_fp(depth_m),
                             K=np.asarray(K, float), iteration=int(iters))
    else:
        from tracking_fast import track_one_fast
        pose = track_one_fast(est, np.ascontiguousarray(rgb), depth_to_fp(depth_m),
                              np.asarray(K, float), int(iters))
    torch.cuda.synchronize()
    return np.asarray(pose, float).reshape(4, 4), time.perf_counter() - t0


def render_depth(est, K, H, W, pose=None):
    """Profundidad renderizada del CAD (para métricas/overlay).

    OJO frames: est.mesh_tensors guarda la malla CENTRADA; la pose para
    renderizar es est.pose_last (también centrada), NO best_pose 4x4
    (esa vive en el frame original del OBJ).
    """
    import Utils as U
    if pose is None:
        pose = est.pose_last
    ob = pose.reshape(1, 4, 4) if torch.is_tensor(pose) else torch.as_tensor(
        np.asarray(pose, float).reshape(1, 4, 4), device="cuda", dtype=torch.float)
    ob = ob.to(device="cuda", dtype=torch.float)  # upstream opera en cuda
    _, depth_r, _ = U.nvdiffrast_render(K=np.asarray(K, float), H=H, W=W, ob_in_cams=ob,
                                          mesh_tensors=est.mesh_tensors, glctx=est.glctx)
    return np.asarray(depth_r.detach().cpu().numpy()).reshape(H, W)


def fit_metrics(depth_obs_m, K, pose, mesh_diameter, est=None, depth_render=None):
    """Métricas de ajuste (NO confianza): cobertura válida, residual mediano, diámetro."""
    if depth_render is None:
        H, W = depth_obs_m.shape[:2]
        depth_render = render_depth(est, K, H, W)  # pose_last centrada, ver render_depth
    sil = depth_render > 0.001
    obs = np.asanyarray(depth_obs_m, dtype=np.float64)
    ok = sil & np.isfinite(obs) & (obs > 0.001)
    frac = float(ok.sum() / max(sil.sum(), 1))
    resid = float(np.median(np.abs(obs[ok] - depth_render[ok])) if ok.any() else np.nan)
    return {"sil_px": int(sil.sum()), "valid_frac": round(frac, 4),
            "resid_med_m": round(resid, 4) if np.isfinite(resid) else None,
            "mesh_diameter_m": round(float(mesh_diameter), 4)}


def render_textured(est, K, H, W, pose=None, return_depth=False):
    """(silueta bool, RGB renderizado 0-255) del CAD con sus colores de vértice."""
    import Utils as U
    if pose is None:
        pose = est.pose_last
    ob = pose.reshape(1, 4, 4) if torch.is_tensor(pose) else torch.as_tensor(
        np.asarray(pose, float).reshape(1, 4, 4), device="cuda", dtype=torch.float)
    ob = ob.to(device="cuda", dtype=torch.float)
    color_r, depth_r, _ = U.nvdiffrast_render(K=np.asarray(K, float), H=H, W=W, ob_in_cams=ob,
                                              mesh_tensors=est.mesh_tensors, glctx=est.glctx)
    rgb = (color_r.detach().cpu().numpy().reshape(H, W, 3).clip(0, 1) * 255).astype(np.uint8)
    depth = depth_r.detach().cpu().numpy().reshape(H, W)
    return (depth, rgb) if return_depth else (depth > 0.001, rgb)


def draw_textured_overlay(rgb, K, pose, est, bbox, alpha=0.5):
    """Foto + CAD texturizado semitransparente + caja + ejes (alternativa al rojo)."""
    H, W = rgb.shape[:2]
    sil, ren = render_textured(est, K, H, W)
    vis = np.ascontiguousarray(rgb).copy()
    vis[sil] = (alpha * ren[sil] + (1 - alpha) * vis[sil]).astype(np.uint8)
    Pc = np.asarray(pose).reshape(4, 4).copy()
    box = getattr(est, 'display_box', None)
    if box is not None:
        Pc = Pc @ box.to_object
        bbox = box.bounds
    vis = draw_posed_3d_box(np.asarray(K, float), img=vis, ob_in_cam=Pc, bbox=bbox)
    Pc[:3, 3] += Pc[:3, :3] @ ((bbox[0] + bbox[1]) / 2)
    vis = draw_xyz_axis(vis, ob_in_cam=Pc, scale=0.1, K=np.asarray(K, float),
                        thickness=3, transparency=0, is_input_rgb=True)
    return vis


def draw_overlay(rgb, K, pose, est, bbox, axis_scale=0.1, alpha=0.5, depth_render=None, fast=False, color_render=None):
    """RGB + CAD semitransparente (rojo si no hay color_render) + caja/ejes."""
    H, W = rgb.shape[:2]
    if depth_render is None:
        depth_render = render_depth(est, K, H, W)
    sil = depth_render > 0.001  # silueta en frame centrado; caja/ejes usan pose original
    vis = np.ascontiguousarray(rgb).copy()
    if color_render is None:
        color_render = np.zeros_like(vis)
        color_render[..., 0] = 255
    vis[sil] = (alpha * color_render[sil] + (1 - alpha) * vis[sil]).astype(np.uint8)
    Pc = np.asarray(pose).reshape(4, 4).copy()  # ejes en el centro (el origen OBJ cae fuera de frame)
    box = getattr(est, 'display_box', None)
    if box is not None:
        Pc = Pc @ box.to_object
        bbox = box.bounds
    vis = draw_posed_3d_box(np.asarray(K, float), img=vis, ob_in_cam=Pc, bbox=bbox)
    Pc[:3, 3] += Pc[:3, :3] @ ((bbox[0] + bbox[1]) / 2)
    if fast:
        from overlay_fast import draw_opaque_axes
        vis = draw_opaque_axes(vis, Pc, np.asarray(K, float), float(axis_scale))
    else:
        vis = draw_xyz_axis(vis, ob_in_cam=Pc,
                            scale=float(axis_scale), K=np.asarray(K, float),
                            thickness=3, transparency=0, is_input_rgb=True)
    return vis, sil
