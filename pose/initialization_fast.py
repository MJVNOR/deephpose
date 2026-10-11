"""Exact crop reuse and an optional experimental coarse-to-fine search."""
import inspect
import types
import numpy as np
import torch
from tracking_fast import crop_with_shared_grid


def shortlist_quality(depth, rendered, mask):
    silhouette = rendered > .001
    valid = silhouette & np.isfinite(depth) & (depth > .001)
    coverage = float(valid.sum()/max(1, silhouette.sum()))
    residual = float(np.median(np.abs(depth[valid]-rendered[valid]))) if valid.any() else None
    iou = float((silhouette & mask).sum()/max(1, (silhouette | mask).sum()))
    return dict(valid_fraction=coverage, residual_m=residual, mask_iou=iou,
                accepted=coverage >= .3 and residual is not None and residual <= .02 and iou >= .5)


def spread_rotation_indices(rotations, count):
    """Greedy farthest-point coverage of SO(3), deterministic for a fixed grid."""
    count = min(count, len(rotations))
    if count < 1:
        raise ValueError('At least one orientation is required')
    matrices = np.asarray(rotations)[:, :3, :3]
    selected = []
    nearest_similarity = np.full(len(matrices), -np.inf)
    for _ in range(count):
        index = 0 if not selected else int(np.argmin(nearest_similarity))
        selected.append(index)
        similarity = np.einsum('ij,kij->k', matrices[index], matrices)
        nearest_similarity = np.maximum(nearest_similarity, similarity)
        nearest_similarity[selected] = np.inf
    return np.asarray(selected)


def register_with_shortlist(est, *, hypotheses=84, finalists=8, progress=None, **kwargs):
    """Experimental coarse-to-fine search; fewer candidates can change the pose."""
    if hypotheses < 1 or finalists < 1 or kwargs.get('iteration', 5) < 1:
        raise ValueError('Search sizes and iterations must be positive')
    originals = (est.refiner.predict, est.scorer.predict, est.rot_grid)
    try:
        refine = shared_crop_predictor(originals[0])
        score = shared_crop_predictor(originals[1])
        indices = spread_rotation_indices(est.rot_grid.detach().cpu().numpy(), hypotheses)
        est.rot_grid = est.rot_grid[indices]

        def staged_refine(**options):
            if progress:
                progress('explorando', len(options['ob_in_cams']))
            iterations = options['iteration']
            coarse = dict(options, iteration=1, get_vis=False)
            poses, _ = refine(**coarse)
            rank_options = {key: options[key] for key in (
                'mesh', 'rgb', 'depth', 'K', 'normal_map', 'mesh_tensors', 'glctx', 'mesh_diameter')}
            scores, _ = score(**rank_options, ob_in_cams=poses.detach().cpu().numpy(), get_vis=False)
            chosen = torch.as_tensor(scores, device=poses.device).argsort(descending=True)[:finalists]
            poses = poses[chosen]
            if iterations > 1:
                if progress:
                    progress('refinando', len(poses))
                final = dict(options, ob_in_cams=poses.detach().cpu().numpy(), iteration=iterations-1)
                return refine(**final)
            return poses, None

        def final_score(**options):
            if progress:
                progress('seleccionando', len(options['ob_in_cams']))
            return score(**options)

        est.refiner.predict, est.scorer.predict = staged_refine, final_score
        return est.register(**kwargs)
    finally:
        est.refiner.predict, est.scorer.predict, est.rot_grid = originals


def shared_crop_predictor(method):
    original = inspect.unwrap(method.__func__)
    namespace = original.__globals__.copy()
    namespace['make_crop_data_batch'] = crop_with_shared_grid(namespace['make_crop_data_batch'])
    adapted = types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    adapted.__kwdefaults__ = original.__kwdefaults__
    return types.MethodType(torch.inference_mode()(adapted),method.__self__)


def register_with_shared_crops(est, *, progress=None, reference=False, **kwargs):
    originals = (est.refiner.predict, est.scorer.predict)
    def phase(name, method):
        def call(*args, **options):
            if progress:
                progress(name, len(options['ob_in_cams']))
            return method(*args, **options)
        return call
    try:
        refiner, scorer = originals if reference else tuple(shared_crop_predictor(m) for m in originals)
        est.refiner.predict = phase('refinando', refiner)
        est.scorer.predict = phase('seleccionando', scorer)
        return est.register(**kwargs)
    finally:
        est.refiner.predict, est.scorer.predict = originals
