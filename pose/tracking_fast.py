"""Tracking adapters and shared crop geometry; upstream files remain unchanged."""
import ast
import inspect
import textwrap
import types
import torch


class _Override:
    def __init__(self, original, name, value):
        self.original, self.name, self.value = original, name, value

    def __getattr__(self, name):
        return self.value if name == self.name else getattr(self.original, name)


class CropWarpCache:
    """Reuse geometry only within one crop, whose transform is immutable."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.matrix = self.key = self.grid = None

    def warp(self, src, M, dsize, mode='bilinear', padding_mode='zeros', align_corners=True, fill_value=None):
        from kornia.geometry.transform import imgwarp as w
        if padding_mode == 'fill':
            return w.warp_perspective(src,M,dsize,mode,padding_mode,align_corners,fill_value)
        B, _, H, W = src.shape
        h, width = dsize
        key = (B,H,W,h,width,src.dtype,src.device)
        if self.matrix is not M or self.key != key:
            transform = w.normalize_homography(M, (H,W), (h,width))
            inverse = w._torch_inverse_cast(transform)
            grid = w.create_meshgrid(h,width,normalized_coordinates=True,device=src.device).to(src.dtype).expand(B,h,width,2)
            self.grid = w.transform_points(inverse[:,None,None],grid)
            self.matrix, self.key = M, key
        return torch.nn.functional.grid_sample(src,self.grid,align_corners=align_corners,
                                              mode=mode,padding_mode=padding_mode)


def crop_with_shared_grid(original):
    function = inspect.unwrap(original)
    namespace = function.__globals__.copy()
    kornia = namespace['kornia']
    cache = CropWarpCache()
    transform = _Override(kornia.geometry.transform, 'warp_perspective', cache.warp)
    geometry = _Override(kornia.geometry, 'transform', transform)
    namespace['kornia'] = _Override(kornia, 'geometry', geometry)
    adapted = types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    adapted.__kwdefaults__ = function.__kwdefaults__
    @torch.inference_mode()
    def cropped(*args, **kwargs):
        cache.reset()
        try:
            return adapted(*args, **kwargs)
        finally:
            cache.reset()  # no grid or pose tensors survive this crop
    return cropped


def without_cache_flush(method):
    """Clone the installed predictor, removing only its two empty_cache calls.

    Keeping the installed function body avoids maintaining a second copy of the
    refinement algorithm. Validate the expected upstream structure before using it.
    The original method and torch.cuda.empty_cache remain untouched.
    """
    original = inspect.unwrap(method.__func__)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef):
        raise RuntimeError('Unsupported refinement source')
    removed = []
    class RemoveFlush(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == 'torch.cuda.empty_cache'
                    and not node.value.args and not node.value.keywords):
                removed.append(node)
                return ast.copy_location(ast.Pass(), node)
            return self.generic_visit(node)
    tree = RemoveFlush().visit(tree)
    if len(removed) != 2:
        raise RuntimeError(f'Expected two refinement cache flushes, found {len(removed)}')
    function.decorator_list = []
    namespace = original.__globals__.copy()
    namespace['make_crop_data_batch'] = crop_with_shared_grid(namespace['make_crop_data_batch'])
    exec(compile(ast.fix_missing_locations(tree), '<tracking_fast:predict>', 'exec'), namespace)
    adapted = torch.inference_mode()(namespace[function.name])
    return types.MethodType(adapted, method.__self__)


def track_one_fast(est, rgb, depth, K, iteration):
    # Same preprocessing, refinement and final frame conversion as track_one.
    # Keep pose_last on GPU instead of its GPU->NumPy->GPU round trip.
    import estimater as upstream
    if est.pose_last is None:
        raise RuntimeError('Initialize pose before tracking')
    if est.debug >= 2:
        return est.track_one(rgb=rgb, depth=depth, K=K, iteration=iteration)
    if not hasattr(est, '_tracking_predict'):
        est._tracking_predict = without_cache_flush(est.refiner.predict)
    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = upstream.erode_depth(depth, radius=2, device='cuda')
    depth = upstream.bilateral_filter_depth(depth, radius=2, device='cuda')
    xyz_map = upstream.depth2xyzmap_batch(
        depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=float('inf'))[0]
    pose, _ = est._tracking_predict(
        mesh=est.mesh, mesh_tensors=est.mesh_tensors, rgb=rgb, depth=depth, K=K,
        ob_in_cams=est.pose_last.reshape(1,4,4), normal_map=None, xyz_map=xyz_map,
        mesh_diameter=est.diameter, glctx=est.glctx, iteration=iteration, get_vis=False)
    est.pose_last = pose
    return (pose @ est.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)
