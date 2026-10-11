"""Resident SAM 3.1 masks for live initialization and recovery."""
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from compat import adapt_session_creation, enable_windows_attention

ROOT = Path(__file__).resolve().parents[2]


class SAM31Mask:
    def __init__(self):
        checkpoint = ROOT/'data/models/SAM31/sam3.1_multiplex.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Descarga SAM 3.1 con experiments/sam31/download_models.py: {checkpoint}')
        enable_windows_attention()
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        self.predictor = adapt_session_creation(build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint), max_num_objects=1, multiplex_count=16,
            use_fa3=False, use_rope_real=False, compile=False, warm_up=False,
            async_loading_frames=False))
        model = self.predictor.model
        model.batched_grounding_batch_size = 1
        model.postprocess_batch_size = 1
        model.hotstart_delay = 0
        # Decoder FFNs need FP32; convert only the vision trunk.
        model.detector.backbone.vision_backbone.trunk.bfloat16()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def __call__(self, rgb, text, thr=.5):
        # Use an image file: upstream's PIL-list loader swaps width and height.
        with tempfile.TemporaryDirectory(prefix='sam31_frame_') as directory:
            path = Path(directory)/'frame.png'
            Image.fromarray(rgb).save(path)
            session = self.predictor.handle_request(dict(type='start_session',
                resource_path=str(path), offload_video_to_cpu=True))['session_id']
            try:
                output = self.predictor.handle_request(dict(type='add_prompt',
                    session_id=session, frame_index=0, text=text,
                    output_prob_thresh=thr))['outputs']
                masks = np.asarray(output['out_binary_masks'])
                scores = np.asarray(output['out_probs'])
                candidates = [i for i in range(len(masks))
                              if scores[i] >= thr and masks[i].any()]
                if not candidates:
                    return None, None, None
                best = max(candidates, key=lambda i: scores[i])
                mask = masks[best].astype(bool)
                if mask.shape != rgb.shape[:2]:
                    raise ValueError(f'SAM 3.1 mask shape {mask.shape} != RGB {rgb.shape[:2]}')
                y, x = np.nonzero(mask)
                return mask, [int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)], float(scores[best])
            finally:
                self.predictor.handle_request(dict(type='close_session', session_id=session))
                # Release per-image activations for FoundationStereo/FoundationPose.
                torch.cuda.empty_cache()
