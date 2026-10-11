"""Isolated EfficientSAM3 EV-M backend with strict checkpoint loading."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT/'data/models/EfficientSAM3/efficientsam3_ft/efficientsam3_efficientvit.pt'


class EfficientMask:
    def __init__(self, variant='ev', precision='bf16'):
        from sam3.model_builder import build_efficientsam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        self.precision = precision
        self.checkpoint = CHECKPOINT if variant == 'ev' else CHECKPOINT.with_name('efficientsam3_tinyvit.pt')
        self.model = build_efficientsam3_image_model(
            device='cpu', backbone_type='efficientvit' if variant == 'ev' else 'tinyvit',
            model_name='b1' if variant == 'ev' else '11m',
            text_encoder_type='MobileCLIP-S0', text_encoder_context_length=16,
            checkpoint_path=None, load_from_HF=False, enable_inst_interactivity=False)
        saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)['model']
        cleaned = {k.removeprefix('detector.').replace('student_trunk.', ''): v for k,v in saved.items()}
        self.model.load_state_dict(cleaned, strict=True)
        self.model.cuda().eval()
        self.processor = Sam3Processor(self.model, device='cuda', confidence_threshold=.5)
        self.parameters = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, rgb, text, thr=.5):
        self.processor.confidence_threshold = thr
        with torch.autocast('cuda', dtype=torch.bfloat16) if self.precision == 'bf16' else nullcontext():
            state = self.processor.set_image(Image.fromarray(np.ascontiguousarray(rgb)))
            state = self.processor.set_text_prompt(text, state)
        if not len(state['scores']):
            return None, None, None
        index = int(state['scores'].argmax().item())
        mask = state['masks'][index].squeeze(0).detach().cpu().numpy().astype(bool)
        box = state['boxes'][index].float().detach().cpu().numpy().round().astype(int).tolist()
        score = float(state['scores'][index].item())
        return mask, box, score
