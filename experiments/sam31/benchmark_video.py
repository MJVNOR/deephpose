"""SAM 3.1 text-prompted video experiment on frozen replay RGB frames."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=161)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--weights-dtype', choices=['fp32','bf16-vision'], default='bf16-vision')
    parser.add_argument('--hotstart-delay', type=int, default=15, help='Buffer temporal oficial; 0 entrega sin esperar frames futuros')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    images = args.out/'inputs'
    images.mkdir()
    source = ROOT/'data/processed/replay_fs_sweep_20261010/fs4'
    manifest = json.loads((source/'manifest.json').read_text())
    entries = manifest['frames'][:args.frames]
    for i, entry in enumerate(entries):
        with np.load(source/entry['bundle']) as bundle:
            cv2.imwrite(str(images/f'{i:06d}.png'), cv2.cvtColor(bundle['rgb'], cv2.COLOR_RGB2BGR))
    summary = dict(mode='offline text-prompted video; cached RGB, no FS or pose loaded',
                   checkpoint_revision='daa63191845a41281374e725f4c9e51c7a824460',
                   upstream_commit='0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc',
                   source_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                   torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                   compile=False, flash_attention_3=False, use_rope_real=False, max_objects=1,
                   multiplex_count=16, grounding_batch_size=1, prompt='computer mouse', runs=[])
    summary['weights_dtype'] = args.weights_dtype
    summary['hotstart_delay'] = args.hotstart_delay

    def save():
        (args.out/'summary.json').write_text(json.dumps(summary, indent=2))

    save()
    try:
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        from compat import enable_windows_attention
        enable_windows_attention()
        summary['decoder_sdpa_backends'] = ['EFFICIENT_ATTENTION', 'MATH']
        loads = []
        original_load = torch.nn.Module.load_state_dict

        def audit(model, *a, **kw):
            result = original_load(model, *a, **kw)
            loads.append(dict(model=type(model).__name__, missing=list(result.missing_keys),
                              unexpected=list(result.unexpected_keys)))
            return result

        start = time.perf_counter()
        with patch.object(torch.nn.Module, 'load_state_dict', audit):
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=str(ROOT/'data/models/SAM31/sam3.1_multiplex.pt'),
                max_num_objects=1, multiplex_count=16, use_fa3=False, use_rope_real=False,
                compile=False, warm_up=False, async_loading_frames=False)
        torch.cuda.synchronize()
        summary['model_load_s'] = time.perf_counter()-start
        summary['checkpoint_loads'] = loads
        if not loads or loads[-1]['missing'] or loads[-1]['unexpected']:
            raise RuntimeError('Final SAM 3.1 model has unmatched checkpoint keys')
        predictor.model.batched_grounding_batch_size = 1
        predictor.model.postprocess_batch_size = 1
        predictor.model.hotstart_delay = args.hotstart_delay
        if args.weights_dtype == 'bf16-vision':
            # Decoder FFNs explicitly disable autocast and require FP32 weights.
            # Compress only the large visual trunk, preserving complex RoPE.
            predictor.model.detector.backbone.vision_backbone.trunk.bfloat16()
            torch.cuda.empty_cache()
        from compat import adapt_session_creation
        adapt_session_creation(predictor)
        summary['parameters'] = sum(p.numel() for p in predictor.model.parameters())
        save()
        for repeat in range(args.repeats):
            run = dict(repeat=repeat, frames=[])
            summary['runs'].append(run)
            session = predictor.handle_request(dict(type='start_session', resource_path=str(images.resolve()),
                                                     offload_video_to_cpu=True))['session_id']
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            initial = predictor.handle_request(dict(type='add_prompt', session_id=session,
                                                     frame_index=0, text='computer mouse'))
            torch.cuda.synchronize()
            run['prompt_s'] = time.perf_counter()-start
            run['initial_objects'] = len(initial['outputs']['out_obj_ids'])
            start = previous = time.perf_counter()
            stream = predictor.handle_stream_request(dict(type='propagate_in_video', session_id=session,
                propagation_direction='forward', start_frame_index=0, evict_cached_frame_outputs=True))
            for output in stream:
                torch.cuda.synchronize()
                now = time.perf_counter()
                index = output['frame_index']
                masks = np.asarray(output['outputs']['out_binary_masks'])
                scores = np.asarray(output['outputs'].get('out_probs', []))
                row = dict(index=index, source_frame=entries[index]['frame'],
                           delivery_s=now-previous, objects=len(masks),
                           object_ids=np.asarray(output['outputs']['out_obj_ids']).tolist(),
                           mask_pixels=[int(m.sum()) for m in masks], scores=scores.tolist())
                if len(masks):
                    best = int(scores.argmax()) if len(scores) else 0
                    np.save(args.out/f"mask_r{repeat}_{entries[index]['frame']}.npy", masks[best])
                    reference = ROOT/f"reports/sam3_reference_20261011/mask_{entries[index]['frame']}.npy"
                    if reference.exists():
                        ref = np.load(reference).astype(bool)
                        mask = masks[best].astype(bool)
                        row['iou_with_sam3_image_mask'] = float((mask & ref).sum()/max((mask | ref).sum(),1))
                run['frames'].append(row)
                print(row, flush=True)
                save()
                previous = time.perf_counter()  # exclude local persistence from next delivery latency
            run['propagation_wall_s_including_output_io'] = time.perf_counter()-start
            run['delivery_sum_s'] = sum(r['delivery_s'] for r in run['frames'])
            run['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
            run['detected_frames'] = sum(any(r['mask_pixels']) for r in run['frames'])
            predictor.handle_request(dict(type='close_session', session_id=session))
            save()
    except BaseException as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


if __name__ == '__main__':
    main()
