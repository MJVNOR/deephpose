"""Paired volume and full-disparity equivalence on fixed recorded inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from config import load
from fs_model import load_model, infer_disp, configure_volumes, _REFERENCE_VOLUMES, fs_core
from fs_volumes import build_gwc_volume, build_concat_volume
from prepare_replay import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((a.replay/'manifest.json').read_text())
    model, cfg, _ = load_model(load()['model']['ckpt'], 4)
    captured = {}
    def capture(name, function):
        def wrapped(*args, **kwargs):
            captured[name] = (args, kwargs)
            return function(*args, **kwargs)
        return wrapped
    fs_core.build_gwc_volume = capture('gwc', _REFERENCE_VOLUMES[0])
    fs_core.build_concat_volume = capture('concat', _REFERENCE_VOLUMES[1])
    with np.load(a.replay/manifest['frames'][0]['raw']) as raw:
        infer_disp(model,cfg,raw['left'],raw['right'],.5,4)
    volume_results = {}
    for index, (name, candidate) in enumerate([('gwc',build_gwc_volume),('concat',build_concat_volume)]):
        args, kwargs = captured[name]
        baseline = _REFERENCE_VOLUMES[index]
        reference = baseline(*args, **kwargs)
        output = candidate(*args, **kwargs)
        max_diff = (reference.float()-output.float()).abs().max().item()
        timings = {'reference': [], 'candidate': []}
        for _ in range(3):
            baseline(*args, **kwargs); candidate(*args, **kwargs)
        for repeat in range(20):
            for label, function in ([('reference',baseline),('candidate',candidate)][::1 if repeat%2==0 else -1]):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); function(*args, **kwargs); end.record(); end.synchronize()
                timings[label].append(start.elapsed_time(end))
        volume_results[name] = {'shape':list(args[0].shape),'dtype':str(args[0].dtype),
            'exact':bool(torch.equal(reference,output)), 'max_abs_diff':max_diff,
            'ms_p50':{k:float(np.median(v)) for k,v in timings.items()}}
    captured.clear()
    for reference_mode in (True,False):
        configure_volumes(reference_mode)
        with np.load(a.replay/manifest['frames'][0]['raw']) as raw:
            for _ in range(3): infer_disp(model,cfg,raw['left'],raw['right'],.5,4)
    rows=[]
    for index, frame in enumerate(manifest['frames']):
        outputs={}; infos={}
        with np.load(a.replay/frame['raw']) as raw:
            for mode in ([True,False] if index%2==0 else [False,True]):
                configure_volumes(mode)
                outputs[mode],infos[mode] = infer_disp(model,cfg,raw['left'],raw['right'],.5,4)
        diff=np.abs(outputs[True]-outputs[False])
        rows.append({'frame':frame['frame'],'exact':bool(np.array_equal(outputs[True],outputs[False])),
                     'max_abs_diff_px':float(diff.max()),
                     'reference_ms':infos[True]['t_infer_s']*1000,
                     'candidate_ms':infos[False]['t_infer_s']*1000})
        if index%20==0: print(rows[-1],flush=True)
    result={'scale':.5,'valid_iters':4,'frames':len(rows),'volumes':volume_results,
            'exact_disparity_frames':sum(r['exact'] for r in rows),
            'max_abs_diff_px':max(r['max_abs_diff_px'] for r in rows),
            'model_ms_p50':{label:float(np.median([r[label+'_ms'] for r in rows])) for label in ('reference','candidate')},
            'sources':{str(path):sha256(path) for path in [Path(__file__),Path('scripts/fs_model.py'),Path('scripts/fs_volumes.py'),a.replay/'manifest.json']}}
    (a.out/'frames.json').write_text(json.dumps(rows,indent=2))
    (a.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if result['exact_disparity_frames']!=len(rows):
        raise RuntimeError('Disparities changed; investigate before enabling optimization')


if __name__ == '__main__': main()
