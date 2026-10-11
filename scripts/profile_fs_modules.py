"""Isolated FS module CUDA-event timings and operator profile; no production hooks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from config import load
from fs_model import load_model,infer_disp,to_3ch,InputPadder
import core.foundation_stereo as fs_core


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    model,cfg,_=load_model(load()['model']['ckpt'],4)
    with np.load(a.raw) as raw:
        left,right=raw['left'],raw['right']
    results={}
    for scale in [.5,.4]:
        for _ in range(3): infer_disp(model,cfg,left,right,scale,4)
        records=[]; starts={}; handles=[]; patched=[]
        def timed(name,function):
            def wrapped(*args,**kwargs):
                start=torch.cuda.Event(enable_timing=True);start.record()
                result=function(*args,**kwargs)
                end=torch.cuda.Event(enable_timing=True);end.record()
                records.append((name,start,end))
                return result
            return wrapped
        for name in ['normalize_image','build_gwc_volume','build_concat_volume','disparity_regression','context_upsample']:
            original=getattr(fs_core,name)
            patched.append((fs_core,name,original));setattr(fs_core,name,timed(name,original))
        for name in ['__init__','__call__']:
            cls=fs_core.Combined_Geo_Encoding_Volume;original=getattr(cls,name)
            patched.append((cls,name,original));setattr(cls,name,timed('geo_volume'+name,original))
        def before(name):
            def hook(module,args):
                event=torch.cuda.Event(enable_timing=True); event.record(); starts[name]=event
            return hook
        def after(name):
            def hook(module,args,output):
                event=torch.cuda.Event(enable_timing=True); event.record(); records.append((name,starts.pop(name),event))
            return hook
        for name,module in model.named_children():
            handles.extend([module.register_forward_pre_hook(before(name)),module.register_forward_hook(after(name))])
        calls=[]; totals=[]
        try:
            for _ in range(5):
                records.clear()
                _,info=infer_disp(model,cfg,left,right,scale,4)
                torch.cuda.synchronize()
                sample={}
                for name,start,end in records: sample[name]=sample.get(name,0)+start.elapsed_time(end)
                calls.append(sample); totals.append(info['t_infer_s']*1000)
        finally:
            for handle in handles: handle.remove()
            for obj,name,original in patched: setattr(obj,name,original)
        shape=to_3ch(left,scale).shape[:2]
        padded=InputPadder((1,3,*shape),divis_by=32,force_square=False).pad(torch.zeros(1,3,*shape))[0].shape
        results[str(scale)]={'input_hw':shape,'padded_hw':list(padded[-2:]),
            'model_ms_p50':float(np.median(totals)),
            'modules_ms_p50':{k:float(np.median([s.get(k,0) for s in calls])) for k in calls[0]}}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            infer_disp(model,cfg,left,right,scale,4)
        events=prof.key_averages()
        (a.out/f'operators_scale{scale}.txt').write_text(events.table(sort_by='self_cuda_time_total',row_limit=30))
        prof.export_chrome_trace(str(a.out/f'trace_scale{scale}.json'))
        print(json.dumps(results[str(scale)],indent=2),flush=True)
    (a.out/'summary.json').write_text(json.dumps({'note':'Five warmed inferences per scale; CUDA-event intervals for immediate-child modules and selected top-level functional work. Repeated calls accumulate per inference. Operator profiler is a separate call, not an FPS measurement.',
        'supported_profiler_activities':[str(x) for x in torch.profiler.supported_activities()],
        'cuda_kernel_trace_available':torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities(),
        'scales':results},indent=2))


if __name__=='__main__': main()
