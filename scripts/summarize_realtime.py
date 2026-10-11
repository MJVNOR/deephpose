"""Summarize full replay telemetry, excluding startup before first publication."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if len(rows)<2:
        raise ValueError(f'Not enough updates: {path}')
    elapsed=rows[-1]['wall_s']-rows[0]['wall_s']
    samples=rows[1:]  # first displayed interval includes camera warmup
    result={'run':path.stem,'updates':len(rows),'lost':sum(r['state']=='LOST' for r in rows),
            'lost_percent':100*sum(r['state']=='LOST' for r in rows)/len(rows),
            'pose_hz':(len(rows)-1)/elapsed,'measured_wall_s':elapsed,
            'source_span_s':(rows[-1]['timestamps_ms'][0]-rows[0]['timestamps_ms'][0])/1000,
            'first_frame':rows[0]['frame_numbers'][0],'last_frame':rows[-1]['frame_numbers'][0],
            'fs_iters':rows[0]['valid_iters'],'track_iters':rows[0]['track_iters'],
            'legacy_costs':rows[0]['legacy_costs'],'sorted_zbuffer':rows[0].get('sorted_zbuffer',True),
            'reference_volumes':rows[0].get('reference_volumes',True),
            'reference_cpu':rows[0].get('reference_cpu',True),
            'reference_tracking':rows[0].get('reference_tracking',True),
            'scale':rows[0].get('scale',.5)}
    for field in ['capture_to_publish_s','fs_model_s','pose_s']:
        values=[r[field]*1000 for r in samples]
        result[field.replace('_s','_ms')]={'p50':float(np.median(values)),'p95':float(np.percentile(values,95))}
    result['residual_mm_p50']=float(np.median([r['fit']['resid_med_m']*1000 for r in samples if r['fit']['resid_med_m'] is not None]))
    tracked=[r for r in samples if r['state'] in ('TRACK','LOST')]
    if tracked and 'stages' in tracked[0]:
        def percentiles(values):
            return {'p50':float(np.median(values)), 'p95':float(np.percentile(values,95))}
        result['stages_ms']={k:percentiles([r['stages'][k]*1000 for r in tracked])
                             for k,v in tracked[0]['stages'].items() if isinstance(v,(int,float))}
        result['fs_details_ms']={k:percentiles([r['stages']['fs_details'][k]*1000 for r in tracked])
                                 for k in tracked[0]['stages']['fs_details'] if k.endswith('_s')}
        result['source_segments']=[]
        first_time=rows[0]['timestamps_ms'][0]
        for start in range(0,int(result['source_span_s'])+1,20):
            segment=[r for r in tracked if start <= (r['timestamps_ms'][0]-first_time)/1000 < start+20]
            if not segment:
                continue
            result['source_segments'].append({'relative_source_s':[start,start+20], 'updates':len(segment),
                'lost':sum(r['state']=='LOST' for r in segment),
                'latency_ms':percentiles([r['capture_to_publish_s']*1000 for r in segment]),
                'valid_frac_p50':float(np.median([r['fit']['valid_frac'] for r in segment]))})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    results=[summarize(path) for path in sorted(a.directory.glob('*.jsonl'))]
    (a.directory/'metrics_summary.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
