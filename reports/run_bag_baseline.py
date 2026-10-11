"""Observational replay runner: preserves live_pose logic; saves telemetry and sparse frames."""
import builtins
import hashlib
import inspect
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'bag_baseline_track5_observed'
OUT.mkdir(exist_ok=True)
original_print = builtins.print
started = time.perf_counter()
last_sample = -1e9
records = []

def observed_print(*args, **kwargs):
    global last_sample
    original_print(*args, **kwargs)
    if not args or not isinstance(args[0], str):
        return
    frame = inspect.currentframe().f_back
    if Path(frame.f_code.co_filename).name != 'live_pose.py':
        return
    loc = frame.f_locals
    if not (args[0].startswith('f') and 't_pose=' in args[0] and 'bd' in loc):
        return
    import cv2
    now = time.perf_counter()
    bd = loc['bd']
    row = {'n':loc['n'], 'state':loc['state'], 'wall_s':now-started,
           'fps_hud':loc['fps'], 'pose_s':loc['dt'], 'fs_model_s':loc['t_infer'],
           'latency_pre_overlay_s':loc['t_lat'], 'queue_before_fs_s':loc['t_age'],
           'capture_to_publish_s':now-bd['t'], 'frame_numbers':bd['fn'],
           'timestamps_ms':bd['ts'], 'fit':loc['fit'], 'pose':loc['pose'].tolist()}
    if now-last_sample >= 5 or row['n'] == 1:
        name = f"frame_{row['n']:04d}_{row['state']}.jpg"
        cv2.imwrite(str(OUT/name), cv2.cvtColor(loc['vis'], cv2.COLOR_RGB2BGR))
        row['sample'] = name
        last_sample = now
    records.append(row)
    with (OUT/'frames.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row)+'\n')

if __name__ == '__main__':
    if (OUT/'frames.jsonl').exists():
        raise SystemExit('Output exists; choose a new run directory to preserve prior evidence.')
    (OUT/'manifest.json').write_text(json.dumps({
        'argv':sys.argv[1:], 'source_sha256':hashlib.sha256((ROOT/'pose/live_pose.py').read_bytes()).hexdigest(),
        'mode':'unchanged application with print observer; sparse JPEG I/O adds overhead; not deterministic offline playback'
    },indent=2))
    builtins.print = observed_print
    sys.argv[0] = str(ROOT/'pose/live_pose.py')
    try:
        runpy.run_path(str(ROOT/'pose/live_pose.py'),run_name='__main__')
    finally:
        builtins.print = original_print
        (OUT/'completion.json').write_text(json.dumps({'wall_s':time.perf_counter()-started,'records':len(records)}))
