"""Regression checks for throughput weighting and exclusion of initialization from stage samples."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_realtime import summarize


class SummaryTests(unittest.TestCase):
    def test_startup_and_lost_samples(self):
        rows=[]
        for index,(wall,state,stage) in enumerate([(100,'INIT',9),(101,'TRACK',.1),(103,'LOST',.3)]):
            rows.append({'state':state,'wall_s':wall,'timestamps_ms':[index*1000],
                'frame_numbers':[index*30],'valid_iters':4,'track_iters':5,'legacy_costs':False,
                'capture_to_publish_s':stage,'fs_model_s':stage,'pose_s':stage,
                'fit':{'resid_med_m':.002,'valid_frac':.8},
                'stages':{'projection_s':stage,'fs_details':{'t_infer_s':stage}}})
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'reports') as tmp:
            path=Path(tmp)/'test.jsonl'
            path.write_text('\n'.join(map(json.dumps,rows)))
            result=summarize(path)
            self.assertTrue(result['sorted_zbuffer'])  # telemetry predating the flag used sorting
            self.assertTrue(result['reference_volumes'])
            self.assertTrue(result['reference_cpu'])
            self.assertTrue(result['reference_tracking'])
            for row in rows:
                row['sorted_zbuffer']=False
                row['scale']=.4
                row['reference_volumes']=False
                row['reference_cpu']=False
                row['reference_tracking']=False
            path.write_text('\n'.join(map(json.dumps,rows)))
            self.assertFalse(summarize(path)['sorted_zbuffer'])
            self.assertEqual(summarize(path)['scale'],.4)
            self.assertFalse(summarize(path)['reference_volumes'])
            self.assertFalse(summarize(path)['reference_cpu'])
            self.assertFalse(summarize(path)['reference_tracking'])
        self.assertAlmostEqual(result['pose_hz'],2/3)
        self.assertEqual(result['lost'],1)
        self.assertAlmostEqual(result['stages_ms']['projection_s']['p50'],200)
        self.assertEqual(result['source_segments'][0]['updates'],2)
        self.assertEqual(result['source_segments'][0]['lost'],1)


if __name__=='__main__':
    unittest.main()
