import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pose'))
sys.path.insert(0, str(ROOT/'experiments/sam31'))
import sam_mask
spec = importlib.util.spec_from_file_location('sam31_backend', ROOT/'experiments/sam31/backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class SAM31Tests(unittest.TestCase):
    def test_default_routes_to_sam31_without_fallback(self):
        fake = Mock()
        with patch.object(sam_mask, '_SAM31', fake), patch.object(sam_mask, '_BACKEND', 'sam3.1'), \
             patch.object(sam_mask, '_THRESHOLD', .5), patch.object(sam_mask, 'load_sam') as old:
            self.assertEqual(sam_mask.configure_sam(), .5)
            rgb = np.zeros((2, 3, 3), np.uint8)
            sam_mask.sam_mask(rgb, 'mouse')
            fake.segment.assert_called_once_with(rgb, 'mouse', .5)
            fake.segment.side_effect = RuntimeError('worker failed')
            with self.assertRaises(RuntimeError):
                sam_mask.sam_mask(rgb, 'mouse')
            old.assert_not_called()

    def test_image_dimensions_mask_selection_and_session_cleanup(self):
        model = backend.SAM31Mask.__new__(backend.SAM31Mask)
        masks = np.zeros((2, 3, 5), bool)
        masks[0, 0, 0] = True
        masks[1, 1:3, 2:5] = True
        requests = []
        def handle(request):
            requests.append(request)
            if request['type'] == 'start_session':
                with Image.open(request['resource_path']) as image:
                    self.assertEqual(image.size, (5, 3))
                return {'session_id': 'test'}
            if request['type'] == 'add_prompt':
                return {'outputs': {'out_binary_masks': masks, 'out_probs': [.6, .9]}}
        model.predictor = Mock(handle_request=handle)
        with patch.object(backend.torch.cuda, 'empty_cache'):
            mask, box, score = model(np.zeros((3, 5, 3), np.uint8), 'mouse')
        np.testing.assert_array_equal(mask, masks[1])
        self.assertEqual(box, [2, 1, 5, 3])
        self.assertEqual(score, .9)
        self.assertEqual(requests[-1]['type'], 'close_session')
        self.assertFalse(Path(requests[0]['resource_path']).exists())
        with patch.object(backend.torch.cuda, 'empty_cache'):
            self.assertEqual(model(np.zeros((3, 5, 3), np.uint8), 'mouse', .95), (None, None, None))
        self.assertEqual(requests[-1]['type'], 'close_session')

    def test_failed_inference_closes_session(self):
        model = backend.SAM31Mask.__new__(backend.SAM31Mask)
        predictor = Mock()
        predictor.handle_request.side_effect = [dict(session_id='test'), RuntimeError('inference failed'), None]
        model.predictor = predictor
        with patch.object(backend.torch.cuda, 'empty_cache'), self.assertRaises(RuntimeError):
            model(np.zeros((3, 5, 3), np.uint8), 'mouse')
        self.assertEqual(predictor.handle_request.call_args.args[0], dict(type='close_session', session_id='test'))


if __name__ == '__main__':
    unittest.main()
