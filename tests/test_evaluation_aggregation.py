import json
import numpy as np
import torch

from scripts.make_smoke_data import main as make_data
import transparent_seg.evaluate as evaluator
from transparent_seg.data import decode_mask, read_manifest
from transparent_seg.metrics import SegmentationMetrics


def test_reused_image_statistics_match_independent_group_updates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    make_data()
    def prediction(model, image, size, device, condition=None):
        return np.indices((image.height, image.width)).sum(axis=0) % 3 == 0
    monkeypatch.setattr(evaluator, 'predict', prediction)
    meta, root = read_manifest('manifests/smoke/validation.json')
    reference = {group: SegmentationMetrics() for group in ('all', 'easy', 'hard')}
    for row in meta['samples']:
        target, valid = decode_mask(root / row['mask'], meta['encoding'])
        pred = np.indices(target.shape).sum(axis=0) % 3 == 0
        reference['all'].update(pred, target, valid)
        reference[row['difficulty']].update(pred, target, valid)
    expected = {group: meter.compute() for group, meter in reference.items()}
    for output in (None, 'evaluation'):
        result = evaluator.evaluate(torch.nn.Identity(), 'manifests/smoke/validation.json', 64, 'cpu', output=output)
        assert result['scores'] == expected
    assert json.load(open('evaluation/metrics.json'))['scores'] == expected
