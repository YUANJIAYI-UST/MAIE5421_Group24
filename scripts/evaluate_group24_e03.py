"""Evaluate an isolated A/D checkpoint once on the frozen D03 test partition."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate

from transparent_seg.data import decode_mask, image_tensor, read_manifest
from transparent_seg.data_v2 import category_array
from transparent_seg.evaluate import load_checkpoint, predict_in_v2_frame
from transparent_seg.evaluate_resumable import (PreparedTarget, atomic_json, code_identity, digest,
                                                erosion_backend, exclusive_run, merge_score)
from transparent_seg.metrics import CategoryRecall, SegmentationMetrics
from transparent_seg.prepare import sha256_file

FOLDER = Path('manifests/member1_final_20261029/d03')
FROZEN = {
    'train': '73bda9250625d74eebcc6c2bb21b35652e5c84c1435591dc0776a62422549ce6',
    'validation': '77841abe665891f53a3c11e652abcdd85d6fc6986a618dc14969c4c35ad8da6f',
    'test': '08d3e715f87586378d146bc4644f10a9bd4eb5b035d45d650b5e4d150c69a839',
    'test_heldout': 'df1187245aa50f2bd52accfdc71d2a25d173f87a2c7769c0a67a2a00390c3d49',
    'test_seen': '7ff632c7f86b427e871f4fea94b0bec755affc743072bd3fcb44434879492c35',
}
INIT = 'fd50559254d2d5fe9b16cbaa6f7fd32452faa4a7f3b9a3ec42f1a6fb6eedc835'


def verified_bytes(root, sample, key):
    path = (root / sample[key]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe manifest path for {sample["id"]}')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sample[key + '_sha256']:
        raise ValueError(f'Unexpected {key} contents for {sample["id"]}')
    return io.BytesIO(data)


def memberships():
    for split, expected in FROZEN.items():
        if sha256_file(FOLDER / f'{split}.json') != expected:
            raise ValueError(f'Frozen D03 {split} manifest differs')
    meta, root = read_manifest(FOLDER / 'test.json')
    groups = {}
    for group, expected in (('seen', 2935), ('heldout', 1493)):
        m, _ = read_manifest(FOLDER / f'test_{group}.json')
        groups[group] = {item['id'] for item in m['samples']}
        if len(groups[group]) != len(m['samples']) or len(groups[group]) != expected:
            raise ValueError(f'Invalid D03 test_{group} count')
    ids = {item['id'] for item in meta['samples']}
    if len(ids) != len(meta['samples']) or len(ids) != 4428 or (
            groups['seen'] & groups['heldout'] or groups['seen'] | groups['heldout'] != ids):
        raise ValueError('D03 test subsets must partition the complete test set')
    return meta, root, groups


@torch.inference_mode()
def evaluate(checkpoint, output, backend):
    meta, root, groups = memberships()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model, ckpt = load_checkpoint(checkpoint, 'cuda')
    config = ckpt['config']
    ablation = config['ablation']
    if (ablation not in {'A', 'D'} or config.get('data_protocol_id') != 'member1_final_20261029_d03'
            or ckpt['manifest_sha256'] != {'train': FROZEN['train'], 'val': FROZEN['validation']}
            or ckpt['initial_shared_state_sha256'] != INIT or ckpt['epoch'] not in range(1, 31)):
        raise ValueError('Expected a fresh isolated A/D checkpoint')
    erode = erosion_backend(backend)
    identity = {'checkpoint_sha256': sha256_file(checkpoint), 'manifests': FROZEN,
                'ablation': ablation, 'epoch': ckpt['epoch'], 'threshold': 0.5, 'precision': 'float32',
                'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__),
                'boundary_backend': backend, 'source': code_identity(), 'script': sha256_file(Path(__file__))}
    if erode:
        import cv2
        identity['opencv'] = cv2.__version__
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        run_chunks(model, config['size'], meta, root, groups, output, identity, erode)


def run_chunks(model, size, meta, root, groups, output, identity, erode):
    receipt = output / 'identity.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != identity:
        raise ValueError('Resume identity differs')
    atomic_json(receipt, identity)
    scores = {g: SegmentationMetrics() for g in ('all', *groups)}
    categories = {g: CategoryRecall() for g in scores}
    rows = []
    size_chunk = 25
    samples = meta['samples']
    for start in range(0, len(samples), size_chunk):
        members = samples[start:start + size_chunk]
        path = output / 'chunks' / f'{start:06d}.json'
        if path.exists():
            block = json.loads(path.read_text(encoding='utf-8'))
            body = block['body']
            if (block['sha256'] != digest(body) or body['identity'] != digest(identity)
                    or [r['id'] for r in body['rows']] != [s['id'] for s in members]):
                raise ValueError(f'Invalid saved evaluation chunk {path}')
        else:
            body = {'identity': digest(identity), 'rows': [evaluate_image(
                model, size, meta, root, sample, erode) for sample in members]}
            atomic_json(path, {'body': body, 'sha256': digest(body)})
        for row in body['rows']:
            subset = 'heldout' if row['id'] in groups['heldout'] else 'seen'
            row['subset'] = subset
            for group in ('all', subset):
                merge_score(scores[group], row['metrics'])
                categories[group].hits += row['hits']
                categories[group].total += row['total']
            rows.append(row)
        if len(rows) % 100 == 0 or len(rows) == len(samples):
            print(f'E03 {identity["ablation"]}: {len(rows)}/{len(samples)}', flush=True)
    results = {g: scores[g].compute() for g in scores}
    counts = {'all': 4428, 'seen': 2935, 'heldout': 1493}
    for group, count in counts.items():
        if results[group]['images'] != count or results[group]['boundary_images'] != count:
            raise ValueError(f'Incomplete original-resolution evaluation: {group}')
    if any(categories['seen'].total[i] for i in (5, 8)) or any(
            categories['heldout'].total[i] == 0 for i in (5, 8)):
        raise ValueError('V2 class membership differs from the isolated protocol')
    with (output / 'per_image.csv.tmp').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'subset', *SegmentationMetrics().compute(), 'hits', 'total'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ('id', 'subset', 'hits', 'total')} | row['metrics'])
    (output / 'per_image.csv.tmp').replace(output / 'per_image.csv')
    atomic_json(output / 'result.json', {'complete': True, 'identity': identity, 'scores': results,
                                        'category_recall': {g: m.compute() for g, m in categories.items()}})


def evaluate_image(model, size, meta, root, sample, erode):
    with Image.open(verified_bytes(root, sample, 'image')) as im:
        image = im.convert('RGB')
    target, valid = decode_mask(verified_bytes(root, sample, 'mask'), meta['encoding'], meta['void_values'])
    if target.shape != (image.height, image.width) or not sample['v2_category_eligible']:
        raise ValueError(f'Mask geometry or eligibility mismatch for {sample["id"]}')
    with Image.open(verified_bytes(root, sample, 'v2_category_mask')) as im:
        labels, unknown = category_array(im, meta['category_encoding'])
    if unknown:
        raise ValueError(f'Unexpected category code for {sample["id"]}')
    logits = model(image_tensor(image, size)[None].cuda())['logits']
    pred = interpolate(logits.float(), target.shape, mode='bilinear', align_corners=False)
    pred = (pred[0, 0].sigmoid() >= 0.5).cpu().numpy()
    orientation = sample['v2_to_v1_orientation']
    v2_pred = pred if orientation == 'none' and labels.shape == target.shape else predict_in_v2_frame(
        logits, labels.shape, orientation)
    category = CategoryRecall()
    category.update(v2_pred, labels)
    if category.total.tolist() != sample['v2_category_pixel_counts']:
        raise ValueError(f'V2 pixel counts differ for {sample["id"]}')
    return {'id': sample['id'], 'metrics': PreparedTarget(target, valid, erode).score(pred),
            'hits': category.hits.tolist(), 'total': category.total.tolist()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--boundary-backend', choices=('scipy', 'opencv'), default='scipy')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output, args.boundary_backend)
