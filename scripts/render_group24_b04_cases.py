"""Render fixed, manually inspected clean-test cases with all four models."""
import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
import torch

from transparent_seg.data import decode_mask, read_manifest
from transparent_seg.evaluate import load_checkpoint, predict
from transparent_seg.metrics import SegmentationMetrics
from transparent_seg.prepare import sha256_file

CASES = {
    'test/hard/Images/178.jpg': 'Thin spectacle lenses',
    'test/hard/Images/5835.jpg': 'Window occluded by office furniture',
    'test/easy/Images/6459.jpg': 'Large storefront glazing',
}
KNOWN_BASELINES = {'A': '3a785feb578ecbd2199684c53130737c872cd6c2f5803f221297f1e9dac0b25a',
                   'C': '69841f34ec0c76b0a2dbdce3a07f8ea5926a6beec847c4e8b03e587efc3b5f0d'}


def tinted(image, regions):
    rgb = np.asarray(image).astype(np.float32)
    for mask, color in regions:
        rgb[mask] = .6 * rgb[mask] + .4 * np.array(color)
    return Image.fromarray(rgb.astype(np.uint8))


def render(checkpoints, output):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    models = {}
    digests = {}
    for name, path in checkpoints.items():
        digests[name] = sha256_file(path)
        if name in KNOWN_BASELINES and digests[name] != KNOWN_BASELINES[name]:
            raise ValueError(f'Unexpected {name} checkpoint')
        model, ck = load_checkpoint(path, 'cuda')
        if ck['config']['ablation'] != name:
            raise ValueError(f'{name} is the wrong ablation')
        models[name] = model, ck['config']['size']
    meta, root = read_manifest('manifests/member1_final_20261029/v1/test.json')
    samples = {s['id']: s for s in meta['samples']}
    width, height = 318, 290
    panel = Image.new('RGB', (6 * width, 110 + len(CASES) * height), '#f4f6f8')
    draw = ImageDraw.Draw(panel)
    draw.text((12, 9), 'B04 | Clean-test case comparison', fill='#182231', font_size=28)
    draw.text((12, 49), 'Green: correct foreground    Red: false positive    Amber: missed foreground',
              fill='#182231', font_size=22)
    for column, label in enumerate(('Input', 'Ground truth', 'A', 'B', 'C', 'D')):
        draw.text((column * width + 12, 81), label, fill='#182231', font_size=20)
    rows = []
    for index, (sid, title) in enumerate(CASES.items()):
        sample = samples[sid]
        with Image.open(root / sample['image']) as im:
            image = im.convert('RGB')
        target, valid = decode_mask(root / sample['mask'], meta['encoding'], meta['void_values'])
        y = target.astype(bool) & valid
        images = [(image.copy(), title), (Image.fromarray(y.astype(np.uint8) * 255).convert('RGB'), sid.rsplit('/', 1)[-1])]
        for name, (model, size) in models.items():
            p = predict(model, image, size, 'cuda') & valid
            meter = SegmentationMetrics()
            meter.update(p, y, valid)
            score = meter.compute()
            rows.append({'case': title, 'sample_id': sid, 'model': name, 'checkpoint_sha256': digests[name], **score})
            overlay = tinted(image, [(p & y, (23, 219, 153)), (p & ~y, (255, 46, 76)), (~p & y, (255, 179, 28))])
            images.append((overlay, f"IoU {score['foreground_iou']:.3f}   Boundary {score['boundary_iou']:.3f}"))
        for column, (im, label) in enumerate(images):
            im.thumbnail((width - 16, height - 47))
            left = column * width + (width - im.width) // 2
            top = 110 + index * height
            panel.paste(im, (left, top))
            draw.text((column * width + 8, top + height - 35), label, fill='#182231', font_size=16)
    panel.save(output / 'B04_failure_cases.png')
    with (output / 'B04_failure_cases.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in 'ABCD':
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render({name: getattr(args, name) for name in 'ABCD'}, args.output)
