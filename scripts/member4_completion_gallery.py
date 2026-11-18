"""Render declared examples and failures; verify predictions against full runs."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transparent_seg.data import read_manifest, decode_mask, image_tensor, MEAN, STD
from transparent_seg.evaluate import load_checkpoint, predict
from transparent_seg.evaluate_resumable import CONDITIONS
from transparent_seg.prepare import sha256_file
from scripts.summarize_member4_completion import RUNS, GSDS, SHAS, TEST, GSD, REPORT, read, save, require

CHECKPOINTS = {
    'A': Path('runs/member2_full_A_20261114/best.pt'),
    'C': Path('runs/member4_full_C_seed42_20261115/best.pt'),
    **{m: Path(f'runs/member4_BD_delivery_20261118/full_runs/B03_{m}/best.pt') for m in ['B', 'D']},
}
FONT = 'C:/Windows/Fonts/arial.ttf'


def thumbnail(image, width, height):
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    return copy


def main():
    require(read(REPORT/'verification.json')['passed'], 'Complete independent audit first')
    selection = read(REPORT/'gallery_selection.json')
    out = REPORT/'gallery'
    out.mkdir(exist_ok=True)
    require(not (out/'index.json').exists(), 'Gallery already completed')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.is_available(), 'Use the same CUDA inference backend')
    models = {}
    for name, checkpoint in CHECKPOINTS.items():
        require(sha256_file(checkpoint) == SHAS[name], 'Checkpoint changed')
        models[name] = load_checkpoint(checkpoint, 'cuda')[0]
    font = ImageFont.truetype(FONT, 19)
    index = []
    for dataset, manifest, names, ids, conditions, runs in [
        ('GSD', GSD, ['A', 'D'], selection['gsd_ids'], ['clean'], GSDS),
        ('Trans10K', TEST, ['A', 'B', 'C', 'D'], selection['test_ids'],
         ['clean', 'brightness_0.6', 'contrast_0.5', 'gamma_1.6'], RUNS),
    ]:
        meta, base = read_manifest(manifest)
        samples = {s['id']: s for s in meta['samples']}
        scores = {}
        for name in names:
            for condition in conditions:
                with (runs[name]/condition/'per_image.csv').open(encoding='utf-8-sig', newline='') as f:
                    scores[name, condition] = {r['id']: r for r in csv.DictReader(f)}
        folder = out/dataset
        folder.mkdir(exist_ok=True)
        fixed = selection['rules_fixed_before_formal_scoring'][
            'GSD_fixed_ids' if dataset == 'GSD' else 'test_fixed_ids']
        for number, identity in enumerate(ids, 1):
            sample = samples[identity]
            with Image.open(base/sample['image']) as im:
                original = im.convert('RGB')
            target, valid = decode_mask(base/sample['mask'], meta['encoding'], meta.get('void_values', []))
            stem = f'{number:02d}'
            # Unaltered source bytes are kept in the data directory; these PNGs
            # are explicitly display exports, not inputs to formal evaluation.
            thumbnail(original, 1280, 960).save(folder/f'{stem}_image.png')
            Image.fromarray(target.astype(np.uint8)*255).save(folder/f'{stem}_GT.png')
            reasons = ['fixed before scoring'] if identity in fixed else []
            reasons += [f['reason'] for f in selection['failures']
                        if f['dataset'] == dataset and f['id'] == identity]
            columns = 2 + len(names)
            canvas = Image.new('RGB', (columns*320, 65+350*len(conditions)), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), f'{dataset} {number:02d} | {identity}', font=font, fill='black')
            draw.text((8, 32), 'Green: predicted foreground. White in GT: annotated foreground.', font=font, fill='black')
            records = {}
            thumb = thumbnail(original, 304, 292)
            for row, condition in enumerate(conditions):
                tensor = image_tensor(original, 512, CONDITIONS[condition]).numpy().transpose(1, 2, 0)
                pixels = np.rint(np.clip(tensor*STD+MEAN, 0, 1)*255).astype(np.uint8)
                input_view = Image.fromarray(pixels).resize(thumb.size, Image.Resampling.LANCZOS)
                label = Image.fromarray(target.astype(np.uint8)*255).resize(thumb.size, Image.Resampling.NEAREST).convert('RGB')
                views, captions = [input_view, label], [condition, 'Ground truth']
                records[condition] = {}
                for name in names:
                    pred = predict(models[name], original, 512, 'cuda', CONDITIONS[condition])
                    checksum = hashlib.sha256(np.packbits(pred).tobytes()).hexdigest()
                    saved = scores[name, condition][identity]
                    require(checksum == saved['prediction_sha256'],
                            f'Gallery differs from formal prediction: {identity}/{name}/{condition}')
                    mask = Image.fromarray(pred.astype(np.uint8)*255)
                    mask.save(folder/f'{stem}_{name}_{condition}_mask.png')
                    rgb = np.array(input_view, dtype=np.float32)
                    foreground = np.array(mask.resize(thumb.size, Image.Resampling.NEAREST)) > 0
                    rgb[foreground] = .5*rgb[foreground]+.5*np.array([0, 220, 160])
                    views.append(Image.fromarray(rgb.astype(np.uint8)))
                    captions.append(f'{name}: IoU {float(saved["foreground_iou"])*100:.2f}%')
                    records[condition][name] = {'iou': float(saved['foreground_iou']),
                        'prediction_sha256': checksum,
                        'mask': f'{dataset}/{stem}_{name}_{condition}_mask.png'}
                for col, (view, caption) in enumerate(zip(views, captions)):
                    top = 65+row*350
                    draw.text((col*320+8, top), caption, font=font, fill='black')
                    canvas.paste(view, (col*320+(320-view.width)//2, top+34+(292-view.height)//2))
            canvas.save(folder/f'{stem}.png')
            index.append({'dataset': dataset, 'id': identity, 'file': f'{dataset}/{stem}.png',
                'image': f'{dataset}/{stem}_image.png', 'ground_truth': f'{dataset}/{stem}_GT.png',
                'original_image': str(base/sample['image']),
                'image_export_is_display_thumbnail': True,
                'selection_reasons': reasons, 'predictions': records,
                'all_prediction_hashes_match_formal_run': True})
            print(f'{dataset}: {number}/{len(ids)} {identity}', flush=True)
    save(out/'index.json', index)


if __name__ == '__main__':
    main()
