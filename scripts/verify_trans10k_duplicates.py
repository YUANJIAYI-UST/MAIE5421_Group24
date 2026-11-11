"""Independently compare exact-duplicate RGB pixels and their binary GT masks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.visualize_trans10k_audit import font, make_panel
from transparent_seg.data import decode_mask


def load(path):
    return json.loads(path.read_text())


def paste_fit(panel, tile, x, y, width=292, height=270):
    tile = tile.copy()
    tile.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel.paste(tile, (x + (width - tile.width) // 2, y + (height - tile.height) // 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/Trans10K')
    parser.add_argument('--audit-dir', default='runs/trans10k_data_audit_20261022/diagnostics')
    parser.add_argument('--output', default='runs/trans10k_data_audit_20261022/duplicates_review')
    args = parser.parse_args()
    root, audit_dir, output = Path(args.root), Path(args.audit_dir), Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new output directory to preserve evidence')
    output.mkdir(parents=True)
    audit = load(audit_dir / 'audit.json')
    rows = {r['id']: r for r in load(audit_dir / 'diagnostic_inventory.json')['samples']}
    result = {'method': 'Independent full decoded RGB array equality; strict binary GT comparison',
              'source_data_modified': False, 'near_duplicates_checked': False, 'groups': []}
    for kind in ('cross_split', 'within_split'):
        panels = []
        for number, group in enumerate(audit[f'{kind}_exact_duplicate_groups'], 1):
            ids = group['ids']
            if len(ids) != 2:
                raise ValueError('This paired visualization requires two members per group')
            images, masks, digests = [], [], []
            for sample_id in ids:
                row = rows[sample_id]
                path = root / row['image']
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest == row['image_sha256']
                assert hashlib.sha256((root / row['mask']).read_bytes()).hexdigest() == row['mask_sha256']
                with Image.open(path) as im:
                    images.append(im.convert('RGB'))
                y, valid = decode_mask(root / row['mask'])
                assert valid.all()
                masks.append(y.astype(bool))
                digests.append(digest)
            assert np.array_equal(np.asarray(images[0]), np.asarray(images[1]))
            assert masks[0].shape == masks[1].shape
            changed = int(np.count_nonzero(masks[0] != masks[1]))
            union = int(np.count_nonzero(masks[0] | masks[1]))
            intersection = int(np.count_nonzero(masks[0] & masks[1]))
            record = {'kind': kind, 'ids': ids, 'pixel_sha256': group['pixel_sha256'],
                      'rgb_arrays_equal': True, 'image_files_identical': digests[0] == digests[1],
                      'binary_masks_equal': changed == 0, 'binary_mask_disagreement_pixels': changed,
                      'binary_mask_disagreement_fraction': changed / masks[0].size,
                      'binary_mask_iou': intersection / union if union else 1.0,
                      'source_image_sha256': digests,
                      'source_mask_sha256': [rows[s]['mask_sha256'] for s in ids],
                      'decision': 'preserve_originals_pending_team_protocol',
                      'comparison': f'{kind}_{number:02d}.jpg'}
            panel = Image.new('RGB', (1232, 396), '#f1f5f9')
            draw = ImageDraw.Draw(panel)
            draw.text((12, 8), f'{kind} #{number:02d}: RGB pixels identical', fill='#0f172a', font=font(19))
            for i, sample_id in enumerate(ids):
                draw.text((12 + i * 612, 36), sample_id, fill='#0f172a', font=font(17))
                for j, (label, tile) in enumerate((('RGB', images[i]), ('Binary GT', Image.fromarray(masks[i].astype(np.uint8) * 255).convert('RGB')))):
                    x = 12 + i * 612 + j * 306
                    draw.text((x, 63), label, fill='#334155', font=font(16))
                    paste_fit(panel, tile, x, 87)
            draw.text((12, 370), f'Binary GT differences: {changed:,} pixels ({changed / masks[0].size:.3%}); mask IoU {record["binary_mask_iou"]:.6f}', fill='#0f172a', font=font(17))
            panel.save(output / record['comparison'], quality=94)
            panels.append(panel)
            result['groups'].append(record)
        for start in range(0, len(panels), 4):
            subset = panels[start:start + 4]
            sheet = Image.new('RGB', (1232, 396 * len(subset)), 'white')
            for index, panel in enumerate(subset):
                sheet.paste(panel, (0, 396 * index))
            sheet.save(output / f'{kind}_contact_{start // 4 + 1:02d}.jpg', quality=94)
    result['summary'] = {
        'cross_split_groups': sum(g['kind'] == 'cross_split' for g in result['groups']),
        'within_split_groups': sum(g['kind'] == 'within_split' for g in result['groups']),
        'cross_split_combinations': dict(Counter('-'.join(sorted({s.split('/')[0] for s in g['ids']})) for g in result['groups'] if g['kind'] == 'cross_split')),
        'groups_with_different_binary_gt': sum(not g['binary_masks_equal'] for g in result['groups']),
        'cross_split_groups_with_different_binary_gt': sum(g['kind'] == 'cross_split' and not g['binary_masks_equal'] for g in result['groups']),
        'all_rgb_array_equal': all(g['rgb_arrays_equal'] for g in result['groups'])}
    result['empty_foreground_cases'] = []
    for sample_id in audit['empty_foreground_samples']:
        row = rows[sample_id]
        with Image.open(root / row['image']) as im:
            image = im.convert('RGB')
        target, valid = decode_mask(root / row['mask'])
        assert not target.any() and valid.all()
        panel = make_panel(image, Image.fromarray(target.astype(np.uint8) * 255).convert('RGB'), image, f'{sample_id} | official GT contains no foreground')
        path = f'empty_foreground_{Path(sample_id).stem}.jpg'
        panel.save(output / path, quality=94)
        result['empty_foreground_cases'].append({'id': sample_id, 'comparison': path, 'review': 'pending', 'data_modified': False})
    (output / 'duplicate_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
