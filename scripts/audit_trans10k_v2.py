"""Audit a pinned V2 parquet mirror against V1; test needs a frozen selection.

Optional preparation dependency: pyarrow==20.0.0. Does not execute dataset code.
Unknown labels and alignment outliers are reported, never silently remapped.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from transparent_seg.data_v2 import category_array, CLASS_NAMES, PALETTE
from transparent_seg.near_duplicates import image_features
from transparent_seg.prepare import sha256_file, write_json


def write_identical(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Refusing to overwrite different V2 source: {path}')
    else:
        path.write_bytes(data)


def audit(args):
    rules = json.loads(Path(args.rules).read_text())
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new V2 audit output directory')
    if args.split == ['train', 'validation']:
        stage = 'train_validation_only'
    elif args.split == ['test'] and args.selection:
        selection = json.loads(Path(args.selection).read_text())
        if selection.get('schema') != 'trans10k_holdout_selection_v1' or selection['rules_sha256'] != sha256_file(args.rules):
            raise ValueError('Test audit needs a matching, frozen train/validation selection')
        stage = 'test_after_selection_freeze'
    else:
        raise ValueError('Audit train/validation first; test separately with --selection')
    inventory = json.loads(Path(args.v1_inventory).read_text())
    v1root = Path(args.v1_root)
    v2root = Path(args.v2_root)
    mirrors = Path(args.mirror)
    file_meta = json.loads((mirrors/'files.json').read_text())
    mirror_files = {r['path']: r for r in file_meta if r['type'] == 'file'}
    lookup = {}
    for r in inventory['samples']:
        if r['split'] not in args.split:
            continue
        key = (r['split'], Path(r['image']).name)
        if key in lookup:
            raise ValueError('V1 filename is ambiguous within its official split')
        lookup[key] = r
    settings = rules['alignment']
    orientation = json.loads(Path(args.orientation_policy).read_text())
    if sha256_file(args.orientation_policy) != rules['orientation_policy_sha256']:
        raise ValueError('D01 orientation policy changed')
    if sha256_file(args.v1_inventory) != orientation['source']['inventory_sha256']:
        raise ValueError('V1 inventory differs from the pinned D01 source')
    for split in args.split:
        if sum(r['split'] == split for r in inventory['samples']) != orientation['source']['official_counts'][split]:
            raise ValueError('V1 source split is not complete')
    rotations = {r['id']:r['operation'] for r in orientation['mask_corrections']}
    rows, source_files, modes = [], {}, Counter()

    def process(item):
        split, record = item
        image_name, mask_name = record['image']['path'], record['mask']['path']
        if not image_name or Path(image_name).name != image_name or not mask_name or Path(mask_name).name != mask_name:
            raise ValueError('Missing/unsafe original V2 filename')
        if mask_name != Path(image_name).stem + '_mask.png':
            raise ValueError('V2 image/mask ID mismatch')
        source = lookup.get((split, image_name))
        if source is None:
            raise ValueError(f'V2 ID does not exist in same V1 split: {split}/{image_name}')
        original = v1root/source['image']
        if sha256_file(original) != source['image_sha256']:
            raise ValueError('V1 image changed since D01 inventory')
        image_path, mask_path = f'{split}/images/{image_name}', f'{split}/masks_12/{mask_name}'
        write_identical(v2root/image_path, record['image']['bytes'])
        write_identical(v2root/mask_path, record['mask']['bytes'])
        with Image.open(BytesIO(record['image']['bytes'])) as v2, Image.open(original) as v1:
            v2.load(); v1.load()
            size = list(v2.size)
            operation = rotations.get(source['id'],'none')
            if operation == 'rotate_90_cw':
                aligned = v2.transpose(Image.Transpose.ROTATE_270)
            elif operation == 'rotate_90_ccw':
                aligned = v2.transpose(Image.Transpose.ROTATE_90)
            elif operation == 'none':
                aligned = v2
            else:
                raise ValueError('Unsupported documented orientation operation')
            aspect = abs((aligned.width/aligned.height)/(v1.width/v1.height) - 1)
            n = settings['thumbnail_size']
            a = np.asarray(v1.convert('RGB').resize((n,n), Image.Resampling.BILINEAR), np.float64)
            b = np.asarray(aligned.convert('RGB').resize((n,n), Image.Resampling.BILINEAR), np.float64)
            rmse = float(np.sqrt(np.mean((a-b)**2)))
            # pHash is a second check alongside filename, split, aspect and pixels.
            h1 = image_features(v1)[0][0]
            h2 = image_features(aligned)[0][0]
            hamming = (h1 ^ h2).bit_count()
        with Image.open(BytesIO(record['mask']['bytes'])) as mask:
            mask.load()
            ids, unknown = category_array(mask, rules['v2_encoding'])
            mask_size, mode = list(mask.size), mask.mode
            colors = {f'#{r:02x}{g:02x}{b:02x}': int((ids == i).sum()) for i,(r,g,b) in enumerate(PALETTE)}
        counts = np.bincount(ids[ids >= 0], minlength=len(CLASS_NAMES)).tolist()
        issues = []
        if size != mask_size:
            issues.append('v2_image_mask_geometry_mismatch')
        if aspect > settings['maximum_relative_aspect_error'] or rmse > settings['maximum_thumbnail_rmse'] or hamming > settings['maximum_phash_hamming']:
            issues.append('v1_v2_visual_alignment_outlier')
        if unknown:
            issues.append('unknown_v2_category_code')
        return {'id': source['id'], 'split': split, 'v2_image': image_path, 'v2_mask': mask_path,
                'v1_image_sha256': source['image_sha256'], 'v2_image_sha256': hashlib.sha256(record['image']['bytes']).hexdigest(),
                'v2_mask_sha256': hashlib.sha256(record['mask']['bytes']).hexdigest(),
                'v1_image_size': source['image_size'], 'v2_image_size': size, 'v2_mask_size': mask_size,
                'v2_mask_mode': mode, 'category_pixel_counts': counts, 'category_ids': [i for i,n in enumerate(counts) if n and i],
                'v2_to_v1_orientation':operation,
                'known_colors_pixel_counts': {k:v for k,v in colors.items() if v}, 'unknown_codes': unknown,
                'alignment': {'relative_aspect_error': aspect, 'thumbnail_rmse': rmse, 'phash_hamming': hamming,
                              'same_encoded_image': source['image_sha256'] == hashlib.sha256(record['image']['bytes']).hexdigest()},
                'issues': issues, 'category_filter_eligible': not issues}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for split in args.split:
            paths = sorted((mirrors/'data').glob(f'{split}-*.parquet'))
            expected_paths = sorted(k for k in mirror_files if k.startswith(f'data/{split}-'))
            if [p.relative_to(mirrors).as_posix() for p in paths] != expected_paths:
                raise ValueError(f'Missing/extra parquet shards for {split}')
            for path in paths:
                rel = path.relative_to(mirrors).as_posix()
                digest = sha256_file(path)
                if digest != mirror_files[rel]['lfs']['oid']:
                    raise ValueError('Parquet SHA256 differs from pinned source')
                source_files[rel] = digest
                for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                    rows.extend(pool.map(process, [(split, r) for r in batch.to_pylist()]))
                    if len(rows) % 256 == 0:
                        print(f'Audited and aligned V2 images: {len(rows)}', flush=True)
    rows.sort(key=lambda r:r['id'])
    if len({r['id'] for r in rows}) != len(rows) or {r['id'] for r in rows} != {r['id'] for r in lookup.values()}:
        raise ValueError('V2 membership is not a bijection with requested V1 split IDs')
    modes.update(r['v2_mask_mode'] for r in rows)
    counts = {}
    for split in args.split:
        subset = [r for r in rows if r['split'] == split]
        counts[split] = {'images': len(subset), 'eligible': sum(r['category_filter_eligible'] for r in subset),
                         'quarantined': sum(not r['category_filter_eligible'] for r in subset)}
    result = {'schema':'trans10k_v2_alignment_inventory_v1', 'stage':stage,
              'rules_sha256':sha256_file(args.rules), 'v1_inventory_sha256':sha256_file(args.v1_inventory),
              'source_type':rules['source_type'], 'mirror_revision':rules['mirror_revision'],
              'parquet_sha256': source_files, 'v2_root_name':v2root.name,
              'encoding':rules['v2_encoding'], 'class_names':list(CLASS_NAMES), 'palette':list(PALETTE),
              'split_counts':counts, 'mask_modes':dict(modes), 'samples':rows}
    if stage == 'test_after_selection_freeze':
        result['selection_sha256_before_test_audit'] = sha256_file(args.selection)
    output.mkdir(parents=True)
    write_json(output/'inventory.json', result)
    write_json(output/'issues.json', [r for r in rows if r['issues']])
    print(json.dumps({'stage':stage, 'counts':counts, 'issues':dict(Counter(i for r in rows for i in r['issues'])),
                      'inventory_sha256':sha256_file(output/'inventory.json')}, indent=2), flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mirror', default='data/archives/Trans10K_v2_mirror')
    p.add_argument('--v2-root', default='data/Trans10K_v2_mirror_3445e93')
    p.add_argument('--v1-root', default='data/Trans10K')
    p.add_argument('--v1-inventory', default='runs/trans10k_data_audit_20261022/diagnostics/diagnostic_inventory.json')
    p.add_argument('--rules', default='configs/data/trans10k_d03_rules_20261028.json')
    p.add_argument('--orientation-policy', default='configs/data/trans10k_v1_review_20261028.json')
    p.add_argument('--split', nargs='+', choices=['train','validation','test'], required=True)
    p.add_argument('--selection')
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    args=p.parse_args()
    if not 1 <= args.workers <= 8:
        p.error('Use 1-8 workers')
    audit(args)


if __name__ == '__main__':
    main()
