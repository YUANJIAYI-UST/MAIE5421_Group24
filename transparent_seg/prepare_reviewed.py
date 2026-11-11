"""Build a traceable V1 derivative without changing any official source file."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .data import decode_mask
from .prepare import sha256_file, write_json


def inside(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe relative path: {relative}')
    resolved = (root / p).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes data root: {relative}')
    return resolved


def cross_overlap(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['pixel_sha256']].append(row)
    return [[r['id'] for r in group] for group in groups.values()
            if len({r['split'] for r in group}) > 1]


def prepare_reviewed(root, output, policy_path, inventory_path, audit_path, workers=4):
    root, output = Path(root).resolve(), Path(output).resolve()
    policy_path, inventory_path, audit_path = map(Path, (policy_path, inventory_path, audit_path))
    policy = json.loads(policy_path.read_text())
    inventory = json.loads(inventory_path.read_text())
    source_audit = json.loads(audit_path.read_text())
    if policy.get('schema') != 'trans10k_local_derivative_policy_v1':
        raise ValueError('Unsupported derivative policy')
    if inventory.get('schema') != 'trans10k_diagnostic_inventory_v1':
        raise ValueError('Requires a full diagnostic inventory')
    if not 1 <= workers <= 8:
        raise ValueError('Use 1-8 workers')
    for path, key in ((inventory_path, 'inventory_sha256'), (audit_path, 'audit_sha256')):
        if sha256_file(path) != policy['source'][key]:
            raise ValueError(f'Source evidence hash mismatch: {path}')
    if output.exists():
        raise FileExistsError('Use a new manifest output directory')
    common_root = root.parent
    masks_dir = inside(common_root, policy['corrected_masks_directory'])
    if masks_dir == root or masks_dir.is_relative_to(root) or root.is_relative_to(masks_dir):
        raise ValueError('Corrected masks must be outside the original source tree')
    rows = sorted(inventory['samples'], key=lambda r: r['id'])
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate sample IDs')
    expected = policy['source']['official_counts']
    if dict(Counter(r['split'] for r in rows)) != expected:
        raise ValueError('Source counts differ from policy')
    if source_audit.get('unpaired_masks'):
        raise ValueError('Unresolved unpaired masks')
    corrections = {c['id']: c for c in policy['mask_corrections']}
    excluded = {c['id']: c for c in policy['exclusions']}
    if len(corrections) != len(policy['mask_corrections']) or len(excluded) != len(policy['exclusions']):
        raise ValueError('Duplicate policy entries')
    if (set(corrections) | set(excluded)) - by_id.keys():
        raise ValueError('Policy contains unknown IDs')
    bad_shapes = {r['id'] for r in rows if r['image_size'] != r['mask_size']}
    if set(corrections) != bad_shapes:
        raise ValueError('Every and only geometry mismatch needs a correction')
    for row in rows:
        if any(i['type'] != 'shape_mismatch' for i in row.get('issues', [])):
            raise ValueError(f'Unresolved source issue: {row["id"]}')
        if row['split'] != row['id'].split('/')[0] or row['id'] != row['image']:
            raise ValueError('Inconsistent source split or image ID')
    for sid, correction in corrections.items():
        if by_id[sid]['split'] != 'train' or correction['operation'] not in ('rotate_90_cw', 'rotate_90_ccw'):
            raise ValueError('Only documented training-mask rotations are supported')
        if correction['source_mask_sha256'] != by_id[sid]['mask_sha256']:
            raise ValueError('Correction refers to a different source mask')
    priority = {'train': 0, 'validation': 1, 'test': 2}
    for sid, exclusion in excluded.items():
        row, kept = by_id[sid], by_id.get(exclusion['retained_id'])
        if row['split'] == 'test':
            raise ValueError('Official test membership must be preserved')
        if not kept or kept['id'] in excluded or priority[kept['split']] <= priority[row['split']]:
            raise ValueError('Invalid retained counterpart')
        if exclusion['reason_code'] != 'cross_split_exact_rgb_duplicate' or row['pixel_sha256'] != kept['pixel_sha256']:
            raise ValueError('Exact duplicate exclusion lacks matching RGB evidence')
    active = [r for r in rows if r['id'] not in excluded]
    if cross_overlap(active):
        raise ValueError('Cross-split exact leakage remains')
    for split in expected:
        actual_images = {p.relative_to(root).as_posix() for p in (root / split).rglob('*')
                         if p.parent.name == 'images' and p.suffix.lower() in ('.jpg', '.jpeg', '.png')}
        actual_masks = {p.relative_to(root).as_posix() for p in (root / split).rglob('*.png')
                        if p.parent.name == 'masks'}
        if actual_images != {r['image'] for r in rows if r['split'] == split} or actual_masks != {r['mask'] for r in rows if r['split'] == split}:
            raise ValueError('Source membership differs from audited inventory')

    def verify(row):
        for key in ('image', 'mask'):
            path = inside(root, row[key])
            if sha256_file(path) != row[key + '_sha256']:
                raise ValueError(f'Source file changed: {row[key]}')
            with Image.open(path) as im:
                if list(im.size) != row[key + '_size']:
                    raise ValueError(f'Source dimensions changed: {row[key]}')
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(verify, rows), 1):
            if n % 1000 == 0:
                print(f'Verified source hashes and headers: {n}/{len(rows)}', flush=True)

    repaired = {}
    repair_log = []
    for sid, correction in sorted(corrections.items()):
        row = by_id[sid]
        destination = masks_dir / row['mask']
        rotation = Image.Transpose.ROTATE_270 if correction['operation'] == 'rotate_90_cw' else Image.Transpose.ROTATE_90
        k = -1 if correction['operation'] == 'rotate_90_cw' else 1
        with Image.open(inside(root, row['mask'])) as im:
            source_pixels = np.array(im)
            rotated = im.transpose(rotation)
            if rotated.size != tuple(row['image_size']):
                raise ValueError('Rotation does not resolve geometry')
            if not np.array_equal(np.asarray(rotated), np.rot90(source_pixels, k)):
                raise ValueError('Rotation changed raw labels')
            if destination.exists():
                with Image.open(destination) as saved:
                    if saved.mode != rotated.mode or not np.array_equal(np.asarray(saved), np.asarray(rotated)):
                        raise ValueError('Existing corrected mask differs; refusing overwrite')
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                rotated.save(destination, format='PNG')
        y, valid = decode_mask(destination)
        if int(y.sum()) != row['foreground_pixels'] or int(valid.sum()) != row['valid_pixels']:
            raise ValueError('Rotation changed decoded foreground/validity counts')
        repaired[sid] = destination
        repair_log.append({**correction, 'output_mask': destination.relative_to(common_root).as_posix(),
                           'output_mask_sha256': sha256_file(destination), 'pixels_preserved': True,
                           'applied_to_copy': True, 'human_visual_review': 'pending'})

    protocol = {'id': policy['id'], 'policy_sha256': sha256_file(policy_path),
                'status': 'local_pilot_pending_human_review', 'source_dataset': 'trans10k_v1',
                'human_visual_review': 'pending', 'member5_review': 'pending',
                'perceptual_candidates_review': 'pending', 'official_test_membership_preserved': True}
    output.mkdir(parents=True)
    manifest_hashes = {}
    split_summary = {}
    for split in expected:
        samples, histogram = [], Counter()
        for row in active:
            if row['split'] != split:
                continue
            mask = repaired.get(row['id'], root / row['mask'])
            samples.append({'id': row['id'], 'image': (root / row['image']).relative_to(common_root).as_posix(),
                            'mask': mask.relative_to(common_root).as_posix(), 'difficulty': row['difficulty'],
                            'height': row['image_size'][1], 'width': row['image_size'][0],
                            'image_sha256': row['image_sha256'],
                            'mask_sha256': sha256_file(mask) if row['id'] in repaired else row['mask_sha256'],
                            'pixel_sha256': row['pixel_sha256'], 'foreground_pixels': row['foreground_pixels'],
                            'valid_pixels': row['valid_pixels'],
                            'source_mask': (root / row['mask']).relative_to(common_root).as_posix(),
                            'source_mask_sha256': row['mask_sha256'],
                            'mask_transform': corrections[row['id']]['operation'] if row['id'] in corrections else 'none'})
            histogram.update(row['mask_channel_sum_histogram'])
        manifest = {'schema_version': 1, 'dataset': policy['id'], 'root': Path(os.path.relpath(common_root, output)).as_posix(),
                    'root_relative_to_manifest': True, 'split': split, 'encoding': 'trans10k_v1_rgb',
                    'void_values': [], 'official_count': expected[split], 'complete_official_split': split == 'test',
                    'synthetic': False, 'data_protocol': protocol, 'effective_count': len(samples),
                    'mask_channel_sum_histogram': dict(sorted(histogram.items())), 'samples': samples}
        path = output / f'{split}.json'
        write_json(path, manifest)
        manifest_hashes[split] = sha256_file(path)
        split_summary[split] = {'source_count': expected[split], 'effective_count': len(samples),
                                'excluded_count': expected[split] - len(samples),
                                'difficulty': dict(Counter(r['difficulty'] for r in samples))}
    audit = {'schema': 'trans10k_derivative_audit_v1', 'status': 'technical_checks_passed_review_pending',
             'data_protocol': protocol, 'source_evidence': policy['source'], 'splits': split_summary,
             'original_files_modified': False, 'source_hashes_verified': len(rows) * 2,
             'geometry_mismatches_after_correction': 0, 'cross_split_exact_overlap': [],
             'corrections': repair_log, 'exclusions': policy['exclusions'],
             'within_split_duplicates': source_audit['within_split_exact_duplicate_groups'],
             'empty_foreground_samples': source_audit['empty_foreground_samples'],
             'perceptual_review_status': 'pending; candidates do not automatically exclude samples',
             'human_review': {'owner': 'pending', 'member5': 'pending'},
             'manifest_sha256': manifest_hashes}
    write_json(output / 'audit.json', audit)
    write_json(output / 'policy.json', policy)
    write_json(output / 'release.json', {'protocol_id': policy['id'], 'manifest_sha256': manifest_hashes,
                                       'policy_sha256': sha256_file(policy_path), 'audit_sha256': sha256_file(output / 'audit.json')})
    print(json.dumps({'counts': split_summary, 'manifest_sha256': manifest_hashes}, indent=2), flush=True)
    return audit
