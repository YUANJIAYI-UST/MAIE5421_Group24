"""Import pinned GSD bytes and retrieve cross-dataset candidates; never score a model."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from transparent_seg.data import decode_mask, read_manifest
from transparent_seg.near_duplicates import METHOD, image_features
from transparent_seg.prepare import sha256_file, write_json

REVISION = 'f8897574bbc2632a440efa1de021fd2b02f89f51'
PARQUET_SHA = 'c1ea301305f04d9d05ef09dbf2982e4ce188fb8d779c85095017c54c3a688963'
TRANS_PINS = {
    'train': '7a3094e0c1aeacf0edd7335f196af79eacd4b551cd9f78cce2cf7c903e910ca2',
    'validation': '861c24027bfddcfba031e253a6518343010e31d296fb66c010f8d88ee9a90497',
    'test': 'ecf3ad309423aa0247183cfc1e315921a7666c47d10700f381199a7dcfb41012'}
EXPECTED = {'train': 4959, 'validation': 995, 'test': 4428}
ROOT = Path('runs/member4_stage5_20261116')
DATA = Path('data/GSD_member4_f8897574_test')
MANIFESTS = Path('manifests/member4_gsd_20261116')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rgb_hash(rgb):
    return hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()


def pair_id(gsd_id, trans_id):
    return hashlib.sha256((gsd_id + '\n' + trans_id).encode()).hexdigest()[:20]


def cross_candidates(g_features, t_features, g_rows, t_rows, block_size=64):
    """Exhaustive rectangular search using exactly the existing three view tests."""
    g = np.asarray(g_features, dtype=np.uint64)
    t = np.asarray(t_features, dtype=np.uint64)
    found = {}
    for lv, rv, threshold, method in (
            (0, 0, METHOD['global_phash_distance'], 'whole_vs_whole'),
            (0, 1, METHOD['center_to_global_phash_distance'], 'whole_vs_center80'),
            (1, 0, METHOD['center_to_global_phash_distance'], 'center80_vs_whole')):
        for start in range(0, len(g_rows), block_size):
            stop = min(start + block_size, len(g_rows))
            pd = np.bitwise_count(g[start:stop, lv, 0, None] ^ t[None, :, rv, 0])
            dd = np.bitwise_count(g[start:stop, lv, 1, None] ^ t[None, :, rv, 1])
            ii, jj = np.nonzero((pd <= threshold) & (dd <= METHOD['maximum_dhash_distance']))
            for local, j in zip(ii.tolist(), jj.tolist()):
                i = start + local
                if g_rows[i]['pixel_sha256'] == t_rows[j]['pixel_sha256']:
                    continue
                found.setdefault((i, j), []).append({
                    'method': method, 'phash_distance': int(pd[local, j]),
                    'dhash_distance': int(dd[local, j])})
    result = []
    for (i, j), matches in found.items():
        a, b = g_rows[i], t_rows[j]
        best = min(matches, key=lambda m: (m['phash_distance'], m['dhash_distance'], m['method']))
        result.append({'pair_id': pair_id(a['id'], b['id']), 'gsd_id': a['id'],
            'trans10k_id': b['id'], 'trans10k_split': b['split'],
            'kind': 'perceptual_candidate', **best, 'matched_views': matches,
            'machine_relationship': 'candidate_only', 'automatic_same_source_decision': False})
    return sorted(result, key=lambda r: (r['phash_distance'], r['dhash_distance'], r['gsd_id'], r['trans10k_id']))


def exact_cross(g_rows, t_rows):
    by_pixel = defaultdict(list)
    for row in t_rows:
        by_pixel[row['pixel_sha256']].append(row)
    result = []
    for a in g_rows:
        for b in by_pixel[a['pixel_sha256']]:
            byte_equal = a['image_sha256'] == b['image_sha256']
            result.append({'pair_id': pair_id(a['id'], b['id']), 'gsd_id': a['id'],
                'trans10k_id': b['id'], 'trans10k_split': b['split'],
                'kind': 'exact_bytes' if byte_equal else 'exact_decoded_rgb',
                'machine_relationship': 'identical_image_bytes' if byte_equal else 'identical_shape_and_rgb_pixels',
                'image_bytes_equal': byte_equal, 'pixel_sha256': a['pixel_sha256'],
                'automatic_same_source_decision': False})
    return result


def subset_from_candidates(full, candidates):
    reasons = defaultdict(list)
    for row in candidates:
        reasons[row['gsd_id']].append({'pair_id': row['pair_id'],
            'trans10k_id': row['trans10k_id'], 'trans10k_split': row['trans10k_split'],
            'reason': 'exact_image_match' if row['kind'] != 'perceptual_candidate' else 'unresolved_perceptual_candidate'})
    retained = [r for r in full['samples'] if r['id'] not in reasons]
    subset = {**full, 'complete_pinned_source_split': False, 'effective_count': len(retained),
        'samples': retained, 'subset_status': 'provisional_pending_human_review',
        'subset_description': 'Excludes exact matches and unresolved cross-dataset candidates against all three Trans10K splits; not proof of no overlap.',
        'evaluation_authorized': False}
    exclusions = [{'gsd_id': r['id'], 'source_id': r['source_id'], 'reasons': reasons[r['id']]}
                  for r in full['samples'] if r['id'] in reasons]
    return subset, exclusions


def export_gsd(source, target, old_inventory):
    sys.path.insert(0, str(PROJECT / 'tmp/member4_data_deps'))
    import pyarrow.parquet as pq
    rows = pq.read_table(source).to_pylist()
    if len(rows) != 813 or len({r['image_id'] for r in rows}) != 813:
        raise ValueError('Pinned GSD test must contain exactly 813 unique source IDs')
    target.mkdir(parents=True, exist_ok=False)
    (target / 'images').mkdir(); (target / 'masks').mkdir()
    inventory, features, histogram = [], [], Counter()
    for index, source_row in enumerate(rows):
        raw, raw_mask = source_row['image']['bytes'], source_row['mask']['bytes']
        with Image.open(io.BytesIO(raw)) as opened:
            ext = {'JPEG': '.jpg', 'PNG': '.png'}[opened.format]
            image = opened.convert('RGB'); rgb = np.asarray(image)
            orientation = opened.getexif().get(274)
        with Image.open(io.BytesIO(raw_mask)) as opened:
            if opened.format != 'PNG' or opened.mode != 'L':
                raise ValueError('Expected native single-channel PNG mask')
            mask = np.asarray(opened).copy()
        if mask.shape != rgb.shape[:2] or not set(np.unique(mask).tolist()) <= {0, 255}:
            raise ValueError('GSD geometry/encoding mismatch')
        image_path, mask_path = f'images/{index:06d}{ext}', f'masks/{index:06d}.png'
        (target / image_path).write_bytes(raw)
        (target / mask_path).write_bytes(raw_mask)
        image_sha, mask_sha = sha256_file(target / image_path), sha256_file(target / mask_path)
        previous = old_inventory[source_row['image_id']]
        if (image_sha != previous['image_sha256'] or mask_sha != previous['mask_sha256']
                or hashlib.sha256(rgb.tobytes()).hexdigest() != previous['rgb_pixel_sha256']):
            raise ValueError('Export differs from pinned preparation audit')
        y, valid = decode_mask(target / mask_path, 'binary_0255', [])
        if not np.array_equal(y, mask == 255) or not valid.all():
            raise ValueError('Project mask reader does not preserve 255 foreground')
        fg = int(np.count_nonzero(mask == 255))
        histogram.update({'0': int(mask.size - fg), '255': fg})
        row = {'id': 'gsd/test/' + source_row['image_id'], 'source_id': source_row['image_id'],
            'source_row': index, 'source_image_path': source_row['image']['path'],
            'source_mask_path': source_row['mask']['path'], 'source_revision': REVISION,
            'image': image_path, 'mask': mask_path, 'height': image.height, 'width': image.width,
            'image_sha256': image_sha, 'mask_sha256': mask_sha, 'pixel_sha256': rgb_hash(rgb),
            'mask_pixel_sha256': hashlib.sha256(mask.tobytes()).hexdigest(),
            'foreground_pixels': fg, 'valid_pixels': int(mask.size), 'mask_mode': 'L',
            'mask_values': np.unique(mask).tolist(), 'exif_orientation': orientation}
        inventory.append(row)
        features.append(image_features(image))
    return inventory, features, dict(histogram)


def internal_duplicates(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['pixel_sha256']].append(r)
    duplicates = []
    for members in groups.values():
        if len(members) < 2:
            continue
        comparisons = []
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                with Image.open(DATA / a['mask']) as im: ma = np.asarray(im).copy()
                with Image.open(DATA / b['mask']) as im: mb = np.asarray(im).copy()
                comparisons.append({'ids': [a['id'], b['id']],
                    'image_bytes_equal': a['image_sha256'] == b['image_sha256'],
                    'mask_different_pixels': int(np.count_nonzero(ma != mb)), 'pixels': int(ma.size)})
        duplicates.append({'issue_id': f'GSD_INTERNAL_{len(duplicates)+1:03d}',
            'ids': [r['id'] for r in members], 'comparisons': comparisons,
            'full_test_action': 'retain_all_original_members_and_masks', 'human_review': 'pending'})
    return duplicates


def extract_trans(row):
    image_path, mask_path = Path(row['base']) / row['image'], Path(row['base']) / row['mask']
    if sha256_file(image_path) != row['image_sha256'] or sha256_file(mask_path) != row['mask_sha256']:
        raise ValueError('Trans10K source bytes changed: ' + row['id'])
    with Image.open(image_path) as im:
        image = im.convert('RGB'); rgb = np.asarray(image)
        if rgb.shape != (row['height'], row['width'], 3) or rgb_hash(rgb) != row['pixel_sha256']:
            raise ValueError('Trans10K decoded pixels differ: ' + row['id'])
        features = image_features(image)
    # Masks remain the final reviewed masks, including member1's recorded corrections.
    with Image.open(mask_path) as im:
        if im.size != (row['width'], row['height']):
            raise ValueError('Trans10K final mask geometry differs: ' + row['id'])
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1-4 workers')
    if any(p.exists() for p in (ROOT, DATA, MANIFESTS)):
        raise FileExistsError('Use new stage5 destinations; existing evidence must be preserved')
    source_dir = Path('data/GSD_source') / REVISION
    source = source_dir / 'data/test-00000-of-00001.parquet'
    source_record = read('runs/member4_readiness_20261114/gsd_source.json')
    for entry in source_record['files']:
        if sha256_file(source_dir / entry['file']) != entry['sha256']:
            raise ValueError('Pinned GSD source file changed: ' + entry['file'])
    if sha256_file(source) != PARQUET_SHA:
        raise ValueError('Pinned parquet hash differs')
    ROOT.mkdir(parents=True); MANIFESTS.mkdir(parents=True)
    started = time.monotonic()
    policy = {'schema': 1, 'prepared_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'data preparation and review only; no model inference',
        'source_revision': REVISION, 'parquet_sha256': PARQUET_SHA,
        'trans10k_manifest_sha256': TRANS_PINS, 'retrieval_method': METHOD,
        'cross_only': True, 'top_k_truncation': False,
        'exact_rule': 'image-byte SHA256 and shape-prefixed decoded RGB SHA256',
        'full_set_rule': 'preserve all 813 source rows including internal duplicates and differing labels',
        'supplementary_rule': 'exclude exact matches, confirmed same-source photographs, and unresolved candidates against train/validation/test; keep confirmed different captures with a separate same-scene flag',
        'internal_duplicate_rule': 'retain both source members and original labels; disclose the discrepancy in both full and supplementary reports',
        'near_candidate_decision': 'not automatic; member4 and member1 must independently review',
        'member5_rule_review': 'pending', 'evaluation_authorized': False,
        'limitations': ['Finite full-image/center-crop hashes can miss strong crops, rotations, and viewpoint changes.',
                       'No candidate is not proof of no overlap.',
                       'Same scene with a different capture is not equivalent to the same source photograph.',
                       'GSD glass-surface and Trans10K transparent-object targets differ.']}
    write_json(ROOT / 'policy_before_retrieval.json', policy)
    old = {r['id']: r for r in read('runs/member4_readiness_20261114/gsd_inventory.json')}
    g_rows, gf, histogram = export_gsd(source, DATA, old)
    full = {'schema_version': 1, 'dataset': 'gsd_hf_test_' + REVISION[:8],
        'root': Path(os.path.relpath(DATA.resolve(), MANIFESTS.resolve())).as_posix(),
        'root_relative_to_manifest': True, 'split': 'test', 'encoding': 'binary_0255',
        'void_values': [], 'synthetic': False, 'complete_official_split': False,
        'complete_pinned_source_split': True, 'effective_count': 813,
        'source': {**source_record, 'parquet_sha256': PARQUET_SHA,
                   'export': 'original image and mask bytes; no recompression, rotation or mask rescaling'},
        'data_protocol': {'id': 'member4_gsd_20261116', 'kind': 'external_transfer_test',
            'status': 'awaiting_human_review', 'policy_sha256': sha256_file(ROOT / 'policy_before_retrieval.json')},
        'evaluation_authorized': False, 'mask_pixel_histogram': histogram, 'samples': g_rows}
    write_json(MANIFESTS / 'test_full.json', full)
    read_manifest(MANIFESTS / 'test_full.json')
    write_json(ROOT / 'gsd_inventory.json', g_rows)
    write_json(ROOT / 'within_gsd_duplicates.json', internal_duplicates(g_rows))
    np.save(ROOT / 'gsd_features.npy', np.asarray(gf, dtype=np.uint64), allow_pickle=False)
    print('GSD: 813 byte-preserving exports, native labels and project reader checks passed.', flush=True)
    t_rows = []
    for split, digest in TRANS_PINS.items():
        manifest = Path(f'manifests/member1_final_20261029/v1/{split}.json')
        if sha256_file(manifest) != digest:
            raise ValueError('Frozen Trans10K manifest differs: ' + split)
        meta, base = read_manifest(manifest)
        if len(meta['samples']) != EXPECTED[split]:
            raise ValueError('Wrong final Trans10K membership')
        t_rows.extend({**r, 'split': split, 'base': base.relative_to(PROJECT).as_posix()} for r in meta['samples'])
    write_json(ROOT / 'trans10k_inventory.json', t_rows)
    tf = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, f in enumerate(pool.map(extract_trans, t_rows), 1):
            tf.append(f)
            if n % 500 == 0:
                print(f'Trans10K bytes/pixels/features: {n}/{len(t_rows)}; {time.monotonic()-started:.1f}s', flush=True)
    np.save(ROOT / 'trans10k_features.npy', np.asarray(tf, dtype=np.uint64), allow_pickle=False)
    write_json(ROOT / 'feature_ids.json', {'gsd': [r['id'] for r in g_rows], 'trans10k': [r['id'] for r in t_rows]})
    exact = exact_cross(g_rows, t_rows)
    near = cross_candidates(gf, tf, g_rows, t_rows)
    candidates = exact + near
    write_json(ROOT / 'cross_candidates.json', {'exact': exact, 'near': near})
    subset, exclusions = subset_from_candidates(full, candidates)
    write_json(MANIFESTS / 'test_supplementary_pending_review.json', subset)
    if subset['samples']:
        read_manifest(MANIFESTS / 'test_supplementary_pending_review.json')
    write_json(ROOT / 'supplementary_exclusions_pending_review.json', exclusions)
    sample_indices = np.linspace(0, 812, 20, dtype=int).tolist()
    review = {'schema': 1, 'audit_binding_sha256': sha256_file(ROOT / 'cross_candidates.json'),
        'policy_sha256': sha256_file(ROOT / 'policy_before_retrieval.json'),
        'full_manifest_sha256': sha256_file(MANIFESTS / 'test_full.json'),
        'allowed_decisions': ['same_source', 'different_capture_same_scene', 'not_same_source', 'uncertain'],
        'pairs': [{**r, 'member4': {'reviewer': '', 'decision': 'pending', 'reason': ''},
                   'member1': {'reviewer': '', 'decision': 'pending', 'reason': ''}}
                  for r in candidates],
        'labels': [{'gsd_id': g_rows[i]['id'], 'source_row': i, 'status': 'pending',
                    'reviewer': '', 'reason': ''} for i in sample_indices],
        'internal_duplicate_issues': [{'issue_id': r['issue_id'], 'status': 'pending', 'reviewer': '', 'reason': ''}
                                      for r in read(ROOT / 'within_gsd_duplicates.json')],
        'member5': {'reviewer': '', 'policy_approved': False, 'target_semantics_acknowledged': False,
                    'limitations_acknowledged': False, 'internal_duplicate_policy_approved': False, 'reason': ''},
        'user_stage5_accepted': False, 'evaluation_authorized': False}
    write_json(ROOT / 'human_review_template.json', review)
    report = {'automatic_preparation': 'passed', 'human_review': 'pending', 'stage6_started': False,
        'gsd_rows': 813, 'gsd_unique_ids': 813, 'byte_preserving_exports_verified': 813,
        'project_reader_masks_verified': 813, 'mask_pixel_histogram': histogram,
        'empty_masks': [r['id'] for r in g_rows if r['foreground_pixels'] == 0],
        'all_foreground_masks': [r['id'] for r in g_rows if r['foreground_pixels'] == r['valid_pixels']],
        'trans10k_images': len(t_rows), 'trans10k_split_counts': EXPECTED,
        'source_images_and_masks_verified': len(t_rows), 'decoded_rgb_hashes_verified': len(t_rows),
        'cross_pairs_examined_per_view': len(g_rows) * len(t_rows), 'views_tested': 3,
        'exact_pairs': len(exact), 'near_candidate_pairs': len(near),
        'candidate_split_counts': dict(Counter(r['trans10k_split'] for r in candidates)),
        'gsd_rows_with_candidates': len(exclusions), 'provisional_supplementary_rows': len(subset['samples']),
        'preview_row_indices': sample_indices, 'model_evaluation': 'not_run',
        'seconds': time.monotonic() - started, 'source': source_record,
        'full_manifest_sha256': sha256_file(MANIFESTS / 'test_full.json'),
        'supplementary_manifest_sha256': sha256_file(MANIFESTS / 'test_supplementary_pending_review.json')}
    write_json(ROOT / 'audit.json', report)
    print(json.dumps({k: report[k] for k in ('gsd_rows', 'trans10k_images', 'exact_pairs',
        'near_candidate_pairs', 'gsd_rows_with_candidates', 'provisional_supplementary_rows', 'seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
