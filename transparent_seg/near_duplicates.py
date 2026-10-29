"""Deterministic perceptual candidates, not an automatic exclusion algorithm."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .prepare import sha256_file, write_json
from .prepare_reviewed import inside

METHOD = {'version': 1, 'phash_bits': 63, 'dhash_bits': 64,
          'global_phash_distance': 8, 'center_to_global_phash_distance': 6,
          'maximum_dhash_distance': 12, 'center_crop_fraction': 0.8,
          'views': ['whole_image', 'center_80_percent'],
          'decision': 'candidate_only_no_automatic_exclusion'}

# Orthonormal DCT-II low-frequency basis; only the required 8x8 coefficients.
# Keeping this in NumPy avoids importing unrelated SciPy sparse native modules.
DCT_BASIS = np.cos(np.pi / 32 * np.arange(8)[:, None] * (np.arange(32)[None, :] + 0.5)) * np.sqrt(2 / 32)
DCT_BASIS[0] /= np.sqrt(2)


def hash_view(image):
    gray = image.convert('L')
    values = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), np.float64)
    # Explicit reductions avoid platform BLAS floating-status warnings and thread variability.
    horizontal = np.einsum('ki,ij->kj', DCT_BASIS, values, optimize=False)
    low = np.einsum('kj,lj->kl', horizontal, DCT_BASIS, optimize=False).ravel()
    if not np.isfinite(low).all():
        raise ValueError('Non-finite perceptual DCT coefficients')
    bits = low > np.median(low[1:])
    bits[0] = False
    phash = int.from_bytes(np.packbits(bits).tobytes(), 'big')
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), np.int16)
    dhash = int.from_bytes(np.packbits(small[:, 1:] > small[:, :-1]).tobytes(), 'big')
    return phash, dhash


def image_features(image):
    # A fixed thumbnail bounds memory while keeping the original aspect ratio.
    image = image.convert('RGB')
    image.thumbnail((256, 256), Image.Resampling.LANCZOS)
    w, h = image.size
    margin_x, margin_y = int(w * 0.1), int(h * 0.1)
    center = image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
    return hash_view(image), hash_view(center)


def find_candidates(features, rows, block_size=256):
    features = np.asarray(features, dtype=np.uint64)
    found = {}
    size = len(rows)
    for left_view, right_view, threshold, method in ((0, 0, 8, 'whole_vs_whole'),
            (0, 1, 6, 'whole_vs_center80'), (1, 0, 6, 'center80_vs_whole')):
        for start in range(0, size, block_size):
            stop = min(size, start + block_size)
            pd = np.bitwise_count(features[start:stop, left_view, 0, None] ^ features[None, :, right_view, 0])
            dd = np.bitwise_count(features[start:stop, left_view, 1, None] ^ features[None, :, right_view, 1])
            mask = (pd <= threshold) & (dd <= 12)
            # Compare every unordered pair without truncating the result set.
            mask &= np.arange(size)[None, :] > np.arange(start, stop)[:, None]
            ii, jj = np.nonzero(mask)
            for i, j in zip(ii.tolist(), jj.tolist()):
                left, right = start + i, j
                if rows[left]['pixel_sha256'] == rows[right]['pixel_sha256']:
                    continue  # Exact groups are already reported separately.
                key = (left, right)
                score = (int(pd[i, j]), int(dd[i, j]), method)
                if key not in found or score < found[key]:
                    found[key] = score
    candidates = []
    for (i, j), (pd, dd, method) in found.items():
        a, b = rows[i], rows[j]
        ids = [a['id'], b['id']]
        candidates.append({'pair_id': hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:20],
                           'ids': ids, 'splits': [a['split'], b['split']],
                           'cross_split': a['split'] != b['split'], 'phash_distance': pd,
                           'dhash_distance': dd, 'method': method,
                           'source_image_sha256': [a['image_sha256'], b['image_sha256']],
                           'decision': 'pending', 'reviewer': None, 'reason': '',
                           'automatic_exclusion': False})
    return sorted(candidates, key=lambda c: (not c['cross_split'], c['phash_distance'], c['dhash_distance'], c['ids']))


def retrieve(root, inventory_path, output, workers=4, nominated_pairs=None):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError('Use a new perceptual retrieval output directory')
    if not 1 <= workers <= 8:
        raise ValueError('Use 1-8 workers')
    inventory = json.loads(Path(inventory_path).read_text())
    rows = sorted(inventory['samples'], key=lambda r: r['id'])
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source IDs')
    output.mkdir(parents=True)
    started = time.monotonic()

    def extract(row):
        path = inside(root, row['image'])
        if sha256_file(path) != row['image_sha256']:
            raise ValueError(f'Source image changed: {row["id"]}')
        with Image.open(path) as image:
            # JPEG draft is deliberately not used: feature extraction uses a single fixed decode path.
            return image_features(image)

    features = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, value in enumerate(pool.map(extract, rows), 1):
            features.append(value)
            if n % 1000 == 0:
                print(f'Perceptual features: {n}/{len(rows)}; elapsed {time.monotonic()-started:.1f}s', flush=True)
    features = np.array(features, dtype=np.uint64)
    candidates = find_candidates(features, rows)
    by_id = {r['id']: r for r in rows}
    keys = {tuple(c['ids']) for c in candidates}
    for pair in nominated_pairs or []:
        ids = sorted(pair)
        if len(ids) != 2 or any(i not in by_id for i in ids):
            raise ValueError('Invalid manually nominated pair')
        if tuple(ids) not in keys:
            a, b = [by_id[i] for i in ids]
            candidates.append({'pair_id': hashlib.sha256('\n'.join(ids).encode()).hexdigest()[:20],
                               'ids': ids, 'splits': [a['split'], b['split']], 'cross_split': a['split'] != b['split'],
                               'phash_distance': None, 'dhash_distance': None, 'method': 'previous_visual_candidate',
                               'source_image_sha256': [a['image_sha256'], b['image_sha256']],
                               'decision': 'pending', 'reviewer': None, 'reason': '', 'automatic_exclusion': False})
    np.save(output / 'features.npy', features, allow_pickle=False)
    write_json(output / 'feature_ids.json', [r['id'] for r in rows])
    report = {'schema': 'trans10k_perceptual_candidates_v1', 'source_inventory_sha256': sha256_file(inventory_path),
              'method': METHOD, 'images_scanned': len(rows), 'source_hashes_verified': len(rows),
              'candidate_count': len(candidates), 'cross_split_count': sum(c['cross_split'] for c in candidates),
              'split_pair_counts': dict(Counter('-'.join(sorted(c['splits'])) for c in candidates)),
              'human_review': 'pending', 'data_changed': False, 'exclusions_applied': 0,
              'limitations': ['Finite perceptual thresholds are candidate retrieval, not proof of no leakage.',
                             'Strong crop, rotation, viewpoint changes and similar scenes may be missed.',
                             'Visually similar objects are not automatically the same source photograph.'],
              'candidates': candidates}
    write_json(output / 'candidates.json', report)
    print(json.dumps({k: report[k] for k in ('images_scanned', 'candidate_count', 'cross_split_count', 'split_pair_counts')}, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data/Trans10K')
    p.add_argument('--inventory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--nominated-pairs', help='JSON list of source-image ID pairs')
    a = p.parse_args()
    nominated = json.loads(Path(a.nominated_pairs).read_text()) if a.nominated_pairs else []
    retrieve(a.root, a.inventory, a.output, a.workers, nominated)


if __name__ == '__main__':
    main()
