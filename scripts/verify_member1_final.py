"""Independent structural checks for the final owner release and its replay."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.prepare import sha256_file, write_json
from transparent_seg.prepare_final import load


def verify(repo, release, replay):
    repo, release, replay = map(Path, (repo, release, replay))
    policy = load(release / 'policy.json')
    for name, digest in policy['input_sha256'].items():
        assert sha256_file(repo / name) == digest, name
    original = load(repo / policy['inputs']['v1_inventory'])['samples']
    original_ids = {r['id'] for r in original}
    source = {r['id']: r for r in original}
    rows = {(kind, split): load(release / kind / f'{split}.json')['samples']
            for kind in ('v1', 'd03') for split in ('train', 'validation', 'test')}
    active = {r['id']: split for split in ('train', 'validation', 'test') for r in rows['v1', split]}
    exclusions = load(release / 'exclusions.json')
    excluded = {r['id'] for r in exclusions}
    assert len(excluded) == len(exclusions) and not excluded & set(active)
    assert excluded | set(active) == original_ids
    assert set(policy['coverage_quarantine']) <= excluded
    assert not any(source[sid]['split'] == 'test' for sid in excluded)
    # Build independent adjacency; removed vertices still connect components.
    graph = defaultdict(set)
    exact = defaultdict(list)
    for r in original:
        exact[r['pixel_sha256']].append(r['id'])
    edges = [list(zip(ids, ids[1:])) for ids in exact.values()]
    owner = load(repo / policy['inputs']['owner_decisions'])
    edges.append([tuple(r['ids']) for r in owner['records'] if r['owner_answer'] == 'A'])
    for group in edges:
        for a, b in group:
            graph[a].add(b)
            graph[b].add(a)
    visited = set()
    component_count = 0
    for start in sorted(graph):
        if start in visited:
            continue
        queue, component = [start], set()
        while queue:
            sid = queue.pop()
            if sid in component:
                continue
            component.add(sid)
            queue.extend(graph[sid] - component)
        visited.update(component)
        assert len({active[sid] for sid in component if sid in active}) <= 1, component
        component_count += 1
    assert 'train/images/3421.jpg' in excluded  # owner A for pair 31
    assert all(s in active for s in ('test/easy/images/6755.jpg', 'test/easy/images/8481.jpg'))
    category = {}
    for key in ('v2_train_val_inventory', 'v2_test_inventory'):
        category.update({r['id']: r for r in load(repo / policy['inputs'][key])['samples']})
    held = set(policy['held_out_class_ids'])
    for split in ('train', 'validation', 'test'):
        parent = {r['id']: r for r in rows['v1', split]}
        actual = {r['id']: r for r in rows['d03', split]}
        expected = {sid for sid in parent if split == 'test' or
                    (category[sid]['category_filter_eligible'] and not held & set(category[sid]['category_ids']))}
        assert set(actual) == expected, 'Holdout filtering is not complete'
        for sid, r in actual.items():
            assert all(r[k] == v for k, v in parent[sid].items()), 'V1 binary fields changed'
    for kind in ('v1', 'd03'):
        folder = release / kind
        audit, rel = load(folder / 'audit.json'), load(folder / 'release.json')
        assert sha256_file(folder / 'audit.json') == rel['audit_sha256']
        for split, digest in rel['manifest_sha256'].items():
            assert sha256_file(folder / f'{split}.json') == digest
        table = load(folder / 'class_counts.json')
        for split, counts in table.items():
            m = load(folder / f'{split}.json')
            assert len(m['samples']) == audit['counts'][split] == m['effective_count']
            values = [category[r['id']]['category_pixel_counts'] for r in m['samples']]
            for entry in counts:
                c = entry['class_id']
                assert entry['images'] == sum(v[c] > 0 for v in values)
                assert entry['native_v2_pixels'] == sum(v[c] for v in values)
        with (folder / 'class_counts.csv').open() as f:
            csv_rows = list(csv.DictReader(f))
        assert len(csv_rows) == sum(len(c) for c in table.values())
        for r in csv_rows:
            entry = table[r['split']][int(r['class_id'])]
            assert int(r['images']) == entry['images'] and int(r['native_v2_pixels']) == entry['native_v2_pixels']
    selection = repo / policy['inputs']['d03_base'] / 'selection.json'
    assert (release / 'd03/selection.json').read_bytes() == selection.read_bytes()
    assert load(selection)['selection_splits'] == ['train', 'validation']
    old_d03 = repo / policy['inputs']['d03_base']
    for subset in ('test', 'test_heldout', 'test_seen'):
        assert load(release / 'd03' / f'{subset}.json')['samples'] == load(old_d03 / f'{subset}.json')['samples']
    files = {p.relative_to(release): p for p in release.rglob('*') if p.is_file()}
    assert set(files) == {p.relative_to(replay) for p in replay.rglob('*') if p.is_file()}
    for name, p in files.items():
        assert p.read_bytes() == (replay / name).read_bytes(), name
    return {'status': 'passed', 'pinned_inputs_unchanged': len(policy['input_sha256']),
            'owner_pair_counts': owner['counts'], 'duplicate_components_checked': component_count,
            'cross_split_confirmed_source_components': 0, 'coverage_quarantine_count': len(policy['coverage_quarantine']),
            'unique_base_exclusions': len(excluded), 'excluded_by_split': dict(Counter(r['split'] for r in exclusions)),
            'frozen_selection_unchanged': True, 'official_test_and_subsets_unchanged': True,
            'holdout_filter_complete': True, 'v1_binary_fields_unchanged': True,
            'class_tables_recounted': True, 'byte_identical_reproduction_files': len(files),
            'source_file_hashes_and_actual_category_masks': 'See separate v1_verification.json and d03_verification.json',
            'member1_processing': 'complete', 'member5_review': 'pending'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='.')
    p.add_argument('--release', required=True)
    p.add_argument('--replay', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError('Preserve existing verification evidence')
    result = verify(a.repo, a.release, a.replay)
    write_json(a.output, result)
    print(result)
