"""Freeze the member-1 owner release from pinned audits and human decisions.

This is a deterministic filter of the existing V1 derivative. It never rewrites
source labels and never reselects held-out categories after test inspection.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import os
from pathlib import Path

from .data_v2 import CLASS_NAMES
from .prepare import sha256_file, write_json


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def duplicate_components(inventory, decisions):
    """Join exact-RGB duplicates and human A links, including removed bridges."""
    rows = {r['id']: r for r in inventory}
    if len(rows) != len(inventory):
        raise ValueError('Duplicate inventory IDs')
    parent = {sid: sid for sid in rows}

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    pixels = defaultdict(list)
    for r in inventory:
        pixels[r['pixel_sha256']].append(r['id'])
    for ids in pixels.values():
        for sid in ids[1:]:
            union(ids[0], sid)
    seen_keys = set()
    for decision in decisions:
        if decision['key'] in seen_keys or decision['owner_answer'] not in {'A', 'B', 'C', 'D'}:
            raise ValueError('Duplicate key or invalid human answer')
        seen_keys.add(decision['key'])
        ids, hashes = decision['ids'], decision['source_image_sha256']
        if len(ids) != 2 or len(hashes) != 2 or ids[0] == ids[1]:
            raise ValueError('Invalid pair')
        for sid, digest in zip(ids, hashes):
            if sid not in rows or rows[sid]['image_sha256'] != digest:
                raise ValueError('Human decision/source identity mismatch')
        if decision['owner_answer'] == 'A':
            union(*ids)
    groups = defaultdict(list)
    for sid in sorted(rows):
        groups[find(sid)].append(sid)
    return sorted((ids for ids in groups.values() if len(ids) > 1), key=lambda x: x[0])


def exclusion_plan(inventory, decisions, quarantine):
    rows = {r['id']: r for r in inventory}
    rank = {'train': 0, 'validation': 1, 'test': 2}
    exclusions, components = {}, []
    for ids in duplicate_components(inventory, decisions):
        highest = max(rank[rows[sid]['split']] for sid in ids)
        keep = [sid for sid in ids if rank[rows[sid]['split']] == highest]
        drop = [sid for sid in ids if rank[rows[sid]['split']] < highest]
        pairs = [d['number'] for d in decisions if d['owner_answer'] == 'A' and set(d['ids']) <= set(ids)]
        component = {'ids': ids, 'retained_highest_split_ids': keep,
                     'excluded_lower_split_ids': drop, 'human_A_pair_numbers': pairs}
        components.append(component)
        for sid in drop:
            exclusions[sid] = {'id': sid, 'split': rows[sid]['split'],
                               'reason_codes': ['confirmed_same_source_across_splits'],
                               'component_ids': ids, 'higher_priority_ids': keep,
                               'human_A_pair_numbers': pairs}
    for sid in sorted(quarantine):
        if sid not in rows or rows[sid]['split'] != 'train':
            raise ValueError('Coverage quarantine must reference a training image')
        item = exclusions.setdefault(sid, {'id': sid, 'split': 'train', 'reason_codes': []})
        item['reason_codes'].append('owner_flagged_annotation_coverage')
    return [exclusions[sid] for sid in sorted(exclusions)], components


def category_counts(samples, category_rows):
    values = [category_rows[s['id']]['category_pixel_counts'] for s in samples]
    return [{'class_id': c, 'class_name': name,
             'images': sum(v[c] > 0 for v in values),
             'native_v2_pixels': sum(v[c] for v in values)}
            for c, name in enumerate(CLASS_NAMES)]


def build(repo, output, policy_path):
    repo, output, policy_path = Path(repo).resolve(), Path(output).resolve(), Path(policy_path).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; releases are immutable')
    policy = load(policy_path)
    for relative, digest in policy['input_sha256'].items():
        if sha256_file(repo / relative) != digest:
            raise ValueError(f'Pinned input changed: {relative}')
    inputs = policy['inputs']
    inventory = load(repo / inputs['v1_inventory'])['samples']
    owner = load(repo / inputs['owner_decisions'])
    decisions = owner['records']
    if len(decisions) != 153 or Counter(d['owner_answer'] for d in decisions) != owner['counts']:
        raise ValueError('Incomplete or inconsistent owner decisions')
    human = load(repo / inputs['human_review'])
    flagged = sorted(r['ids'][0] for r in human['records'] if r['kind'] == 'geometry' and r['decision'] == 'incorrect')
    if flagged != sorted(policy['coverage_quarantine']):
        raise ValueError('Coverage quarantine differs from human decisions')
    base_path = repo / inputs['v1_base']
    old_d03 = repo / inputs['d03_base']
    base = {s: load(base_path / f'{s}.json') for s in ('train', 'validation', 'test')}
    old = {s: load(old_d03 / f'{s}.json') for s in ('train', 'validation', 'test', 'test_heldout', 'test_seen')}
    old_audit = load(base_path / 'audit.json')
    exclusions, components = exclusion_plan(inventory, decisions, flagged)
    excluded = {r['id'] for r in exclusions}
    if not {r['id'] for r in old_audit['exclusions']} <= excluded:
        raise ValueError('Cannot reintroduce previous exact duplicates')
    selection = load(old_d03 / 'selection.json')
    if selection['held_out_class_ids'] != policy['held_out_class_ids'] or selection['test_labels_used_for_selection']:
        raise ValueError('Frozen train/validation-only selection changed')
    category_rows = {}
    for key in ('v2_train_val_inventory', 'v2_test_inventory'):
        for r in load(repo / inputs[key])['samples']:
            if r['id'] in category_rows:
                raise ValueError('Duplicate V2 alignment ID')
            category_rows[r['id']] = r
    if set(category_rows) != {r['id'] for r in inventory}:
        raise ValueError('Incomplete V2 alignment')
    protocol = {'id': policy['id'] + '_v1', 'policy_sha256': sha256_file(policy_path),
                'status': 'owner_complete_pending_independent_review', 'source_dataset': 'trans10k_v1',
                'human_visual_review': '18_geometry_and_153_candidate_pairs_complete',
                'perceptual_candidates_review': 'owner_complete_with_one_recorded_test_only_uncertainty',
                'member1_delivery': 'complete', 'member5_review': 'pending',
                'official_test_membership_preserved': True}
    output.mkdir(parents=True)
    write_json(output / 'policy.json', policy)
    manifests, summaries = {}, {}
    inventory_rows = {r['id']: r for r in inventory}
    for kind, sources in (('v1', base), ('d03', old)):
        folder = output / kind
        folder.mkdir()
        p = deepcopy(protocol)
        if kind == 'd03':
            p.update({'id': policy['id'] + '_d03', 'kind': 'v2_category_holdout',
                      'held_out_class_ids': policy['held_out_class_ids'],
                      'selection_sha256': sha256_file(old_d03 / 'selection.json'),
                      'rules_sha256': selection['rules_sha256'],
                      'source_mirror_review': 'owner_accepted_pinned_mirror_with_provenance_limitation'})
        for name, source in sources.items():
            m = deepcopy(source)
            m['samples'] = [s for s in source['samples'] if s['id'] not in excluded]
            if name.startswith('test') and m['samples'] != source['samples']:
                raise ValueError('Official test or frozen test subset changed')
            m.update({'dataset': p['id'], 'data_protocol': p, 'effective_count': len(m['samples']),
                      'root': Path(os.path.relpath(repo / 'data', folder)).as_posix()})
            m['complete_official_split'] = name == 'test'
            if kind == 'v1':
                hist = Counter()
                for s in m['samples']:
                    hist.update(inventory_rows[s['id']]['mask_channel_sum_histogram'])
                m['mask_channel_sum_histogram'] = dict(sorted(hist.items(), key=lambda x: int(x[0])))
            write_json(folder / f'{name}.json', m)
            manifests[(kind, name)] = m
        counts = {name: len(manifests[kind, name]['samples']) for name in sources}
        hashes = {name: sha256_file(folder / f'{name}.json') for name in sources}
        class_table = {name: category_counts(manifests[kind, name]['samples'], category_rows) for name in sources}
        write_json(folder / 'class_counts.json', class_table)
        with (folder / 'class_counts.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['split', 'class_id', 'class_name', 'images', 'native_v2_pixels'])
            writer.writeheader()
            for split, values in class_table.items():
                for row in values:
                    writer.writerow({'split': split, **row})
        audit = {'schema': 'member1_owner_final_audit_v1', 'status': 'owner_complete_pending_independent_review',
                 'data_protocol': p, 'counts': counts, 'manifest_sha256': hashes,
                 'original_files_modified': False, 'official_test_membership_preserved': True,
                 'v1_binary_targets_preserved': True, 'review_counts': owner['counts'],
                 'coverage_quarantine': flagged, 'base_exclusions': exclusions,
                 'cross_split_exact_overlap': [], 'limitations': policy['limitations'],
                 'empty_foreground_policy': policy['empty_foreground_policy']}
        if kind == 'v1':
            audit['duplicate_components'] = components
            audit['corrections'] = old_audit['corrections']
            audit['correction_review'] = {'aligned_and_retained': 10, 'flagged_and_quarantined': 8,
                                          'original_correction_log_is_historical': True}
        else:
            active = {s['id'] for split in base for s in manifests['v1', split]['samples']}
            filtered = [r for r in load(old_d03 / 'audit.json')['exclusions'] if r['id'] in active]
            audit.update({'category_exclusions_after_base_filter': filtered,
                          'test_category_quarantine': [], 'test_subset_partition_complete': True,
                          'base_manifest_sha256': summaries['v1']['manifest_sha256'],
                          'selection_frozen_before_test': True, 'selection_recomputed': False,
                          'selection_base_is_historical': True,
                          'held_out_pixels_train_validation': {s: {str(c): class_table[s][c]['native_v2_pixels']
                             for c in policy['held_out_class_ids']} for s in ('train', 'validation')}})
            if any(n for v in audit['held_out_pixels_train_validation'].values() for n in v.values()):
                raise ValueError('Held-out category leaked into train/validation')
            # Keep the exact historical selection bytes and its pre-test hash.
            (folder / 'selection.json').write_bytes((old_d03 / 'selection.json').read_bytes())
            (folder / 'rules.json').write_bytes((old_d03 / 'rules.json').read_bytes())
        write_json(folder / 'audit.json', audit)
        release = {'protocol_id': p['id'], 'manifest_sha256': hashes,
                   'audit_sha256': sha256_file(folder / 'audit.json'), 'policy_sha256': sha256_file(policy_path)}
        write_json(folder / 'release.json', release)
        summaries[kind] = {'counts': counts, **release}
    handling = []
    retained = {s['id'] for split in base for s in manifests['v1', split]['samples']}
    for d in decisions:
        handling.append({**d, 'excluded_ids_in_final': sorted(set(d['ids']) & excluded),
                         'retained_ids_in_final': sorted(set(d['ids']) & retained),
                         'handling_complete': True,
                         'handling': policy['decision_handling'][d['owner_answer']]})
    write_json(output / 'pair_handling.json', handling)
    write_json(output / 'exclusions.json', exclusions)
    write_json(output / 'release.json', {'schema': 'member1_owner_final_release_v1',
               'status': protocol['status'], 'summaries': summaries,
               'owner_processing_complete': True, 'independent_review_completed': False,
               'all_pair_handling_recorded': len(handling), 'unique_base_exclusions': len(excluded),
               'official_counts': {'train': 5000, 'validation': 1000, 'test': 4428},
               'limitations': policy['limitations']})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--policy', default='configs/data/member1_final_20261029.json')
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.output, args.policy), indent=2))


if __name__ == '__main__':
    main()
