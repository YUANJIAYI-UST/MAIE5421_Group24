"""Two-stage V2 category isolation, preserving all V1 binary targets.

Select using train/validation only, persist selection, then build test subsets.
All outputs are review candidates; author-mirror and D01/D02 reviews stay open.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
from itertools import combinations
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .data import read_manifest
from .data_v2 import CLASS_NAMES, category_array
from .prepare import sha256_file, write_json
from .prepare_reviewed import inside, cross_overlap


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_inventory(inventory, splits):
    if inventory.get('schema') != 'trans10k_v2_alignment_inventory_v1':
        raise ValueError('Expected audited V2 alignment inventory')
    if inventory.get('class_names') != list(CLASS_NAMES):
        raise ValueError('V2 class order differs from author definition')
    rows = inventory['samples']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate V2 inventory IDs')
    if set(r['split'] for r in rows) != set(splits):
        raise ValueError('Unexpected splits: selection must use train/validation only')
    for row in rows:
        if row['split'] != row['id'].split('/')[0]:
            raise ValueError('V2 ID/split mismatch')
        counts = row['category_pixel_counts']
        if len(counts) != 12 or any(type(n) != int or n < 0 for n in counts):
            raise ValueError('Invalid V2 category pixel counts')
        if row['category_ids'] != [i for i,n in enumerate(counts) if i and n]:
            raise ValueError('V2 category presence/count mismatch')
        if row['category_filter_eligible'] != (not row['issues']):
            raise ValueError('V2 quarantine status is inconsistent')
    return {r['id']: r for r in rows}


def eligible_base(base_dir, inventory, splits):
    lookup = validate_inventory(inventory, splits)
    manifests, eligible, quarantined = {}, {}, []
    common_root = None
    for split in splits:
        meta, root = read_manifest(Path(base_dir)/f'{split}.json')
        if meta['split'] != split or meta['encoding'] != 'trans10k_v1_rgb' or meta.get('void_values'):
            raise ValueError('Holdout must retain V1 binary encoding and split identity')
        if common_root is not None and root != common_root:
            raise ValueError('Base manifests must share a data root')
        common_root = root
        manifests[split] = meta
        eligible[split] = []
        for sample in meta['samples']:
            row = lookup.get(sample['id'])
            if row is None or row['split'] != split or row['v1_image_sha256'] != sample['image_sha256']:
                raise ValueError('V1/V2 inventory alignment does not cover the base manifest')
            if row['category_filter_eligible']:
                eligible[split].append(row)
            else:
                quarantined.append({'id':row['id'], 'split':split, 'reason_codes':row['issues'],
                                    'unknown_codes':row['unknown_codes'], 'decision':'conservative_quarantine_pending_human_review'})
    return manifests, eligible, quarantined, common_root


def category_counts(rows):
    return [{'class_id':i, 'class_name':name,
             'images':sum(i in r['category_ids'] if i else r['category_pixel_counts'][0] > 0 for r in rows),
             'native_v2_pixels':sum(r['category_pixel_counts'][i] for r in rows)} for i,name in enumerate(CLASS_NAMES)]


def select_holdout(base_dir, inventory_path, rules_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Selection is immutable; use a new version directory')
    inv, rules = load_json(inventory_path), load_json(rules_path)
    if inv['stage'] != 'train_validation_only' or inv['rules_sha256'] != sha256_file(rules_path):
        raise ValueError('Selection needs a matching train/validation-only audit')
    settings = rules['selection']
    if settings['splits'] != ['train','validation'] or settings['number_of_held_out_classes'] != 2:
        raise ValueError('This protocol selects exactly two classes from train/validation')
    manifests, eligible, quarantine, _ = eligible_base(base_dir, inv, ['train','validation'])
    counts = {s:category_counts(rows) for s,rows in eligible.items()}
    viable_classes = [c for c in range(1,12)
                      if counts['train'][c]['images'] >= settings['minimum_class_train_images']
                      and counts['validation'][c]['images'] >= settings['minimum_class_validation_images']]
    pairs = []
    for pair in combinations(viable_classes, 2):
        retained = {s:[r for r in rows if not set(pair).intersection(r['category_ids'])] for s,rows in eligible.items()}
        reasons = []
        for s in eligible:
            if len(retained[s]) < math.ceil(settings['minimum_retained_fraction_each_split'] * len(manifests[s]['samples'])):
                reasons.append(f'{s}_retained_below_minimum')
            if settings['require_each_remaining_class_in_both_splits']:
                present = {i for r in retained[s] for i in r['category_ids']}
                if set(range(1,12)) - set(pair) - present:
                    reasons.append(f'{s}_missing_seen_class')
        score = [min(counts['validation'][c]['images'] for c in pair),
                 min(counts['train'][c]['images'] for c in pair),
                 len(retained['train']), len(retained['validation'])]
        pairs.append({'classes':list(pair), 'score':score, 'eligible':not reasons, 'rejection_reasons':reasons})
    ranked = sorted([p for p in pairs if p['eligible']], key=lambda p:tuple(-x for x in p['score'])+tuple(p['classes']))
    if not ranked:
        raise ValueError('No pair satisfies the predeclared train/validation criteria; revise rules explicitly')
    chosen = ranked[0]['classes']
    stats = {'basis':'eligible base train/validation images; native V2 mask pixel units',
             'raw_v2_counts':{s:category_counts([r for r in inv['samples'] if r['split']==s]) for s in eligible},
             'eligible_base_counts':counts, 'quarantine':quarantine,
             'base_counts':{s:len(m['samples']) for s,m in manifests.items()},
             'eligible_counts':{s:len(r) for s,r in eligible.items()}}
    selection = {'schema':'trans10k_holdout_selection_v1', 'id':rules['id'],
                 'held_out_class_ids':chosen, 'held_out_class_names':[CLASS_NAMES[c] for c in chosen],
                 'selection_splits':['train','validation'], 'test_labels_used_for_selection':False,
                 'rules_sha256':sha256_file(rules_path), 'train_validation_inventory_sha256':sha256_file(inventory_path),
                 'base_manifest_sha256':{s:sha256_file(Path(base_dir)/f'{s}.json') for s in manifests},
                 'selection_settings':settings, 'winning_score':ranked[0]['score'],
                 'status':'frozen_computational_candidate_pending_owner_and_member5_review',
                 'source_type':rules['source_type'], 'mirror_revision':rules['mirror_revision'],
                 'v1_binary_targets':'unchanged', 'selection_is_not_a_claim_of_author_data_certification':True}
    output.mkdir(parents=True)
    write_json(output/'class_counts_train_validation.json', stats)
    write_json(output/'pair_candidates.json', pairs)
    selection['class_counts_sha256'] = sha256_file(output/'class_counts_train_validation.json')
    selection['pair_candidates_sha256'] = sha256_file(output/'pair_candidates.json')
    write_json(output/'selection.json', selection)
    print(json.dumps(selection, indent=2), flush=True)
    return selection


def build_holdout(base_dir, inventory_path, test_inventory_path, selection_path, rules_path, output, workers=4):
    output, base_dir = Path(output).resolve(), Path(base_dir)
    if output.exists():
        raise FileExistsError('Use a new holdout output directory')
    inv, test_inv, selection, rules = map(load_json, (inventory_path, test_inventory_path, selection_path, rules_path))
    if selection['schema'] != 'trans10k_holdout_selection_v1' or selection['selection_splits'] != ['train','validation'] or selection['test_labels_used_for_selection']:
        raise ValueError('Invalid train/validation-only selection')
    if sha256_file(inventory_path) != selection['train_validation_inventory_sha256'] or sha256_file(rules_path) != selection['rules_sha256']:
        raise ValueError('Selection evidence changed after freeze')
    if inv['rules_sha256'] != selection['rules_sha256'] or test_inv['rules_sha256'] != selection['rules_sha256']:
        raise ValueError('Audits used different alignment/encoding rules')
    if test_inv['stage'] != 'test_after_selection_freeze' or test_inv['selection_sha256_before_test_audit'] != sha256_file(selection_path):
        raise ValueError('Test audit must reference this exact pre-existing selection')
    if inv['mirror_revision'] != rules['mirror_revision'] or test_inv['mirror_revision'] != rules['mirror_revision'] or inv['v1_inventory_sha256'] != test_inv['v1_inventory_sha256']:
        raise ValueError('Train/validation/test source versions differ')
    if inv['v2_root_name'] != test_inv['v2_root_name'] or Path(inv['v2_root_name']).name != inv['v2_root_name']:
        raise ValueError('V2 root must be a shared data-root subdirectory')
    for split,digest in selection['base_manifest_sha256'].items():
        if sha256_file(base_dir/f'{split}.json') != digest:
            raise ValueError('Base train/validation manifest changed after freeze')
    manifests, eligible, quarantine, common = eligible_base(base_dir, inv, ['train','validation'])
    test_meta, test_eligible, test_quarantine, test_common = eligible_base(base_dir, test_inv, ['test'])
    if common != test_common:
        raise ValueError('Test has a different data root')
    test_manifest = test_meta['test']
    if (not test_manifest.get('complete_official_split') or test_manifest.get('synthetic')
            or len(test_manifest['samples']) != test_manifest.get('official_count')
            or {s['id'] for s in test_manifest['samples']} != {r['id'] for r in test_inv['samples']}):
        raise ValueError('Holdout build requires the complete unchanged official test base')
    manifests.update(test_meta)
    lookup = {**validate_inventory(inv, ['train','validation']), **validate_inventory(test_inv, ['test'])}
    held = set(selection['held_out_class_ids'])
    if len(held) != 2 or not held <= set(range(1,12)):
        raise ValueError('Exactly two valid foreground held-out IDs required')
    # Check the frozen choice can be recomputed from its original evidence.
    # No test data enters this recomputation; a modified choice is rejected.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        reproduced = select_holdout(base_dir, inventory_path, rules_path, Path(temp)/'selection')
        if reproduced != selection:
            raise ValueError('Selection does not match deterministic train/validation rule')
    v2root = inside(common, inv['v2_root_name'])

    def verify_category(row):
        for key in ('v2_image','v2_mask'):
            if sha256_file(inside(v2root,row[key])) != row[key+'_sha256']:
                raise ValueError(f'V2 source file changed: {row["id"]}')
        with Image.open(inside(v2root,row['v2_mask'])) as mask:
            ids,unknown = category_array(mask, rules['v2_encoding'])
            counts = np.bincount(ids[ids >= 0],minlength=12).tolist()
            if counts != row['category_pixel_counts'] or unknown != row['unknown_codes'] or list(mask.size) != row['v2_mask_size']:
                raise ValueError('V2 class statistics differ from actual mask pixels')

    base_samples = [s for m in manifests.values() for s in m['samples']]
    def verify_v1(sample):
        for key in ('image','mask'):
            if sha256_file(inside(common,sample[key])) != sample[key+'_sha256']:
                raise ValueError(f'V1 file changed: {sample["id"]}')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n,_ in enumerate(pool.map(verify_category, lookup.values()),1):
            if n % 1000 == 0:
                print(f'Verified V2 hashes and category pixels: {n}/{len(lookup)}',flush=True)
        for _ in pool.map(verify_v1,base_samples):
            pass
    source_rows = [{**s,'split':split} for split,m in manifests.items() for s in m['samples']]
    if cross_overlap(source_rows):
        raise ValueError('Exact cross-split image leakage in V1 base')
    protocol = {'id':selection['id'], 'kind':'v2_category_holdout',
                'selection_sha256':sha256_file(selection_path), 'rules_sha256':sha256_file(rules_path),
                'held_out_class_ids':sorted(held), 'source_dataset':'trans10k_v1',
                'status':'local_pilot_pending_human_review', 'human_visual_review':'pending',
                'member5_review':'pending', 'source_mirror_review':'pending',
                'perceptual_candidates_review':'pending', 'official_test_membership_preserved':True}
    selected, exclusions = {}, list(quarantine)
    for split in ('train','validation'):
        keep = {r['id'] for r in eligible[split] if not held.intersection(r['category_ids'])}
        selected[split] = [s for s in manifests[split]['samples'] if s['id'] in keep]
        for row in eligible[split]:
            overlap = sorted(held.intersection(row['category_ids']))
            if overlap:
                exclusions.append({'id':row['id'], 'split':split, 'reason_codes':['contains_held_out_category'],
                                   'held_out_class_ids':overlap, 'all_category_ids':row['category_ids'],
                                   'mixed_seen_unseen':bool(set(row['category_ids'])-held),
                                   'decision':'exclude_entire_image_from_training_and_validation'})
    selected['test'] = manifests['test']['samples']
    held_test = {r['id'] for r in test_eligible['test'] if held.intersection(r['category_ids'])}
    seen_test = {r['id'] for r in test_eligible['test'] if not held.intersection(r['category_ids'])}
    selected['test_heldout'] = [s for s in selected['test'] if s['id'] in held_test]
    selected['test_seen'] = [s for s in selected['test'] if s['id'] in seen_test]
    if any(not samples for samples in selected.values()):
        raise ValueError('Empty holdout subset; cannot issue a usable manifest')
    hashes, split_summary, split_counts = {}, {}, {}
    output.mkdir(parents=True)
    for name,samples in selected.items():
        split = 'test' if name.startswith('test') else name
        meta = copy.deepcopy(manifests[split])
        meta.pop('mask_channel_sum_histogram',None)  # Base histogram is invalid for a filtered subset.
        meta.update({'dataset':selection['id'], 'data_protocol':protocol, 'subset':name,
                     'root':Path(os.path.relpath(common,output)).as_posix(), 'root_relative_to_manifest':True,
                     'complete_official_split':name == 'test', 'effective_count':len(samples),
                     'category_encoding':rules['v2_encoding'], 'samples':[]})
        for source in samples:
            sample=copy.deepcopy(source)
            row=lookup[sample['id']]
            sample.update({'v2_category_mask':(Path(inv['v2_root_name'])/row['v2_mask']).as_posix(),
                           'v2_category_mask_sha256':row['v2_mask_sha256'],
                           'v2_category_ids':row['category_ids'], 'v2_category_pixel_counts':row['category_pixel_counts'],
                           'v2_mask_size':row['v2_mask_size'], 'v2_category_eligible':row['category_filter_eligible'],
                           'v2_to_v1_orientation':row['v2_to_v1_orientation'],
                           'v2_issues':row['issues']})
            if split in ('train','validation') and (not row['category_filter_eligible'] or held.intersection(row['category_ids'])):
                raise ValueError('Held-out category leaked into a training/validation sample')
            # Every original V1 sample field, including mask hash and transform, must be identical.
            if any(sample[k] != v for k,v in source.items()):
                raise AssertionError('V1 binary target/sample changed')
            meta['samples'].append(sample)
        write_json(output/f'{name}.json',meta)
        hashes[name]=sha256_file(output/f'{name}.json')
        split_summary[name]={'images':len(samples),'difficulty':dict(Counter(s['difficulty'] for s in samples))}
        split_counts[name]=category_counts([lookup[s['id']] for s in samples])
    if {s['id'] for s in selected['test']} != {s['id'] for s in manifests['test']['samples']}:
        raise AssertionError('Official test membership changed')
    for split in ('train','validation'):
        if any(split_counts[split][c]['native_v2_pixels'] for c in held):
            raise AssertionError('Nonzero held-out training/validation pixels')
    audit={'schema':'trans10k_holdout_audit_v1', 'status':'technical_checks_passed_human_review_pending',
           'data_protocol':protocol, 'held_out_class_names':selection['held_out_class_names'],
           'base_manifest_sha256':{s:sha256_file(base_dir/f'{s}.json') for s in manifests},
           'base_counts':{s:len(m['samples']) for s,m in manifests.items()}, 'splits':split_summary,
           'selection_uses_train_validation_only':True, 'official_test_membership_preserved':True,
           'v1_binary_targets_preserved':True, 'v1_binary_files_rehashed':len(base_samples)*2,
           'v2_category_files_redecoded':len(lookup), 'cross_split_exact_overlap':[],
           'held_out_pixels_train_validation':{s:{str(c):split_counts[s][c]['native_v2_pixels'] for c in sorted(held)} for s in ('train','validation')},
           'exclusions':sorted(exclusions,key=lambda r:r['id']), 'test_category_quarantine':test_quarantine,
           'test_subset_partition_complete':len(selected['test_heldout'])+len(selected['test_seen'])+len(test_quarantine)==len(selected['test']),
           'fine_category_pixel_units':'native V2 resolution, not original V1 resolution',
           'source_caveat':'third-party mirror; author archive equivalence unverified',
           'semantic_scope':'absence according to V2 labels; visual semantics and D02 near duplicates still need human review',
           'manifest_sha256':hashes}
    write_json(output/'class_counts.json',split_counts)
    write_json(output/'exclusions.json',audit['exclusions'])
    write_json(output/'selection.json',selection)
    write_json(output/'rules.json',rules)
    write_json(output/'audit.json',audit)
    write_json(output/'release.json',{'protocol_id':selection['id'],'manifest_sha256':hashes,
                                     'audit_sha256':sha256_file(output/'audit.json'),
                                     'selection_sha256':sha256_file(selection_path),
                                     'train_validation_inventory_sha256':sha256_file(inventory_path),
                                     'test_inventory_sha256':sha256_file(test_inventory_path)})
    print(json.dumps({'splits':split_summary, 'held_out':selection['held_out_class_names'], 'manifest_sha256':hashes},indent=2),flush=True)
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['select','build'])
    p.add_argument('--base',required=True)
    p.add_argument('--inventory',required=True)
    p.add_argument('--rules',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--test-inventory')
    p.add_argument('--selection')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.stage=='select':
        if args.test_inventory or args.selection:
            p.error('Selection cannot accept test evidence')
        select_holdout(args.base,args.inventory,args.rules,args.output)
    else:
        if not args.test_inventory or not args.selection or not 1<=args.workers<=8:
            p.error('Build requires --test-inventory and --selection, and 1-8 workers')
        build_holdout(args.base,args.inventory,args.test_inventory,args.selection,args.rules,args.output,args.workers)


if __name__=='__main__':
    main()
