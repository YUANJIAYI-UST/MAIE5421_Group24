"""Versioned single-owner GSD review; retain the audit and never run inference."""
import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.member4_gsd_prepare import ROOT, MANIFESTS, read
from transparent_seg.data import read_manifest
from transparent_seg.prepare import sha256_file, write_json

WORKFLOW = {
    'id': 'gsd_single_owner_v2',
    'authorized_change': 'User requested single-owner review and reduced team handoff before stage6.',
    'required': ['one named owner', 'all fixed label samples viewed',
                 'internal duplicate policy acknowledged', 'questions resolved or investigated',
                 'explicit stage5 acceptance'],
    'optional': ['nearest-neighbour diagnostic browsing', 'teammate help on specific questions'],
    'unchanged': ['source bytes and membership', 'retrieval algorithm and thresholds',
                  'retain internal duplicates and original labels',
                  'exclude same-source and unresolved cross candidates from supplementary set',
                  'no GSD score-based selection', 'stage6 requires a separate instruction'],
    'supersedes': 'Mandatory member1/member5 sign-offs and per-normal-image written reasons.',
}
DEFAULT_OUTPUT = ROOT / 'review_single_owner_v2'
BINDINGS = {
    'candidate_sha256': 'cross_candidates.json',
    'retrieval_policy_sha256': 'policy_before_retrieval.json',
    'audit_sha256': 'audit.json',
    'diagnostics_sha256': 'retrieval_diagnostics.json',
    'internal_duplicates_sha256': 'within_gsd_duplicates.json',
    'legacy_template_sha256': 'human_review_template.json',
    'preview_index_sha256': 'preview_index.json',
}
WORKFLOW_SHA = hashlib.sha256(json.dumps(WORKFLOW, sort_keys=True,
                                        ensure_ascii=False).encode()).hexdigest()


def make_template(source=ROOT, manifest=MANIFESTS / 'test_full.json'):
    source, manifest = Path(source), Path(manifest)
    old = read(source / 'human_review_template.json')
    for key, path in [('audit_binding_sha256', source / 'cross_candidates.json'),
                      ('policy_sha256', source / 'policy_before_retrieval.json'),
                      ('full_manifest_sha256', manifest)]:
        if old[key] != sha256_file(path):
            raise ValueError('Original audit binding changed: ' + key)
    full = read(manifest)
    samples = {r['id']: r for r in full['samples']}
    labels = [{'number': n, 'gsd_id': row['gsd_id'], 'source_row': row['source_row']}
              for n, row in enumerate(old['labels'], 1)]
    if not labels or len({r['gsd_id'] for r in labels}) != len(labels):
        raise ValueError('Label sample list is empty or duplicated')
    if any(r['gsd_id'] not in samples for r in labels):
        raise ValueError('Label sample outside full manifest')
    pairs = [{**{k: v for k, v in r.items() if k not in ('member4', 'member1')},
              'decision': 'pending', 'reason': ''} for r in old['pairs']]
    issues = [{'issue_id': r['issue_id'], 'ids': r['ids'],
               'status': 'pending', 'note': ''}
              for r in read(source / 'within_gsd_duplicates.json')]
    return {
        'schema': 2, 'workflow': WORKFLOW['id'], 'workflow_sha256': WORKFLOW_SHA,
        'source_bindings': {**{k: sha256_file(source / p) for k, p in BINDINGS.items()},
                            'full_manifest_sha256': sha256_file(manifest)},
        'full_count': len(full['samples']), 'label_samples': labels,
        'reviewer': '', 'reviewed_on': '',
        'label_review': {'status': 'pending', 'summary': ''},
        'internal_duplicate_reviews': issues,
        'diagnostic_review': {'status': 'not_viewed', 'note': ''},
        'questions': [], 'pairs': pairs,
        'policy_acknowledged': False, 'stage5_accepted': False,
        'evaluation_authorized': False,
    }


def validate_decisions(review, template):
    """Validate real owner feedback and evidence; no teammate signatures required."""
    for key in ('schema', 'workflow', 'workflow_sha256', 'source_bindings',
                'full_count', 'label_samples'):
        if review.get(key) != template[key]:
            raise ValueError('Review evidence changed: ' + key)
    if review.get('evaluation_authorized') is not False:
        raise ValueError('Review cannot authorize stage6')
    if not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip():
        raise ValueError('Owner name is required')
    try:
        date.fromisoformat(review.get('reviewed_on', ''))
    except (ValueError, TypeError):
        raise ValueError('Review date must be YYYY-MM-DD') from None
    label = review.get('label_review', {})
    if label.get('status') not in ('no_obvious_issue', 'issues_recorded'):
        raise ValueError('All fixed label samples must be viewed')
    expected_issues = template['internal_duplicate_reviews']
    actual_issues = review.get('internal_duplicate_reviews', [])
    if len(actual_issues) != len(expected_issues):
        raise ValueError('Internal duplicate review missing')
    for actual, expected in zip(actual_issues, expected_issues):
        if any(actual.get(k) != expected[k] for k in ('issue_id', 'ids')):
            raise ValueError('Internal duplicate evidence changed')
        if actual.get('status') != 'keep_original_and_disclose':
            raise ValueError('Internal duplicate question requires investigation')
    diagnostic = review.get('diagnostic_review', {})
    if diagnostic.get('status') not in ('not_viewed', 'viewed_no_concern'):
        raise ValueError('Diagnostic question requires investigation')
    questions = review.get('questions', [])
    if not isinstance(questions, list):
        raise ValueError('Questions must be a list')
    sample_ids = {r['gsd_id'] for r in template['label_samples']}
    for q in questions:
        if q.get('kind') != 'label':
            raise ValueError('Cross-dataset nomination requires investigation and a new candidate audit')
        if q.get('gsd_id') not in sample_ids or not str(q.get('observation', '')).strip():
            raise ValueError('Question must identify a sampled image and observation')
        if (q.get('status') != 'resolved_keep_original'
                or not str(q.get('resolution', '')).strip()):
            raise ValueError('Unresolved label question requires investigation')
    if (label['status'] == 'issues_recorded') != bool(questions):
        raise ValueError('Label summary and question records disagree')
    if review.get('policy_acknowledged') is not True or review.get('stage5_accepted') is not True:
        raise ValueError('Owner policy acknowledgement and stage5 acceptance are required')
    pairs, originals = review.get('pairs', []), template['pairs']
    if len(pairs) != len(originals):
        raise ValueError('Candidate count changed')
    decisions = []
    for actual, expected in zip(pairs, originals):
        evidence = {k: v for k, v in expected.items() if k not in ('decision', 'reason')}
        if any(actual.get(k) != v for k, v in evidence.items()):
            raise ValueError('Candidate evidence changed')
        choice = actual.get('decision')
        if (choice not in ('same_source', 'not_same_source', 'different_capture_same_scene', 'uncertain')
                or not str(actual.get('reason', '')).strip()):
            raise ValueError('Candidate decision and reason are required')
        if expected['kind'] != 'perceptual_candidate' and choice != 'same_source':
            raise ValueError('An exact decoded-image match cannot be marked different')
        decisions.append({**evidence, 'reviewer': review['reviewer'], 'decision': choice,
                          'reason': actual['reason'],
                          'exclude_from_supplementary': choice in ('same_source', 'uncertain')})
    return decisions


def freeze_review(review_path, output, source=ROOT, manifest=MANIFESTS / 'test_full.json'):
    """Only write a new release after supplied human review; never evaluate."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve existing releases; choose a new output directory')
    template, review = make_template(source, manifest), read(review_path)
    decisions = validate_decisions(review, template)
    full, base = read_manifest(manifest)
    excluded = {d['gsd_id'] for d in decisions if d['exclude_from_supplementary']}
    retained = [r for r in full['samples'] if r['id'] not in excluded]
    if not retained:
        raise ValueError('Supplementary set would be empty')
    output.mkdir(parents=True)
    full = {**full, 'root': Path(os.path.relpath(base, output.resolve())).as_posix(),
            'root_relative_to_manifest': True, 'review_status': 'single_owner_spot_check_recorded',
            'review_workflow': WORKFLOW['id'], 'review_record_sha256': sha256_file(review_path),
            'evaluation_authorized': False, 'next_gate': 'Separate stage6 instruction required',
            'data_protocol': {**full.get('data_protocol', {}),
                              'status': 'single_owner_spot_check_recorded'}}
    subset = {**full, 'samples': retained, 'effective_count': len(retained),
              'complete_pinned_source_split': False, 'subset_status': 'single_owner_reviewed',
              'subset_limitations': 'Finite retrieval cannot prove no overlap; original internal duplicates retained.'}
    write_json(output / 'test_full.json', full)
    write_json(output / 'test_supplementary.json', subset)
    write_json(output / 'human_review_record.json', review)
    write_json(output / 'review_workflow.json', WORKFLOW)
    write_json(output / 'pair_decisions.json', decisions)
    write_json(output / 'exclusions.json', [d for d in decisions if d['exclude_from_supplementary']])
    write_json(output / 'same_scene_different_capture.json',
               [d for d in decisions if d['decision'] == 'different_capture_same_scene'])
    read_manifest(output / 'test_full.json'); read_manifest(output / 'test_supplementary.json')
    write_json(output / 'release.json', {
        'workflow': WORKFLOW['id'], 'full_count': len(full['samples']),
        'supplementary_count': len(retained), 'same_membership': len(retained) == len(full['samples']),
        'owner': review['reviewer'], 'label_samples_viewed': len(template['label_samples']),
        'diagnostic_review': review['diagnostic_review'], 'independent_team_signoffs_required': False,
        'stage6_started': False, 'evaluation_authorized': False,
        'manifest_sha256': {p.name: sha256_file(p) for p in output.glob('test*.json')},
    })
    print('Single-owner review recorded; stage6 has not started.', flush=True)


def build_packet(output=DEFAULT_OUTPUT, source=ROOT, manifest=MANIFESTS / 'test_full.json'):
    output, source = Path(output), Path(source)
    if output.exists():
        raise FileExistsError('Preserve old review packages; choose a new output directory')
    template = make_template(source, manifest)
    index = read(source / 'preview_index.json')
    if [r['gsd_id'] for r in index['labels']] != [r['gsd_id'] for r in template['label_samples']]:
        raise ValueError('Preview sample identities do not match')
    output.mkdir(parents=True)
    paths = [r['file'] for r in index['labels']] + [r['file'] for r in index['pairs']]
    paths += [f"previews/{r['issue_id']}.png" for r in template['internal_duplicate_reviews']]
    paths += [p.relative_to(source).as_posix() for p in sorted((source / 'previews').glob('*_contact_*.png'))]
    # Optional diagnostics use contact sheets without the legacy two-reviewer caption.
    for relative in paths:
        src, dst = source / relative, output / relative
        if not src.resolve().is_relative_to(source.resolve()):
            raise ValueError('Preview outside source directory')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for name in ('audit.json', 'cross_candidates.json', 'retrieval_diagnostics.json',
                 'within_gsd_duplicates.json', 'policy_before_retrieval.json'):
        shutil.copyfile(source / name, output / name)
    write_json(output / 'human_review_template.json', template)
    write_json(output / 'review_workflow.json', WORKFLOW)
    packet_index = {**index, 'diagnostic_pairs': [
        {**r, 'file': f"previews/diagnostics_contact_{(n // 6) + 1:02d}.png"}
        for n, r in enumerate(index.get('diagnostic_pairs', []))]}
    write_json(output / 'preview_index.json', packet_index)
    payload = json.dumps(template, ensure_ascii=False).replace('<', '\\u003c')
    page = (Path(__file__).parent / 'templates/gsd_single_owner_review.html').read_text(encoding='utf-8')
    (output / 'review.html').write_text(page.replace('__REVIEW_DATA__', payload), encoding='utf-8', newline='\n')
    instructions = Path('docs/25_MEMBER4_SIMPLE_REVIEW_GUIDE.txt')
    if instructions.is_file():
        shutil.copyfile(instructions, output / '使用说明.txt')
    write_json(output / 'files_sha256.json', {p.relative_to(output).as_posix(): sha256_file(p)
                                              for p in sorted(output.rglob('*')) if p.is_file()})
    packet = output / 'GSD_single_owner_review_v2.zip'
    with zipfile.ZipFile(packet, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and p != packet:
                z.write(p, p.relative_to(output).as_posix())
    print(json.dumps({'output': str(output), 'packet': str(packet), 'packet_sha256': sha256_file(packet),
                      'human_review': 'pending', 'stage6_started': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build'); p.add_argument('--output', default=str(DEFAULT_OUTPUT))
    p = sub.add_parser('freeze'); p.add_argument('--review', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build_packet(args.output)
    else:
        freeze_review(args.review, args.output)
