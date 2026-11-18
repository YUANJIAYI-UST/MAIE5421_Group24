"""A single real owner can finish; incomplete/evidence-changing records cannot."""
from copy import deepcopy
import json

from PIL import Image
import pytest

from scripts.member4_gsd_review_simple import (
    WORKFLOW, WORKFLOW_SHA, make_template, validate_decisions, freeze_review,
)
from transparent_seg.prepare import sha256_file, write_json


def record_fixture(kind='perceptual_candidate'):
    template = {
        'schema': 2, 'workflow': WORKFLOW['id'], 'workflow_sha256': WORKFLOW_SHA,
        'source_bindings': {'full_manifest_sha256': 'pinned'}, 'full_count': 3,
        'label_samples': [{'number': 1, 'gsd_id': 'g0', 'source_row': 0}],
        'internal_duplicate_reviews': [{'issue_id': 'duplicate1', 'ids': ['g0', 'g1'],
                                       'status': 'pending', 'note': ''}],
        'pairs': [{'pair_id': 'p', 'gsd_id': 'g0', 'trans10k_id': 't0',
                   'trans10k_split': 'train', 'kind': kind, 'decision': 'pending', 'reason': ''}],
    }
    review = deepcopy(template)
    review.update(reviewer='Test Owner', reviewed_on='2026-11-16',
                  label_review={'status': 'no_obvious_issue', 'summary': ''},
                  diagnostic_review={'status': 'not_viewed', 'note': ''}, questions=[],
                  policy_acknowledged=True, stage5_accepted=True, evaluation_authorized=False)
    review['internal_duplicate_reviews'][0]['status'] = 'keep_original_and_disclose'
    review['pairs'][0].update(decision='not_same_source', reason='Different objects and scene')
    return template, review


def test_one_owner_needs_no_teammate_names_or_per_image_reasons():
    t, r = record_fixture()
    t['pairs'] = []; r['pairs'] = []
    assert validate_decisions(r, t) == []
    assert r['diagnostic_review']['status'] == 'not_viewed'
    assert 'member1' not in r and 'member5' not in r


@pytest.mark.parametrize('choice,exclude', [
    ('same_source', True), ('uncertain', True), ('not_same_source', False),
    ('different_capture_same_scene', False),
])
def test_single_owner_preserves_cross_candidate_handling(choice, exclude):
    t, r = record_fixture()
    r['pairs'][0]['decision'] = choice
    assert validate_decisions(r, t)[0]['exclude_from_supplementary'] == exclude


@pytest.mark.parametrize('change', [
    'owner', 'date', 'labels_pending', 'sample_swapped', 'binding', 'workflow',
    'internal_pending', 'internal_changed', 'diagnostic_question', 'policy',
    'acceptance', 'stage6', 'pair_missing', 'pair_changed', 'pair_pending',
])
def test_single_owner_still_rejects_missing_review_and_changed_evidence(change):
    t, r = record_fixture()
    if change == 'owner': r['reviewer'] = ' '
    elif change == 'date': r['reviewed_on'] = '2026-02-31'
    elif change == 'labels_pending': r['label_review']['status'] = 'pending'
    elif change == 'sample_swapped': r['label_samples'][0]['gsd_id'] = 'g2'
    elif change == 'binding': r['source_bindings']['full_manifest_sha256'] = 'new'
    elif change == 'workflow': r['workflow_sha256'] = 'new'
    elif change == 'internal_pending': r['internal_duplicate_reviews'][0]['status'] = 'needs_help'
    elif change == 'internal_changed': r['internal_duplicate_reviews'][0]['ids'] = ['g0', 'g2']
    elif change == 'diagnostic_question': r['diagnostic_review']['status'] = 'needs_help'
    elif change == 'policy': r['policy_acknowledged'] = False
    elif change == 'acceptance': r['stage5_accepted'] = False
    elif change == 'stage6': r['evaluation_authorized'] = True
    elif change == 'pair_missing': r['pairs'] = []
    elif change == 'pair_changed': r['pairs'][0]['trans10k_split'] = 'test'
    elif change == 'pair_pending': r['pairs'][0]['decision'] = 'pending'
    with pytest.raises(ValueError):
        validate_decisions(r, t)


def test_resolved_label_observation_is_recorded_without_changing_labels():
    t, r = record_fixture()
    r['label_review']['status'] = 'issues_recorded'
    r['questions'] = [{'kind': 'label', 'gsd_id': 'g0', 'observation': 'Sticker is foreground',
                      'status': 'unresolved', 'resolution': ''}]
    with pytest.raises(ValueError, match='Unresolved'):
        validate_decisions(r, t)
    r['questions'][0].update(status='resolved_keep_original', resolution='Keep source surface annotation and disclose scope')
    assert len(validate_decisions(r, t)) == 1
    r['questions'][0]['kind'] = 'cross_dataset'
    with pytest.raises(ValueError, match='new candidate audit'):
        validate_decisions(r, t)


def test_summary_cannot_hide_questions_or_claim_missing_question_details():
    t, r = record_fixture()
    r['label_review']['status'] = 'issues_recorded'
    with pytest.raises(ValueError, match='disagree'):
        validate_decisions(r, t)


def test_exact_match_cannot_be_relabeled_different_in_simple_review():
    t, r = record_fixture('exact_decoded_rgb')
    with pytest.raises(ValueError, match='exact decoded-image'):
        validate_decisions(r, t)
    r['pairs'][0]['decision'] = 'same_source'
    assert validate_decisions(r, t)[0]['exclude_from_supplementary']


def write_mini_audit(tmp_path):
    source = tmp_path / 'audit'; source.mkdir()
    data = tmp_path / 'data'; data.mkdir()
    manifest = tmp_path / 'full.json'
    rows = []
    for i in range(3):
        Image.new('RGB', (4, 3), 'white').save(data / f'{i}.jpg')
        Image.new('L', (4, 3), 255).save(data / f'{i}.png')
        rows.append({'id': f'g{i}', 'image': f'{i}.jpg', 'mask': f'{i}.png'})
    write_json(manifest, {'schema_version': 1, 'split': 'test', 'encoding': 'binary_0255',
                         'root': 'data', 'root_relative_to_manifest': True,
                         'complete_pinned_source_split': True, 'samples': rows})
    t, _ = record_fixture()
    for filename, obj in [('cross_candidates.json', t['pairs']),
                          ('policy_before_retrieval.json', {'keep_original': True}),
                          ('audit.json', {'test_fixture': True}),
                          ('retrieval_diagnostics.json', {'diagnostic_pairs': []}),
                          ('within_gsd_duplicates.json', [{'issue_id': 'duplicate1', 'ids': ['g0', 'g1']}]),
                          ('preview_index.json', {'labels': []})]:
        write_json(source / filename, obj)
    write_json(source / 'human_review_template.json', {
        'audit_binding_sha256': sha256_file(source / 'cross_candidates.json'),
        'policy_sha256': sha256_file(source / 'policy_before_retrieval.json'),
        'full_manifest_sha256': sha256_file(manifest),
        'labels': [{'gsd_id': 'g0', 'source_row': 0}], 'pairs': t['pairs']})
    template = make_template(source, manifest)
    _, review = record_fixture()
    for k in ('schema', 'workflow', 'workflow_sha256', 'source_bindings', 'full_count', 'label_samples'):
        review[k] = deepcopy(template[k])
    return source, manifest, template, review


def test_freeze_writes_new_manifests_preserves_full_and_never_authorizes_stage6(tmp_path):
    source, manifest, _, r = write_mini_audit(tmp_path)
    original = manifest.read_bytes()
    r['pairs'][0]['decision'] = 'uncertain'
    path = tmp_path / 'review.json'; write_json(path, r)
    output = tmp_path / 'release'
    freeze_review(path, output, source, manifest)
    full = json.loads((output / 'test_full.json').read_text())
    subset = json.loads((output / 'test_supplementary.json').read_text())
    release = json.loads((output / 'release.json').read_text())
    assert len(full['samples']) == 3 and [r['id'] for r in subset['samples']] == ['g1', 'g2']
    assert full['review_status'] == 'single_owner_spot_check_recorded'
    assert full['evaluation_authorized'] is False and subset['evaluation_authorized'] is False
    assert release['stage6_started'] is False and release['independent_team_signoffs_required'] is False
    assert manifest.read_bytes() == original
    with pytest.raises(FileExistsError): freeze_review(path, output, source, manifest)


def test_freeze_rechecks_current_audit_binding_before_writing(tmp_path):
    source, manifest, _, r = write_mini_audit(tmp_path)
    path = tmp_path / 'review.json'; write_json(path, r)
    write_json(source / 'audit.json', {'changed': True})
    output = tmp_path / 'release'
    with pytest.raises(ValueError, match='source_bindings'):
        freeze_review(path, output, source, manifest)
    assert not output.exists()
