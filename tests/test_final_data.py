"""Regression checks for owner decisions, transitive leakage and preservation."""
import pytest
from transparent_seg.prepare_final import duplicate_components, exclusion_plan


def row(sid, split, pixel=None):
    return {'id': sid, 'split': split, 'pixel_sha256': pixel or sid, 'image_sha256': 'sha-' + sid}


def pair(number, a, b, answer='A'):
    return {'number': number, 'key': str(number), 'ids': [a, b],
            'source_image_sha256': ['sha-' + a, 'sha-' + b], 'owner_answer': answer}


def test_transitive_exact_and_human_links_keep_all_test():
    rows = [row('a', 'train', 'same'), row('b', 'validation', 'same'),
            row('c', 'test'), row('d', 'test')]
    # b can already be removed in an earlier version; it must still bridge a-c.
    decisions = [pair(1, 'b', 'c'), pair(2, 'c', 'd')]
    excluded, groups = exclusion_plan(rows, decisions, [])
    assert {r['id'] for r in excluded} == {'a', 'b'}
    assert groups[0]['retained_highest_split_ids'] == ['c', 'd']
    assert excluded[0]['human_A_pair_numbers'] == [1, 2]


@pytest.mark.parametrize('answer', ['B', 'C', 'D'])
def test_other_answers_never_imply_same_photo(answer):
    rows = [row('a', 'train'), row('b', 'test')]
    assert duplicate_components(rows, [pair(1, 'a', 'b', answer)]) == []


def test_same_split_duplicates_are_preserved_and_coverage_is_separate():
    rows = [row('a', 'train'), row('b', 'train')]
    excluded, groups = exclusion_plan(rows, [pair(31, 'a', 'b')], ['b'])
    assert len(groups) == 1
    assert excluded == [{'id': 'b', 'split': 'train', 'reason_codes': ['owner_flagged_annotation_coverage']}]


def test_owner_A_is_used_even_when_ai_disagrees():
    decision = {**pair(31, 'a', 'b'), 'ai_recommendation': 'B'}
    excluded, _ = exclusion_plan([row('a', 'train'), row('b', 'test')], [decision], [])
    assert [r['id'] for r in excluded] == ['a']


def test_tampered_review_hash_is_rejected():
    decision = pair(1, 'a', 'b')
    decision['source_image_sha256'][0] = 'changed'
    with pytest.raises(ValueError, match='identity mismatch'):
        duplicate_components([row('a', 'train'), row('b', 'test')], [decision])


def test_test_quarantine_is_rejected():
    with pytest.raises(ValueError, match='training image'):
        exclusion_plan([row('a', 'test')], [], ['a'])


def test_duplicate_or_blank_review_answer_is_rejected():
    rows = [row('a', 'train'), row('b', 'test')]
    with pytest.raises(ValueError):
        duplicate_components(rows, [pair(1, 'a', 'b'), pair(1, 'a', 'b')])
    with pytest.raises(ValueError):
        duplicate_components(rows, [pair(1, 'a', 'b', '')])
