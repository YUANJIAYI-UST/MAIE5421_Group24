import numpy as np
import copy
import pytest

from scripts.member4_gsd_prepare import cross_candidates, exact_cross, subset_from_candidates
from scripts.member4_gsd_review import validate_decisions


def row(identity, pixel, byte=None, split='test'):
    return {'id': identity, 'source_id': identity, 'pixel_sha256': pixel,
            'image_sha256': byte or pixel, 'split': split}


def test_rectangular_retrieval_matches_scalar_oracle_and_all_pairs():
    rng = np.random.default_rng(123)
    g = rng.integers(0, 2**63, size=(3, 2, 2), dtype=np.uint64)
    t = rng.integers(0, 2**63, size=(7, 2, 2), dtype=np.uint64)
    t[0, 0] = g[0, 0]
    t[1, 1] = g[1, 0]
    t[2, 0] = g[2, 1]
    t[3, 0] = g[0, 0] ^ np.array([(1 << 8) - 1, (1 << 12) - 1], dtype=np.uint64)
    t[4, 0] = g[0, 0] ^ np.array([(1 << 9) - 1, 0], dtype=np.uint64)
    t[5] = t[0]  # Both candidates must survive; retrieval is not top-1.
    gr = [row('g'+str(i), 'g'+str(i)) for i in range(3)]
    tr = [row('t'+str(i), 't'+str(i), split='train') for i in range(7)]
    tr[6]['pixel_sha256'] = gr[0]['pixel_sha256']; t[6] = g[0]
    expected = set()
    for i in range(3):
        for j in range(7):
            if gr[i]['pixel_sha256'] == tr[j]['pixel_sha256']:
                continue
            if any((int(g[i, lv, 0]) ^ int(t[j, rv, 0])).bit_count() <= threshold
                   and (int(g[i, lv, 1]) ^ int(t[j, rv, 1])).bit_count() <= 12
                   for lv, rv, threshold in ((0, 0, 8), (0, 1, 6), (1, 0, 6))):
                expected.add((gr[i]['id'], tr[j]['id']))
    result = cross_candidates(g, t, gr, tr, block_size=2)
    assert {(r['gsd_id'], r['trans10k_id']) for r in result} == expected
    assert ('g0','t3') in expected and ('g0','t4') not in expected
    assert ('g0','t0') in expected and ('g0','t5') in expected
    assert all(not r['automatic_same_source_decision'] for r in result)
    assert len({r['pair_id'] for r in result}) == len(result)


def test_exact_cross_retains_all_matches_and_separates_bytes_from_pixels():
    g = [row('g1', 'p', 'a'), row('g2', 'q', 'b')]
    t = [row('t1', 'p', 'a', 'train'), row('t2', 'p', 'other', 'validation')]
    result = exact_cross(g, t)
    assert [r['kind'] for r in result] == ['exact_bytes', 'exact_decoded_rgb']
    assert {r['trans10k_split'] for r in result} == {'train','validation'}


def test_provisional_subset_excludes_unresolved_without_changing_full_membership():
    full = {'samples': [row('g1','p'),row('g2','p'),row('g3','q')], 'effective_count':3}
    candidates = [{'gsd_id':'g1','trans10k_id':'t','trans10k_split':'test',
                   'pair_id':'pair1','kind':'perceptual_candidate'}]
    subset, exclusions = subset_from_candidates(full, candidates)
    assert [r['id'] for r in subset['samples']] == ['g2','g3']
    assert len(full['samples']) == 3
    assert subset['evaluation_authorized'] is False
    assert exclusions[0]['reasons'][0]['reason'] == 'unresolved_perceptual_candidate'


def review_fixture(kind='perceptual_candidate'):
    template={'schema':1, 'audit_binding_sha256':'audit','policy_sha256':'policy','full_manifest_sha256':'full',
        'allowed_decisions':['same_source','not_same_source','different_capture_same_scene','uncertain'],
        'pairs':[{'pair_id':'p','gsd_id':'g','trans10k_id':'t','trans10k_split':'train','kind':kind,
                  'member4':{},'member1':{}}], 'labels':[{'gsd_id':'g'}], 'internal_duplicate_issues':[{'issue_id':'i'}]}
    r=copy.deepcopy(template)
    r['pairs'][0]['member4']={'reviewer':'Person4','decision':'not_same_source','reason':'different arrangement'}
    r['pairs'][0]['member1']={'reviewer':'Person1','decision':'not_same_source','reason':'different viewpoint and objects'}
    r['labels'][0].update(status='pass',reviewer='Person4',reason='alignment and polarity inspected')
    r['internal_duplicate_issues'][0].update(status='acknowledged_keep',reviewer='Person1',reason='retain both')
    r['member5']={'reviewer':'Person5','reason':'reviewed fixed policy','policy_approved':True,
        'target_semantics_acknowledged':True,'limitations_acknowledged':True,'internal_duplicate_policy_approved':True}
    r['user_stage5_accepted']=True
    r['cross_audit_review']={role:{'reviewer':person,'status':'reviewed','reason':'retrieval and diagnostics inspected'} for role,person in [('member4','Person4'),('member1','Person1')]}
    return template,r


def test_human_review_unresolved_excluded_same_scene_separately_recorded():
    t,r=review_fixture()
    assert not validate_decisions(r,t)[0]['exclude_from_supplementary']
    r['pairs'][0]['member4']['decision']='uncertain'
    assert validate_decisions(r,t)[0]['reason']=='unresolved_or_disagreement'
    for role in ('member4','member1'):
        r['pairs'][0][role]['decision']='different_capture_same_scene'
    assert validate_decisions(r,t)[0]['reason']=='different_capture_same_scene'
    assert not validate_decisions(r,t)[0]['exclude_from_supplementary']


@pytest.mark.parametrize('change', ['pending','same_reviewer','policy','acceptance','tamper','missing','cross_question'])
def test_review_cannot_freeze_unapproved_or_modified_evidence(change):
    t,r=review_fixture()
    if change=='pending': r['pairs'][0]['member1']['decision']='pending'
    elif change=='same_reviewer': r['pairs'][0]['member1']['reviewer']='Person4'
    elif change=='policy': r['member5']['policy_approved']=False
    elif change=='acceptance': r['user_stage5_accepted']=False
    elif change=='tamper': r['pairs'][0]['trans10k_split']='test'
    elif change=='missing': r['labels']=[]
    elif change=='cross_question': r['cross_audit_review']['member1']['status']='question'
    with pytest.raises(ValueError): validate_decisions(r,t)


def test_exact_image_evidence_cannot_be_relabelled_as_different_source():
    t,r=review_fixture('exact_decoded_rgb')
    with pytest.raises(ValueError,match='exact decoded-image'):
        validate_decisions(r,t)
    for role in ('member4','member1'): r['pairs'][0][role]['decision']='same_source'
    assert validate_decisions(r,t)[0]['exclude_from_supplementary']
