import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from transparent_seg.data import read_manifest, SegmentationDataset, validate_derivative_use
from transparent_seg.near_duplicates import image_features, find_candidates
from transparent_seg.prepare import sha256_file, write_json
from transparent_seg.prepare_reviewed import prepare_reviewed


def fixture_policy(tmp_path):
    root = tmp_path / 'data' / 'source'
    rng = np.random.default_rng(4)
    images = [rng.integers(0, 256, (24, 32, 3), dtype=np.uint8) for _ in range(3)]
    specs = [('train', 'a', 0, False), ('train', 'b', 1, True), ('train', 'c', 2, False),
             ('validation', 'd', 0, False), ('validation', 'e', 2, False), ('test', 'f', 2, False)]
    rows = []
    for split, name, index, rotated in specs:
        image = root / split / 'images' / f'{name}.png'
        mask = root / split / 'masks' / f'{name}_mask.png'
        image.parent.mkdir(parents=True, exist_ok=True)
        mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(images[index]).save(image)
        label = np.zeros((24, 32, 3), np.uint8)
        label[2:15, 4:20] = [255, 0, 0]
        label[16:21, 19:26] = 255
        if rotated:
            label = np.rot90(label)
        Image.fromarray(label).save(mask)
        codes, counts = np.unique(label.astype(np.int16).sum(2), return_counts=True)
        rows.append({'id': image.relative_to(root).as_posix(), 'image': image.relative_to(root).as_posix(),
                     'mask': mask.relative_to(root).as_posix(), 'split': split, 'difficulty': split,
                     'image_sha256': sha256_file(image), 'mask_sha256': sha256_file(mask),
                     'pixel_sha256': hashlib.sha256(str(images[index].shape).encode() + images[index].tobytes()).hexdigest(),
                     'image_size': [32, 24], 'mask_size': [24, 32] if rotated else [32, 24],
                     'foreground_pixels': 243, 'valid_pixels': 768,
                     'mask_channel_sum_histogram': {str(int(k)): int(v) for k, v in zip(codes, counts)},
                     'issues': [{'type': 'shape_mismatch'}] if rotated else []})
    inventory = tmp_path / 'inventory.json'
    audit = tmp_path / 'source_audit.json'
    write_json(inventory, {'schema': 'trans10k_diagnostic_inventory_v1', 'not_for_training': True, 'samples': rows})
    write_json(audit, {'unpaired_masks': [], 'within_split_exact_duplicate_groups': [], 'empty_foreground_samples': []})
    policy = {'schema': 'trans10k_local_derivative_policy_v1', 'id': 'fixture_derivative',
              'corrected_masks_directory': 'corrected', 'source': {'official_counts': {'train': 3, 'validation': 2, 'test': 1},
              'inventory_sha256': sha256_file(inventory), 'audit_sha256': sha256_file(audit)},
              'mask_corrections': [{'id': rows[1]['id'], 'operation': 'rotate_90_cw', 'source_mask_sha256': rows[1]['mask_sha256']}],
              'exclusions': [{'id': rows[i]['id'], 'retained_id': rows[j]['id'], 'reason_code': 'cross_split_exact_rgb_duplicate'}
                             for i,j in ((0,3),(2,5),(4,5))]}
    policy_path = tmp_path / 'policy.json'
    write_json(policy_path, policy)
    return root, inventory, audit, policy_path, policy, rows


def build(tmp_path, name='out', fixture=None):
    f = fixture or fixture_policy(tmp_path)
    root, inventory, audit, policy_path, _, _ = f
    output = tmp_path / name
    result = prepare_reviewed(root, output, policy_path, inventory, audit)
    return output, result, f


def test_derivative_rotates_copy_preserves_labels_test_membership_and_reproduces(tmp_path):
    f = fixture_policy(tmp_path)
    root, _, _, _, _, rows = f
    before = {r[k]: sha256_file(root / r[k]) for r in rows for k in ('image', 'mask')}
    output, audit, _ = build(tmp_path, fixture=f)
    assert [audit['splits'][s]['effective_count'] for s in ('train','validation','test')] == [1,1,1]
    assert {k: sha256_file(root / k) for k in before} == before
    training, base = read_manifest(output / 'train.json')
    with Image.open(base / training['samples'][0]['mask']) as im:
        repaired = np.array(im)
    with Image.open(root / rows[1]['mask']) as im:
        original = np.array(im)
    np.testing.assert_array_equal(repaired, np.rot90(original, -1))
    assert not training['complete_official_split']
    test, _ = read_manifest(output / 'test.json')
    assert test['complete_official_split'] and test['samples'][0]['image_sha256'] == rows[-1]['image_sha256']
    assert test['samples'][0]['mask_sha256'] == rows[-1]['mask_sha256']
    assert SegmentationDataset(output / 'train.json', 32, training=True)[0]['target'].shape == (1,32,32)
    other, _, _ = build(tmp_path, name='other', fixture=f)
    for name in ('train.json','validation.json','test.json','audit.json','release.json','policy.json'):
        assert (output/name).read_bytes() == (other/name).read_bytes()


@pytest.mark.parametrize('mutation', ['source_bytes', 'remove_exclusion', 'test_exclusion', 'missing_rotation', 'unsafe_path'])
def test_derivative_fails_closed_before_issuing_manifests(tmp_path, mutation):
    f = fixture_policy(tmp_path)
    root, inventory, source_audit, policy_path, policy, rows = f
    if mutation == 'source_bytes':
        (root / rows[0]['image']).write_bytes(b'tampered')
    elif mutation == 'remove_exclusion':
        policy['exclusions'].pop(0)
    elif mutation == 'test_exclusion':
        policy['exclusions'].append({'id':rows[-1]['id'], 'retained_id':rows[2]['id'], 'reason_code':'cross_split_exact_rgb_duplicate'})
    elif mutation == 'missing_rotation':
        policy['mask_corrections'] = []
    else:
        policy['corrected_masks_directory'] = '../outside'
    write_json(policy_path, policy)
    with pytest.raises(ValueError):
        prepare_reviewed(root, tmp_path/'out', policy_path, inventory, source_audit)
    assert not (tmp_path/'out/train.json').exists()


def test_derived_manifest_is_portable_but_cannot_escape_its_data_root(tmp_path, monkeypatch):
    output, _, _ = build(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    meta, base = read_manifest(output/'train.json')
    assert base == tmp_path/'data'
    meta['samples'][0]['image'] = '../../outside.png'
    write_json(output/'train.json', meta)
    with pytest.raises(ValueError, match='outside-root'):
        read_manifest(output/'train.json')


def test_derivative_training_requires_explicit_pinned_pilot_and_rejects_research(tmp_path):
    output, _, _ = build(tmp_path)
    train,_ = read_manifest(output/'train.json')
    val,_ = read_manifest(output/'validation.json')
    config={'purpose':'pilot','train_manifest':str(output/'train.json'),'val_manifest':str(output/'validation.json'),
            'data_protocol_id': 'fixture_derivative',
            'expected_manifest_sha256':{s:sha256_file(output/f'{s}.json') for s in ('train','validation')}}
    validate_derivative_use(config, train, val)
    for change in ({'purpose':'research'}, {'data_protocol_id':'different'}, {'expected_manifest_sha256':{}}):
        with pytest.raises(ValueError):
            validate_derivative_use({**config,**change},train,val)
    modified = copy.deepcopy(val)
    modified['data_protocol']['policy_sha256']='changed'
    with pytest.raises(ValueError,match='protocols differ'):
        validate_derivative_use(config,train,modified)
    plain_train,plain_val=copy.deepcopy(train),copy.deepcopy(val)
    plain_train.pop('data_protocol');plain_val.pop('data_protocol')
    with pytest.raises(ValueError,match='requires data_protocol'):
        validate_derivative_use(config,plain_train,plain_val)


def textured_image():
    rng = np.random.default_rng(80)
    coarse = rng.integers(0,256,(20,24,3),dtype=np.uint8)
    return Image.fromarray(coarse).resize((480,400),Image.Resampling.BICUBIC)


def candidate_rows(n):
    return [{'id':f'{"train" if i==0 else "test"}/images/{i}.jpg', 'split':'train' if i==0 else 'test',
             'image_sha256':str(i), 'pixel_sha256':str(i)} for i in range(n)]


def test_perceptual_retrieval_finds_resize_compression_and_center_crop_without_auto_exclusion(tmp_path):
    image = textured_image()
    image.resize((240,200),Image.Resampling.LANCZOS).save(tmp_path/'reencoded.jpg',quality=75)
    with Image.open(tmp_path/'reencoded.jpg') as im:
        reencoded=im.copy()
    cropped=image.crop((48,40,432,360))
    features=[image_features(im) for im in (image,reencoded,cropped)]
    candidates=find_candidates(features,candidate_rows(3),block_size=1)
    ids={tuple(c['ids']) for c in candidates}
    assert ('train/images/0.jpg','test/images/1.jpg') in ids
    assert ('train/images/0.jpg','test/images/2.jpg') in ids
    assert all(c['decision']=='pending' and not c['automatic_exclusion'] for c in candidates)
    assert candidates == find_candidates(features,candidate_rows(3),block_size=3)


def test_perceptual_retrieval_separates_exact_groups_and_unrelated_images():
    first=textured_image()
    other=Image.fromarray(np.random.default_rng(222).integers(0,256,(400,480,3),np.uint8))
    rows=candidate_rows(3)
    rows[1]['pixel_sha256']=rows[0]['pixel_sha256']
    assert find_candidates([image_features(im) for im in (first,first,other)],rows)==[]
