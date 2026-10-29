import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from transparent_seg.data import decode_mask, read_manifest, SegmentationDataset, validate_derivative_use
from transparent_seg.data_v2 import CLASS_NAMES, PALETTE, category_array, read_category_mask
from transparent_seg.prepare import sha256_file, write_json
from transparent_seg.prepare_holdout import select_holdout, build_holdout


def fixture_holdout(tmp_path):
    data = tmp_path/'data'
    base = tmp_path/'manifests'/'base'
    base.mkdir(parents=True)
    rows=[]
    rng=np.random.default_rng(42)
    # Classes 1/2 are most supported. Include a mixed (1,3) image and
    # unknown label image: neither may remain in training/validation.
    specs=[(1,), (1,), (2,), (2,), (2,), (1,3), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), ('unknown',)]
    for split in ('train','validation','test'):
        samples=[]
        for n,classes in enumerate(specs):
            sid=f'{split}/images/{n}.png'
            image=data/'v1'/sid
            binary=data/'v1'/split/'masks'/f'{n}_mask.png'
            fine=data/'v2'/split/'masks_12'/f'{n}_mask.png'
            v2image=data/'v2'/sid
            for path in (image,binary,fine,v2image):path.parent.mkdir(parents=True,exist_ok=True)
            pixels=rng.integers(0,256,(24,32,3),np.uint8)
            Image.fromarray(pixels).save(image);v2image.write_bytes(image.read_bytes())
            # V1 foreground deliberately differs from V2. Filtering must not rewrite it.
            v1=np.zeros((24,32,3),np.uint8);v1[1:5,1:7]=255;v1[5:10,7:15]=[255,0,0]
            Image.fromarray(v1).save(binary)
            rgb=np.zeros((24,32,3),np.uint8)
            if classes==('unknown',):rgb[2:8,2:10]=[255,0,0]
            else:
                for offset,c in enumerate(classes):rgb[2+offset*8:8+offset*8,2:10]=PALETTE[c]
            Image.fromarray(rgb).save(fine)
            with Image.open(fine) as im:ids,unknown=category_array(im,'author_rgb_palette')
            counts=np.bincount(ids[ids>=0],minlength=12).tolist()
            phash=hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
            samples.append({'id':sid,'image':image.relative_to(data).as_posix(),'mask':binary.relative_to(data).as_posix(),
                            'image_sha256':sha256_file(image),'mask_sha256':sha256_file(binary),'pixel_sha256':phash,
                            'difficulty':split,'width':32,'height':24,'foreground_pixels':64,'valid_pixels':768})
            rows.append({'id':sid,'split':split,'v1_image_sha256':sha256_file(image),'v2_image':sid,
                         'v2_image_sha256':sha256_file(v2image),'v2_mask':fine.relative_to(data/'v2').as_posix(),
                         'v2_mask_sha256':sha256_file(fine),'v2_mask_size':[32,24],'v2_to_v1_orientation':'none',
                         'category_pixel_counts':counts,'category_ids':[i for i,x in enumerate(counts) if i and x],
                         'category_filter_eligible':not unknown,'unknown_codes':unknown,
                         'issues':['unknown_v2_category_code'] if unknown else []})
        write_json(base/f'{split}.json',{'schema_version':1,'root':'../../data','root_relative_to_manifest':True,
                   'dataset':'fixture_base','split':split,'encoding':'trans10k_v1_rgb','void_values':[],
                   'synthetic':False,'official_count':len(samples),'complete_official_split':split=='test',
                   'samples':samples,'mask_channel_sum_histogram':{'obsolete':100}})
    rules=json.loads((Path(__file__).parents[1]/'configs/data/trans10k_d03_rules_20261028.json').read_text())
    rules['selection'].update({'minimum_class_train_images':1,'minimum_class_validation_images':1,
                               'minimum_retained_fraction_each_split':0.5})
    rulefile=tmp_path/'rules.json';write_json(rulefile,rules)
    common={'schema':'trans10k_v2_alignment_inventory_v1','class_names':list(CLASS_NAMES),
            'rules_sha256':sha256_file(rulefile),'mirror_revision':rules['mirror_revision'],
            'v2_root_name':'v2','v1_inventory_sha256':'fixture_source'}
    tv=tmp_path/'train_val.json';write_json(tv,{**common,'stage':'train_validation_only',
                                            'samples':[r for r in rows if r['split']!='test']})
    test=tmp_path/'test.json';write_json(test,{**common,'stage':'test_after_selection_freeze',
                                           'samples':[r for r in rows if r['split']=='test']})
    return base,tv,test,rulefile,data


def selected_fixture(tmp_path):
    base,tv,test,rules,data=fixture_holdout(tmp_path)
    folder=tmp_path/'selection'
    select_holdout(base,tv,rules,folder)
    selection=folder/'selection.json'
    ti=json.loads(test.read_text());ti['selection_sha256_before_test_audit']=sha256_file(selection);write_json(test,ti)
    return base,tv,test,rules,data,selection


def test_v2_decodes_author_ids_and_exact_palette_without_guessing():
    indexed=Image.fromarray(np.arange(12,dtype=np.uint8)[None,:])
    assert read_category_mask_from_image(indexed)==list(range(12))
    rgb=Image.fromarray(np.array(PALETTE,np.uint8)[None,:,:])
    ids,unknown=category_array(rgb,'author_rgb_palette')
    assert ids.tolist()==[list(range(12))] and not unknown
    red=Image.fromarray(np.array([[[255,0,0]]],np.uint8))
    ids,unknown=category_array(red,'author_rgb_palette')
    assert ids.item()==-1 and unknown=={'#ff0000':1}
    rgba=Image.fromarray(np.array([[[255,51,7,0]]],np.uint8))
    assert category_array(rgba,'author_rgb_palette')[0].item()==-1
    with pytest.raises(ValueError):category_array(rgb,'author_index_0_11')


def read_category_mask_from_image(im):
    return category_array(im,'author_index_0_11')[0][0].tolist()


def test_selection_rejects_test_evidence_and_is_independent_of_test_counts(tmp_path):
    base,tv,test,rules,data=fixture_holdout(tmp_path)
    one=select_holdout(base,tv,rules,tmp_path/'s1')
    assert one['held_out_class_ids']==[1,2]
    # No test files exist for the second selection: it still succeeds identically.
    test.unlink();(base/'test.json').unlink()
    two=select_holdout(base,tv,rules,tmp_path/'s2')
    assert one==two
    bad=json.loads(tv.read_text());bad['samples'][0]['split']='test';write_json(tv,bad)
    with pytest.raises(ValueError,match='splits'):
        select_holdout(base,tv,rules,tmp_path/'bad')


def test_holdout_excludes_mixed_images_keeps_v1_labels_and_full_test_and_reproduces(tmp_path):
    base,tv,test,rules,data,selection=selected_fixture(tmp_path)
    original={p:sha256_file(p) for p in data.rglob('*') if p.is_file()}
    out=tmp_path/'manifests'/'holdout'
    audit=build_holdout(base,tv,test,selection,rules,out)
    assert audit['v1_binary_targets_preserved'] and audit['official_test_membership_preserved']
    for split in ('train','validation'):
        meta,root=read_manifest(out/f'{split}.json')
        assert len(meta['samples'])==9
        assert all(not set(s['v2_category_ids'])&{1,2} for s in meta['samples'])
        assert f'{split}/images/5.png' not in {s['id'] for s in meta['samples']}
        assert 'mask_channel_sum_histogram' not in meta
        for s in meta['samples']:
            expected={x['id']:x for x in json.loads((base/f'{split}.json').read_text())['samples']}[s['id']]
            assert all(s[k]==v for k,v in expected.items())
            y,_=decode_mask(root/s['mask'])
            fine=read_category_mask(root/s['v2_category_mask'],'author_rgb_palette')
            assert not np.array_equal(y,fine>0)
    full,_=read_manifest(out/'test.json')
    subset,_=read_manifest(out/'test_heldout.json')
    seen,_=read_manifest(out/'test_seen.json')
    assert len(full['samples'])==16 and len(subset['samples'])==6 and len(seen['samples'])==9
    assert full['complete_official_split'] and not subset['complete_official_split']
    assert len(audit['test_category_quarantine'])==1
    assert {p:sha256_file(p) for p in original}==original
    assert SegmentationDataset(out/'train.json',32,training=True)[0]['target'].shape==(1,32,32)
    other=tmp_path/'manifests'/'holdout_again';build_holdout(base,tv,test,selection,rules,other)
    assert {p.name:p.read_bytes() for p in out.iterdir()}=={p.name:p.read_bytes() for p in other.iterdir()}
    config={'purpose':'pilot','data_protocol_id':full['data_protocol']['id'],
            'train_manifest':str(out/'train.json'),'val_manifest':str(out/'validation.json'),
            'expected_manifest_sha256':audit['manifest_sha256']}
    tr,_=read_manifest(out/'train.json');va,_=read_manifest(out/'validation.json')
    validate_derivative_use(config,tr,va)
    leaked=copy.deepcopy(tr);leaked['samples'][0]['v2_category_ids'].append(1)
    with pytest.raises(ValueError,match='Held-out'):
        validate_derivative_use(config,leaked,va)
    unsafe=copy.deepcopy(tr);unsafe['samples'][0]['v2_category_mask']='../../outside.png'
    write_json(out/'unsafe.json',unsafe)
    with pytest.raises(ValueError,match='outside-root'):
        read_manifest(out/'unsafe.json')
    with pytest.raises(ValueError,match='pilot-only'):
        validate_derivative_use({**config,'purpose':'research'},tr,va)


@pytest.mark.parametrize('mutation',['v2_pixels','v1_pixels','inventory_counts','selection','test_freeze','base_manifest','partial_test'])
def test_holdout_rejects_changed_sources_and_protocol_before_publishing(tmp_path,mutation):
    base,tv,test,rules,data,selection=selected_fixture(tmp_path)
    if mutation=='v2_pixels':
        p=data/'v2/train/masks_12/5_mask.png';p.write_bytes(b'corrupt')
    elif mutation=='v1_pixels':
        p=data/'v1/train/masks/5_mask.png';p.write_bytes(b'corrupt')
    elif mutation=='inventory_counts':
        x=json.loads(tv.read_text());x['samples'][0]['category_pixel_counts'][1]=0;write_json(tv,x)
    elif mutation=='selection':
        x=json.loads(selection.read_text());x['held_out_class_ids']=[2,3];write_json(selection,x)
    elif mutation=='test_freeze':
        x=json.loads(test.read_text());x['selection_sha256_before_test_audit']='wrong';write_json(test,x)
    elif mutation=='base_manifest':
        p=base/'train.json';x=json.loads(p.read_text());x['samples'].pop();write_json(p,x)
    else:
        p=base/'test.json';x=json.loads(p.read_text());x['samples'].pop();write_json(p,x)
    out=tmp_path/'manifests'/'bad'
    with pytest.raises((ValueError,OSError)):
        build_holdout(base,tv,test,selection,rules,out)
    assert not out.exists()
