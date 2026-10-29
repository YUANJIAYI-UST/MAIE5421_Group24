"""Verify a derivative's file hashes, source integrity, isolation and real-data loading."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.data import SegmentationDataset, read_manifest, validate_derivative_use
from transparent_seg.losses import segmentation_loss
from transparent_seg.model import TransparentSegformer
from transparent_seg.prepare import sha256_file, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',required=True);p.add_argument('--source-root',default='data/Trans10K')
    p.add_argument('--inventory',required=True);p.add_argument('--reproduced-manifests')
    p.add_argument('--output',required=True);p.add_argument('--cpu-step',action='store_true')
    a=p.parse_args();folder=Path(a.manifests);source=Path(a.source_root).resolve();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    release=json.loads((folder/'release.json').read_text());inventory=json.loads(Path(a.inventory).read_text())
    originals={r['id']:r for r in inventory['samples']}
    original_hashes={source/r[k]:r[k+'_sha256'] for r in originals.values() for k in ('image','mask')}
    manifests={};roots={};effective_hashes={}
    for split in ('train','validation','test'):
        path=folder/f'{split}.json'
        assert sha256_file(path)==release['manifest_sha256'][split]
        meta,root=read_manifest(path);manifests[split]=meta;roots[split]=root
        assert meta['split']==split and meta['effective_count']==len(meta['samples'])
        assert meta['encoding']=='trans10k_v1_rgb' and meta['void_values']==[]
        for row in meta['samples']:
            assert row['id'] in originals and originals[row['id']]['split']==split
            assert row['image_sha256']==originals[row['id']]['image_sha256']
            for key in ('image','mask'):
                effective_hashes[root/row[key]]=row[key+'_sha256']
            with Image.open(root/row['mask']) as im:assert im.size==(row['width'],row['height'])
    paths={**original_hashes,**effective_hashes}
    def verify(task):
        path,expected=task
        if sha256_file(path)!=expected:raise ValueError(f'File hash mismatch: {path}')
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(verify,paths.items()))
    hashes={s:{r['pixel_sha256'] for r in m['samples']} for s,m in manifests.items()}
    assert not hashes['train']&hashes['validation']
    assert not hashes['train']&hashes['test']
    assert not hashes['validation']&hashes['test']
    test_ids={r['id'] for r in manifests['test']['samples']}
    assert test_ids=={r['id'] for r in originals.values() if r['split']=='test'}
    for r in manifests['test']['samples']:assert r['mask_sha256']==originals[r['id']]['mask_sha256']
    reproduced=None
    if a.reproduced_manifests:
        names={p.name for p in folder.iterdir() if p.is_file()}
        assert names=={p.name for p in Path(a.reproduced_manifests).iterdir() if p.is_file()}
        for name in sorted(names):
            assert (folder/name).read_bytes()==(Path(a.reproduced_manifests)/name).read_bytes()
        reproduced=True
    config={'purpose':'pilot','train_manifest':str(folder/'train.json'),'val_manifest':str(folder/'validation.json'),
            'data_protocol_id':release['protocol_id'],'expected_manifest_sha256':release['manifest_sha256']}
    validate_derivative_use(config,manifests['train'],manifests['validation'])
    dataset=SegmentationDataset(folder/'train.json',512,True)
    fixed=[i for i,r in enumerate(dataset.samples) if r['mask_transform']!='none']
    indices=sorted(set(fixed+[0,len(dataset)//2,len(dataset)-1]))
    for i in indices:
        sample=dataset[i]
        assert sample['image'].shape==(3,512,512) and sample['target'].shape==(1,512,512)
        assert torch.isfinite(sample['image']).all() and sample['valid'].all()
        assert set(sample['target'].unique().tolist()) <= {0.0,1.0}
    validation=SegmentationDataset(folder/'validation.json',512,False)
    for i in (0,len(validation)//2,len(validation)-1):
        sample=validation[i];row=validation.samples[i]
        assert sample['target'].shape==(1,row['height'],row['width'])
    cpu_step=None
    if a.cpu_step:
        torch.set_num_threads(4);torch.manual_seed(42)
        batch=dataset[fixed[0] if fixed else 0];model=TransparentSegformer(boundary=True,pretrained=False).train()
        started=time.monotonic()
        loss=segmentation_loss(model(batch['image'][None],return_boundary=True),batch['target'][None],batch['valid'][None],0.1,1)
        assert torch.isfinite(loss['total'])
        loss['total'].backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum()>0 for g in gradients)
        cpu_step={'input_size':512,'batch_size':1,'device':'cpu','sample_id':batch['id'],
                  'loss':float(loss['total'].detach()),'finite_gradients':True,'seconds':time.monotonic()-started,
                  'initialization':'random; software diagnostic only; not a pretrained GPU pilot or benchmark'}
    result={'source_files_unchanged':True,'source_files_verified':len(original_hashes),
            'all_distinct_source_and_effective_files_verified':len(paths),'official_test_membership_and_gt_unchanged':True,
            'cross_split_exact_rgb_overlaps':0,'effective_counts':{s:len(m['samples']) for s,m in manifests.items()},
            'corrected_samples_loaded_at_512':len(fixed),'train_samples_loaded':len(indices),'validation_samples_loaded':3,
            'byte_identical_regeneration':reproduced,'manifest_sha256':release['manifest_sha256'],
            'cpu_diagnostic':cpu_step,'human_review':manifests['train']['data_protocol'].get('human_visual_review','pending'),'member5_review':'pending',
            'environment':{'python':platform.python_version(),'macOS':platform.mac_ver()[0],'torch':str(torch.__version__)}}
    write_json(output,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
