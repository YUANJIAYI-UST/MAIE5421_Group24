"""Independently verify actual holdout masks, V1 targets, partitions and replay."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from transparent_seg.data import read_manifest, SegmentationDataset
from transparent_seg.data_v2 import read_category_mask
from transparent_seg.prepare import sha256_file, write_json


def verify(manifests,base,output,reproduced=None):
    manifests,base,output=Path(manifests),Path(base),Path(output)
    if output.exists():raise FileExistsError('Preserve existing verification evidence')
    audit=json.loads((manifests/'audit.json').read_text())
    held=set(audit['data_protocol']['held_out_class_ids'])
    loaded={name:read_manifest(manifests/f'{name}.json') for name in ('train','validation','test','test_heldout','test_seen')}
    original={s:read_manifest(base/f'{s}.json')[0] for s in ('train','validation','test')}
    labels={s:{x['id']:x for x in m['samples']} for s,m in original.items()}
    ids={name:{s['id'] for s in m['samples']} for name,(m,_) in loaded.items()}
    assert ids['test']==set(labels['test'])
    assert not ids['train']&ids['validation'] and not ids['train']&ids['test'] and not ids['validation']&ids['test']
    assert not ids['test_heldout']&ids['test_seen']
    quarantined={s['id'] for s in audit['test_category_quarantine']}
    assert ids['test_heldout']|ids['test_seen']|quarantined==ids['test']
    assert not quarantined&(ids['test_heldout']|ids['test_seen'])
    pixels={s:{x['pixel_sha256'] for x in loaded[s][0]['samples']} for s in ('train','validation','test')}
    assert not pixels['train']&pixels['validation'] and not pixels['train']&pixels['test'] and not pixels['validation']&pixels['test']
    category_results={}
    def inspect(item):
        name,sample,root=item
        split='test' if name.startswith('test') else name
        source=labels[split][sample['id']]
        assert all(sample[k]==v for k,v in source.items()),'V1 sample fields changed'
        for key in ('image','mask'):
            assert sha256_file(root/sample[key])==source[key+'_sha256'],'V1 source changed'
        assert sha256_file(root/sample['v2_category_mask'])==sample['v2_category_mask_sha256']
        if sample['v2_category_eligible']:
            fine=read_category_mask(root/sample['v2_category_mask'],loaded[name][0]['category_encoding'])
            counts=np.bincount(fine.ravel(),minlength=12).tolist()
            assert counts==sample['v2_category_pixel_counts']
            present={i for i,n in enumerate(counts) if i and n}
            assert present==set(sample['v2_category_ids'])
            if name in ('train','validation','test_seen'):assert not present&held
            if name=='test_heldout':assert present&held
            return name,sample['id'],counts
        assert name=='test' and sample['id'] in quarantined
        return name,sample['id'],None
    work=[(name,s,root) for name,(m,root) in loaded.items() for s in m['samples']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(name,sid,counts) in enumerate(pool.map(inspect,work),1):
            category_results[(name,sid)]=counts
            if n%2000==0:print(f'Independently checked manifest rows: {n}/{len(work)}',flush=True)
    holdout_pixels={s:{str(c):sum(category_results[(s,x)][c] for x in ids[s]) for c in sorted(held)} for s in ('train','validation')}
    assert all(n==0 for d in holdout_pixels.values() for n in d.values())
    # Exercise production data loading, including every retained corrected orientation.
    loader_checks=[]
    for split in ('train','validation'):
        ds=SegmentationDataset(manifests/f'{split}.json',512,training=split=='train')
        indices=sorted(set(range(min(3,len(ds))))|{i for i,x in enumerate(ds.samples) if x.get('mask_transform','none')!='none'})
        for i in indices:
            batch=ds[i]
            assert tuple(batch['image'].shape)==(3,512,512)
            if split=='train':assert tuple(batch['target'].shape)==(1,512,512)
            assert set(batch['target'].unique().tolist())<={0.0,1.0}
            loader_checks.append({'split':split,'id':batch['id'],'target_shape':list(batch['target'].shape)})
    byte_replay=None
    if reproduced:
        reproduced=Path(reproduced)
        originals={p.name:p for p in manifests.iterdir() if p.is_file()}
        assert set(originals)=={p.name for p in reproduced.iterdir() if p.is_file()}
        byte_replay={name:path.read_bytes()==(reproduced/name).read_bytes() for name,path in originals.items()}
        assert all(byte_replay.values())
    result={'status':'passed','counts':{n:len(v) for n,v in ids.items()},
            'v1_fields_and_current_binary_mask_hashes_preserved':True,
            'official_test_membership_unchanged':True,'test_subsets_disjoint_and_complete':True,
            'held_out_pixels_train_validation':holdout_pixels,'current_v2_masks_redecoded_per_manifest_rows':len(work),
            'production_loader_checks':loader_checks,'byte_identical_reproduction':byte_replay,
            'manifest_sha256':{n:sha256_file(manifests/f'{n}.json') for n in loaded},
            'human_review':loaded['train'][0]['data_protocol'].get('human_visual_review','pending'),'gpu_training':'not_run','formal_experiment':'not_run'}
    write_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('production_loader_checks','manifest_sha256')},indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifests',required=True);p.add_argument('--base',required=True)
    p.add_argument('--output',required=True);p.add_argument('--reproduced')
    a=p.parse_args();verify(a.manifests,a.base,a.output,a.reproduced)


if __name__=='__main__':main()
