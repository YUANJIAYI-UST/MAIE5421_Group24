"""Independent arithmetic/membership review; does not rerun E03 inference."""
import csv
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transparent_seg.prepare import sha256_file
from scripts.summarize_member4_completion import ROOT, read, save, require

SOURCE=Path('tmp/member4_completion_20261118/member2_source/reports/group24_b03_b04_e03_20261118')
MANIFEST=Path('manifests/member1_final_20261029/d03')


def main():
    paths={name:MANIFEST/(filename+'.json') for name,filename in
           [('train','train'),('validation','validation'),('test','test'),
            ('test_seen','test_seen'),('test_heldout','test_heldout')]}
    manifests={k:read(p) for k,p in paths.items()}
    test={r['id']:r for r in manifests['test']['samples']}
    members={g:{s['id'] for s in manifests['test_'+g]['samples']} for g in ['seen','heldout']}
    require(not members['seen'] & members['heldout'] and members['seen'] | members['heldout']==set(test),
            'E03 test groups do not form a disjoint partition')
    require([len(members[g]) for g in ['seen','heldout']]==[2935,1493],'Unexpected E03 groups')
    for split,count in [('train',3230),('validation',646)]:
        samples=manifests[split]['samples']
        require(len(samples)==count and all(s['v2_category_pixel_counts'][5]==s['v2_category_pixel_counts'][8]==0
                                           for s in samples),'Held-out category present in training/validation manifest')
    with (SOURCE/'E03_category_recall.csv').open(encoding='utf-8-sig',newline='') as f:
        table={(r['model'],r['subset'],int(r['category_id'])):r for r in csv.DictReader(f)}
    checks=[]
    for model in ['A','D']:
        result=read(SOURCE/f'E03_{model}/result.json')
        require(result['complete'],'Incomplete E03 result')
        require(result['identity']['manifests']=={k:sha256_file(p) for k,p in paths.items()},'E03 manifest hash differs')
        with (SOURCE/f'E03_{model}/per_image.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        require(len(rows)==len({r['id'] for r in rows})==4428 and {r['id'] for r in rows}==set(test),
                'Missing or duplicate E03 predictions')
        for row in rows:
            sample=test[row['id']]
            require(row['id'] in members[row['subset']],'Incorrect E03 group')
            row['hits']=json.loads(row['hits']);row['total']=json.loads(row['total'])
            require(row['total']==sample['v2_category_pixel_counts'],'V2 class denominator differs from frozen manifest')
            require(all(0<=h<=t for h,t in zip(row['hits'],row['total'])),'Impossible class recall count')
            require(int(row['tp'])+int(row['fn'])==sample['foreground_pixels'],'V1 foreground differs')
            require(sum(int(row[k]) for k in ['tp','fp','fn','tn'])==sample['valid_pixels'],'V1 valid pixels differ')
        for group in ['all','seen','heldout']:
            chosen=[r for r in rows if group=='all' or r['subset']==group]
            tp,fp,fn,tn=[sum(int(r[k]) for r in chosen) for k in ['tp','fp','fn','tn']]
            recomputed={'foreground_iou':tp/(tp+fp+fn),'region_f1':2*tp/(2*tp+fp+fn),
                'boundary_iou':math.fsum(float(r['boundary_iou']) for r in chosen)/len(chosen),
                'background_fpr':fp/(fp+tn)}
            score=result['scores'][group]
            require(score['images']==score['boundary_images']==len(chosen),'E03 group count differs')
            require([score[k] for k in ['tp','fp','fn','tn']]==[tp,fp,fn,tn],'E03 confusion sum differs')
            require(all(math.isclose(score[k],v,rel_tol=0,abs_tol=1e-12) for k,v in recomputed.items()),
                    'E03 metric aggregation differs')
            for category in range(1,12):
                hits=sum(r['hits'][category] for r in chosen)
                total=sum(r['total'][category] for r in chosen)
                recall=hits/total if total else None
                reference=result['category_recall'][group][str(category)]
                require((reference['hits'],reference['total'],reference['recall'])==(hits,total,recall),
                        'E03 category JSON differs')
                exported=table[model,group,category]
                require(int(exported['hits'])==hits and int(exported['total'])==total
                        and (float(exported['recall']) if exported['recall'] else None)==recall,
                        'E03 category table differs')
            checks.append({'model':model,'subset':group,'images':len(chosen),'passed':True})
    result={'passed':True,'scope':'Read-only E03 table arithmetic, membership and recorded category denominators.',
        'member2_commit':'224d008c6b3a2ac27403ec86a0e71b972bcb5da2',
        'source_directory':str(SOURCE),'checks':checks,'category_rows':len(table),
        'train_validation_heldout_counts_zero':True,'test_partition_exact':True,
        'source_files_sha256':{str(p.relative_to(SOURCE)):sha256_file(p) for p in
            [SOURCE/'E03_category_recall.csv',*[SOURCE/f'E03_{m}'/n for m in ['A','D'] for n in ['result.json','per_image.csv']]]},
        'limitations':['No E03 model download, inference rerun, or independent mask decoding in this review.',
                      'Recorded original-coordinate denominators checked against the frozen manifest.',
                      'This review does not turn GSD scene transfer into an unseen-category experiment.']}
    save(ROOT/'E03_table_cross_review.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
