"""Independently verify and summarize the completed four-model experiments."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from transparent_seg.evaluate_resumable import CONDITIONS, digest, read_chunk
from transparent_seg.prepare import sha256_file

ROOT=Path('runs/member4_completion_20261118')
REPORT=Path('reports/member4_completion_20261118')
COUNTS=('tp','fp','fn','tn')
METRICS=('foreground_iou','region_f1','boundary_iou','background_fpr')
TEST=Path('manifests/member1_final_20261029/v1/test.json')
GSD=Path('manifests/member4_gsd_reviewed_v2_20261116/test_full.json')
RUNS={'A':Path('runs/member4_A_full_test_20261115'),
      'B':ROOT/'B_test','C':Path('runs/member4_C_full_test_20261115'),'D':ROOT/'D_test'}
GSDS={'A':Path('runs/member4_stage6_20261116/A_full'),'D':ROOT/'D_GSD'}
SHAS={'A':'3a785feb578ecbd2199684c53130737c872cd6c2f5803f221297f1e9dac0b25a',
      'B':'e219d6bf2f7f7cc57711d8df7354202efd81360108247f16dd4cc72765549d81',
      'C':'69841f34ec0c76b0a2dbdce3a07f8ea5926a6beec847c4e8b03e587efc3b5f0d',
      'D':'49498d53b63b13368ceafa5200842a7136c41172c9decad6000c23d1c22b93a4'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(ok,message):
    if not ok: raise ValueError(message)


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def audit_run(model,folder,manifest,names):
    meta=read(manifest);samples=meta['samples'];ids=[s['id'] for s in samples]
    result,identity=read(folder/'result.json'),read(folder/'identity.json')
    require(len(ids)==len(set(ids)), 'Manifest duplicate IDs')
    require(result['complete'] and result['image_count']==len(ids) and result['conditions']==names,
            f'Incomplete run: {folder}')
    require(identity['checkpoint_sha256']==SHAS[model] and identity['manifest_sha256']==sha256_file(manifest),
            f'Model/data identity differs: {folder}')
    require(identity['precision']=='float32' and identity['threshold']==.5
            and identity['original_resolution'] and identity['size']==512,
            f'Inference protocol differs: {folder}')
    identity_hash=digest(identity)
    require(result['identity_sha256']==identity_hash,'Identity digest changed')
    chunk_size=identity['chunk_size'];rows=[]
    for start in range(0,len(ids),chunk_size):
        block=read_chunk(folder/'chunks'/f'{start:06d}.json',identity_hash,start,
                         ids[start:start+chunk_size],names)
        require(block is not None,f'Invalid/incomplete saved block: {folder}/{start}')
        rows.extend(block['rows'])
    require([r['id'] for r in rows]==ids,'Chunk membership/order differs')
    require([r['difficulty'] for r in rows]==[s.get('difficulty','unspecified') for s in samples],
            'Easy/hard group assignment differs')
    groups=['all']+(['easy','hard'] if manifest==TEST else [])
    table=[];verification=[]
    for condition in names:
        with (folder/condition/'per_image.csv').open(encoding='utf-8-sig',newline='') as f:
            exported=list(csv.DictReader(f))
        require([r['id'] for r in exported]==ids,'CSV IDs missing, duplicated or reordered')
        for row,sample,csv_row in zip(rows,samples,exported):
            item=row['conditions'][condition];score=item['metrics']
            require(score['images']==score['boundary_images']==1 and score['skipped_all_void']==0,
                    'An image or its boundary was skipped')
            require(score['tp']+score['fn']==sample['foreground_pixels']
                    and sum(score[k] for k in COUNTS)==sample.get('valid_pixels',sample['width']*sample['height']),
                    'Pixel accounting differs from frozen label')
            require(csv_row['prediction_sha256']==item['prediction_sha256'],'Prediction hash export differs')
            for key in COUNTS:
                require(int(csv_row[key])==score[key],'CSV pixel counts differ')
            for key in METRICS:
                require(score[key] is not None and math.isfinite(score[key]) and 0<=score[key]<=1,
                        'Invalid per-image metric')
                require(float(csv_row[key])==score[key],'CSV per-image score differs')
        for group in groups:
            chosen=[r['conditions'][condition]['metrics'] for r,s in zip(rows,samples)
                    if group=='all' or s.get('difficulty')==group]
            counts={k:sum(r[k] for r in chosen) for k in COUNTS}
            tp,fp,fn,tn=[counts[k] for k in COUNTS]
            reference={**counts,'foreground_iou':tp/(tp+fp+fn),
                'region_f1':2*tp/(2*tp+fp+fn),
                'boundary_iou':math.fsum(r['boundary_iou'] for r in chosen)/len(chosen),
                'background_fpr':fp/(fp+tn)}
            score=result['scores'][condition][group]
            require(score['images']==score['boundary_images']==len(chosen),'Group image count differs')
            require(all(score[k]==reference[k] for k in COUNTS),'Pooled counts differ')
            require(all(math.isclose(score[k],reference[k],rel_tol=0,abs_tol=1e-12) for k in METRICS),
                    'Independent metric aggregation differs')
            require(read(folder/condition/'metrics.json')['scores'][group]==score,'Condition export differs')
            table.append({'model':model,'condition':condition,'subset':group,**score})
        verification.append({'model':model,'condition':condition,'images':len(ids),
                             'membership_exact':True,'counts_and_metrics_verified':True})
    return table,verification,rows,identity


def main():
    require(read(ROOT/'state.json')['status']=='experiments_complete','Formal queue incomplete')
    require(read(ROOT/'A_GSD_bridge_verification.json')['passed'],'Old/new GSD bridge failed')
    frozen=read(ROOT/'protocol.json')['inputs']
    supplementary=GSD.with_name('test_supplementary.json')
    require(sha256_file(GSD)==frozen['gsd_full']
            and sha256_file(supplementary)==frozen['gsd_supplementary']
            and read(GSD)['samples']==read(supplementary)['samples'],
            'Frozen GSD full/supplementary identity changed after scoring')
    for name,expected in read(GSD.parent/'files_sha256.json').items():
        require(sha256_file(GSD.parent/name)==expected,'A frozen GSD review record changed')
    require(not REPORT.exists(),'Use a new report version; do not overwrite a completed package')
    REPORT.mkdir(parents=True)
    table=[];audits=[];identities={};raw={}
    for model,folder in RUNS.items():
        part,audit,rows,identity=audit_run(model,folder,TEST,list(CONDITIONS))
        table.extend(part);audits.extend(audit);identities[model]=identity;raw[model]=rows
    require(len({v['source_inventory_sha256'] for v in identities.values()})==1,
            'Four models used different image/mask bytes')
    write_csv(REPORT/'ABCD_condition_results.csv',table)
    index={(r['model'],r['condition'],r['subset']):r for r in table}
    robust=[]
    for model in RUNS:
        for group in ['all','easy','hard']:
            for metric in METRICS:
                clean=index[(model,'clean',group)][metric]*100
                mean=math.fsum(index[(model,c,group)][metric]*100 for c in CONDITIONS if c!='clean')/12
                robust.append({'model':model,'subset':group,'metric':metric,
                               'clean_percent':clean,'perturbed_mean_percent':mean,'drop_pp':clean-mean})
    write_csv(REPORT/'ABCD_robustness_summary.csv',robust)
    ri={(r['model'],r['subset'],r['metric']):r for r in robust}
    effects=[]
    for first,second in [('C','A'),('D','B'),('B','A'),('D','C'),('D','A')]:
        for group in ['all','easy','hard']:
            for metric in METRICS:
                a,b=ri[(first,group,metric)],ri[(second,group,metric)]
                effects.append({'comparison':first+'_minus_'+second,'subset':group,'metric':metric,
                                'clean_delta_pp':a['clean_percent']-b['clean_percent'],
                                'perturbed_mean_delta_pp':a['perturbed_mean_percent']-b['perturbed_mean_percent'],
                                'drop_delta_pp':a['drop_pp']-b['drop_pp']})
    write_csv(REPORT/'ABCD_effects_percentage_points.csv',effects)
    clean=[{'model':m,'images':4428,**{k+'_percent':index[(m,'clean','all')][k]*100 for k in METRICS}} for m in RUNS]
    write_csv(REPORT/'ABCD_clean_results.csv',clean)
    supplied=read('runs/member4_BD_delivery_20261118/B03_B04_summary.json')
    tolerance=read(ROOT/'protocol.json')['clean_cross_hardware_metric_tolerance_fraction']
    clean_checks=[]
    for row in supplied['clean_test']:
        if row['model'] not in ['B','D']: continue
        local=index[(row['model'],'clean',row['subset'])]
        deltas={k:local[k]-row[k] for k in METRICS}
        require(all(abs(v)<=tolerance for v in deltas.values()),'B/D clean score differs from release beyond frozen tolerance')
        clean_checks.append({'model':row['model'],'subset':row['subset'],'fraction_deltas':deltas,
                             'tolerance_fraction':tolerance,'passed':True})
    gsd_table=[];gsd_scores={};gsd_raw={};gsd_identities={}
    for model,folder in GSDS.items():
        part,audit,rows,identity=audit_run(model,folder,GSD,['clean'])
        gsd_identities[model]=identity
        audits.extend(audit);gsd_scores[model]=part[0];gsd_raw[model]=rows
        for cohort in ['full','supplementary']:
            gsd_table.append({'model':model,'cohort':cohort,'images':813,
                              **{k+'_percent':part[0][k]*100 for k in METRICS},
                              'same_ordered_cohort':True,'evaluation_path':str(folder)})
    require(len({v['source_inventory_sha256'] for v in gsd_identities.values()})==1,
            'A/D GSD image/mask source inventories differ')
    write_csv(REPORT/'GSD_AD_transfer_results.csv',gsd_table)
    gsd_delta={k:(gsd_scores['D'][k]-gsd_scores['A'][k])*100 for k in METRICS}
    save(REPORT/'GSD_D_minus_A_pp.json',gsd_delta)
    protocol=read(ROOT/'protocol.json')
    policy=protocol['qualitative_selection']
    ad={r['id']:r['conditions']['clean']['metrics'] for r in gsd_raw['A']}
    dd={r['id']:r['conditions']['clean']['metrics'] for r in gsd_raw['D']}
    failures=[]
    for name,ids in [
        ('three lowest D image IoU',sorted(dd,key=lambda k:(dd[k]['foreground_iou'],k))[:3]),
        ('three lowest D-minus-A image IoU',sorted(dd,key=lambda k:(dd[k]['foreground_iou']-ad[k]['foreground_iou'],k))[:3])]:
        failures.extend({'dataset':'GSD','id':k,'condition':'clean','reason':name,
                         'first_model':'A','second_model':'D','first_iou':ad[k]['foreground_iou'],
                         'second_iou':dd[k]['foreground_iou']} for k in ids)
    bd={m:{r['id']:r['conditions']['gamma_1.6']['metrics'] for r in raw[m]} for m in ['B','D']}
    for key in sorted(bd['D'],key=lambda k:(bd['D'][k]['foreground_iou']-bd['B'][k]['foreground_iou'],k))[:3]:
        failures.append({'dataset':'Trans10K','id':key,'condition':'gamma_1.6',
            'reason':'three lowest D-minus-B image IoU under gamma_1.6',
            'first_model':'B','second_model':'D','first_iou':bd['B'][key]['foreground_iou'],
            'second_iou':bd['D'][key]['foreground_iou']})
    write_csv(REPORT/'failure_selection.csv',failures)
    save(REPORT/'gallery_selection.json',{'rules_fixed_before_formal_scoring':policy,'failures':failures,
        'gsd_ids':list(dict.fromkeys(policy['GSD_fixed_ids']+[r['id'] for r in failures if r['dataset']=='GSD'])),
        'test_ids':list(dict.fromkeys(policy['test_fixed_ids']+[r['id'] for r in failures if r['dataset']=='Trans10K']))})
    save(REPORT/'verification.json',{'passed':True,'test_conditions':52,'images_per_test_condition':4428,
        'test_image_condition_predictions':52*4428,'new_B_D_test_predictions':26*4428,
        'gsd_models':['A','D'],'gsd_records_per_model':813,'all_audits':audits,
        'four_models_share_source_inventory':True,'full_and_supplementary_GSD_identical':True,
        'B_D_release_clean_recheck':clean_checks,'source_identities':identities,
        'A_D_GSD_same_source_inventory':True,'gsd_source_identities':gsd_identities})
    overall={model:{metric:ri[(model,'all',metric)] for metric in METRICS} for model in RUNS}
    save(REPORT/'summary.json',{'generated_at':datetime.now(timezone.utc).isoformat(),
        'status':'formal_evaluations_verified','stage6_experiments_complete':True,
        'stage7_packaging_complete':False,'ABCD':overall,'GSD':gsd_scores,'GSD_D_minus_A_pp':gsd_delta,
        'source_runs':{m:str(p) for m,p in RUNS.items()},'model_sha256':SHAS,
        'limitations':['One training seed; different training GPUs/Python environments.',
            'No significance claim; both improvements and regressions retained.',
            'GSD scene transfer is distinct from E03 unseen-category evaluation.',
            '813 GSD records contain 812 unique decoded photographs; frozen duplicates retained.',
            'Finite duplicate search cannot establish complete absence of overlap.']})
    for name in ['intake_verification.json','preflight.json','protocol.json','A_GSD_bridge_verification.json']:
        save(REPORT/name,read(ROOT/name))
    for name in ['E03_table_cross_review.json','AC_reuse_verification.json','GSD_independent_verification.json',
                 'scheduling_appendix.json','parallel_trial_check.json','D_parallel_state.json']:
        if (ROOT/name).exists():
            save(REPORT/name,read(ROOT/name))
    if (ROOT/'remote_delivery_recheck.json').exists():
        save(REPORT/'remote_delivery_recheck.json',read(ROOT/'remote_delivery_recheck.json'))
    print(json.dumps({'report':str(REPORT),'ABCD_foreground_iou':{m:overall[m]['foreground_iou'] for m in RUNS},
                      'GSD_D_minus_A_pp':gsd_delta},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
