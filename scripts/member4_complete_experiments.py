"""Run the remaining member 4 B/D robustness and A/D GSD comparisons."""
import argparse
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
DEPS = PROJECT/'.local_deps/member4_opencv_4_12'
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
    os.environ['PYTHONPATH'] = str(DEPS) + os.pathsep + str(PROJECT)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import torch
from transparent_seg.evaluate_resumable import atomic_json, code_identity, digest, run as evaluate
from transparent_seg.prepare import sha256_file

ROOT = PROJECT/'runs/member4_completion_20261118'
DELIVERY = PROJECT/'runs/member4_BD_delivery_20261118/full_runs'
TEST = PROJECT/'manifests/member1_final_20261029/v1/test.json'
GSD = PROJECT/'manifests/member4_gsd_reviewed_v2_20261116'
A = PROJECT/'runs/member2_full_A_20261114/best.pt'
FIXED_SHA = {
 'test':'ecf3ad309423aa0247183cfc1e315921a7666c47d10700f381199a7dcfb41012',
 'gsd_full':'cf33e8de136c5fd0be03c1e4ed70c0a6c866e734df5ca3cf1a5b5b8f6c767d82',
 'gsd_supplementary':'29a75ebbf98b5aa019ad4f08a544ffbb36056cdefc512676a0f2bfad2b54b60a',
 'A':'3a785feb578ecbd2199684c53130737c872cd6c2f5803f221297f1e9dac0b25a',
 'B':'e219d6bf2f7f7cc57711d8df7354202efd81360108247f16dd4cc72765549d81',
 'D':'49498d53b63b13368ceafa5200842a7136c41172c9decad6000c23d1c22b93a4',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(ok,message):
    if not ok: raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    return {**code_identity()['files'],**{
        str(p.relative_to(PROJECT)).replace('\\','/'):sha256_file(p)
        for p in [Path(__file__), PROJECT/'scripts/member4_verify_BD_delivery.py']}}


def gate():
    require(read(ROOT/'intake_verification.json')['passed'], 'B/D intake has not passed')
    require(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0), 'Expected local 5090')
    for key,path in [('test',TEST),('gsd_full',GSD/'test_full.json'),
                     ('gsd_supplementary',GSD/'test_supplementary.json'),('A',A),
                     ('B',DELIVERY/'B03_B/best.pt'),('D',DELIVERY/'B03_D/best.pt')]:
        require(sha256_file(path)==FIXED_SHA[key], 'Frozen input changed: '+key)
    for name,expected in read(GSD/'files_sha256.json').items():
        require(sha256_file(GSD/name)==expected, 'GSD frozen record changed: '+name)
    require(read(GSD/'test_full.json')['samples']==read(GSD/'test_supplementary.json')['samples'],
            'GSD supplementary cohort differs; cannot reuse predictions')
    # Permit only the explicitly checked backend addition relative to the prior run.
    old=read(PROJECT/'runs/member4_A_full_test_20261115/identity.json')['code']['files']
    for name,expected in old.items():
        if name!='transparent_seg/evaluate_resumable.py':
            require(sha256_file(PROJECT/name)==expected, 'Unexpected core source change: '+name)
    suites=ET.parse(ROOT/'tests_02.xml').getroot()
    require(all(int(s.attrib.get('failures',0))==0 and int(s.attrib.get('errors',0))==0
                for s in suites.iter('testsuite')), 'Regression tests have not passed')


def preflight():
    gate()
    prior=ROOT/'preflight.json'
    if prior.exists():
        require(read(prior)['source_sha256']==sources(), 'Preflight source identity changed')
        print('Verified preflight already exists',flush=True); return
    equivalence=[]
    for model in ['B','D']:
        folder=ROOT/'preflight'/f'{model}_equivalence8'
        result=evaluate(DELIVERY/f'B03_{model}/best.pt',
                        PROJECT/'runs/member4_stage3_20261115/equivalence8.json',folder,
                        'cuda',verify_legacy=True,boundary_backend='opencv',
                        resume=(folder/'identity.json').exists())
        require(result['complete'] and result['image_count']==8, 'Equivalence check incomplete')
        equivalence.append({'model':model,'images':8,'conditions':13,
                            'prediction_pixels_and_metrics_exact':True,'seconds':result['elapsed_seconds']})
    folder=ROOT/'preflight/D_timing40'
    timing=evaluate(DELIVERY/'B03_D/best.pt',PROJECT/'runs/member4_stage3_20261115/timing40.json',
                    folder,'cuda',boundary_backend='opencv',resume=(folder/'identity.json').exists())
    require(timing['complete'] and timing['image_count']==40,'Timing run incomplete')
    chunks=[read(p)['body'] for p in sorted((folder/'chunks').glob('*.json'))]
    processing=sum(p['seconds'] for p in chunks)
    estimate=processing/40*4428*2
    value={'passed':True,'source_sha256':sources(),'equivalence':equivalence,
           'timing_images':40,'timing_conditions':13,'timing_processing_seconds':processing,
           'estimated_B_D_test_processing_seconds':estimate,
           'planning_range_seconds':[estimate*1.25,estimate*1.5],
           'boundary_backend':'opencv','gpu':torch.cuda.get_device_name(0),
           'notes':['No change to model inference, threshold, metric definition or test membership.',
                    'Timing extrapolation is approximate; startup, GSD and reporting are additional.']}
    atomic_json(prior,value); print(json.dumps(value),flush=True)


def compare_A_bridge():
    original=PROJECT/'runs/member4_stage6_20261116/A_full'
    bridge=ROOT/'A_GSD_bridge'
    old,new=read(original/'result.json'),read(bridge/'result.json')
    require(new['complete'] and new['image_count']==813 and old['scores']==new['scores'],
            'A GSD fast-backend aggregate differs from frozen result')
    require((original/'clean/per_image.csv').read_bytes()==(bridge/'clean/per_image.csv').read_bytes(),
            'A GSD per-image metrics or prediction hashes differ')
    atomic_json(ROOT/'A_GSD_bridge_verification.json',{'passed':True,'images':813,
        'all_prediction_hashes_and_metrics_exact':True,
        'original':str(original.relative_to(PROJECT)),'bridge':str(bridge.relative_to(PROJECT))})


def queue(resume):
    gate(); require(read(ROOT/'preflight.json')['passed'],'Preflight missing')
    require(read(ROOT/'preflight.json')['source_sha256']==sources(),'Preflight code changed')
    plan=[('A_GSD_bridge',A,GSD/'test_full.json','clean'),
          ('D_GSD',DELIVERY/'B03_D/best.pt',GSD/'test_full.json','clean'),
          ('B_test',DELIVERY/'B03_B/best.pt',TEST,'all'),
          ('D_test',DELIVERY/'B03_D/best.pt',TEST,'all')]
    protocol=ROOT/'protocol.json'
    if protocol.exists():
        require(resume,'Use --resume to continue a recorded queue')
        require(read(protocol)['source_sha256']==sources(),'Queue source changed')
    else:
        require(not resume,'No recorded queue to resume')
        atomic_json(protocol,{'authorized_at':now(),'user_instruction':
            '检查成员2提交是否可用；可用则按照之前计划推进，完成成员4全部剩余任务。',
            'source_sha256':sources(),'git':code_identity()['commit'],'inputs':FIXED_SHA,
            'steps':[(n,str(c.relative_to(PROJECT)),str(m.relative_to(PROJECT)),k) for n,c,m,k in plan],
            'precision':'float32','threshold':0.5,'size':512,'chunk_size':25,
            'boundary_backend':'opencv','training':False,'target_tuning':False,
            'clean_cross_hardware_metric_tolerance_fraction':1e-6,
            'qualitative_selection':{'GSD_fixed_ids':read(PROJECT/'runs/member4_stage6_20261116/qualitative_selection.json')['fixed_ids'],
                'GSD_failures':'three lowest D image IoU; three lowest D-minus-A image IoU; ties by image ID',
                'test_fixed_ids':[x['id'] for x in read(PROJECT/'runs/member4_stage4_20261115/results/gallery/gallery_index.json')[:12]],
                'test_failures':'three lowest D-minus-B image IoU under gamma_1.6; ties by image ID'},
            'reporting':'Reuse sealed A/C and A-GSD records; do not rewrite their historical status.'})
        for name in sources():
            target=ROOT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PROJECT/name,target)
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    awake=bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)) if os.name=='nt' else False
    try:
        for label,checkpoint,manifest,conditions in plan:
            output=ROOT/label
            if (output/'result.json').exists() and read(output/'result.json')['complete']:
                require(read(output/'identity.json')['code']==code_identity(),'Completed run code changed')
                if label=='A_GSD_bridge': compare_A_bridge()
                continue
            args=[sys.executable,'-X','utf8','-u','-B','-m','transparent_seg.evaluate_resumable',
                  '--checkpoint',str(checkpoint),'--manifest',str(manifest),'--output',str(output),
                  '--device','cuda','--conditions',conditions,'--chunk-size','25','--boundary-backend','opencv']
            if (output/'identity.json').exists(): args.append('--resume')
            atomic_json(ROOT/'state.json',{'status':'running','step':label,'updated_at':now(),'arguments':args})
            log=logs/f'{label}_{time.time_ns()}.log'
            with log.open('x',encoding='utf-8') as stream:
                process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,cwd=PROJECT)
                print(f'START {label} PID={process.pid} LOG={log}',flush=True)
                code=process.wait()
            require(code==0,f'{label} failed ({code}); see {log}')
            if label=='A_GSD_bridge': compare_A_bridge()
            print(f'COMPLETE {label}',flush=True)
        atomic_json(ROOT/'state.json',{'status':'experiments_complete','updated_at':now(),
                                      'completed_steps':[p[0] for p in plan]})
    except BaseException as exc:
        if 'process' in locals() and process.poll() is None:
            process.terminate();process.wait()
        atomic_json(ROOT/'state.json',{'status':'failed','updated_at':now(),'error':str(exc)})
        raise
    finally:
        if awake: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','run'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    os.chdir(PROJECT)
    if args.stage=='preflight': preflight()
    else: queue(args.resume)
