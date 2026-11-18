"""Archive numerical experiment evidence without images, datasets or weights."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_member4_completion import ROOT, REPORT, RUNS, GSDS, read, save, require


def main():
    require(read(REPORT/'verification.json')['passed'],'Results must pass independent verification')
    archive=Path('deliverables/member4_final_evaluation_records_20261118.zip')
    require(not archive.exists(),'Archive already exists')
    selected=set()
    for folder in [*RUNS.values(),*GSDS.values(),ROOT/'A_GSD_bridge']:
        for name in ['identity.json','result.json','source_inventory.json','progress.json']:
            selected.add(folder/name)
        selected.update((folder/'chunks').glob('*.json'))
        selected.update((folder/'attempts').glob('*.json'))
        selected.update(folder.glob('*/per_image.csv'))
        selected.update(folder.glob('*/metrics.json'))
    for folder in [ROOT/'source_snapshot',Path('runs/member4_stage4_20261115/source_snapshot'),
                   Path('runs/member4_stage6_20261116/source_snapshot'),ROOT/'logs',
                   ROOT/'scheduling_source',
                   Path('manifests/member4_gsd_reviewed_v2_20261116')]:
        selected.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ['.json','.py','.txt','.log'])
    selected.update(ROOT.glob('*.json'))
    selected.update(ROOT.glob('*.jsonl'))
    selected.update(ROOT.glob('parallel_controller*.log'))
    selected.update(ROOT/name for name in ['queue.log','postprocessing.log','tests_02.log','tests_02.xml'])
    for split in ['train','validation','test']:
        selected.add(Path('manifests/member1_final_20261029/v1')/(split+'.json'))
    for folder in [Path('runs/member2_full_A_20261114'),Path('runs/member4_full_C_seed42_20261115'),
                   *[Path('runs/member4_BD_delivery_20261118/full_runs')/('B03_'+m) for m in ['B','D']]]:
        for name in ['config.json','environment.json','history.jsonl','pip_freeze.txt']:
            path=folder/name
            if path.exists():selected.add(path)
    selected.update(p for p in REPORT.iterdir() if p.suffix in ['.csv','.json'])
    selected={p for p in selected if p.is_file()}
    require(all(p.suffix not in ['.pt','.pth','.safetensors','.jpg','.png'] for p in selected),
            'Unexpected data/weight payload')
    entries={p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(selected)}
    instructions='''Member 4 final numerical records, run label 20261118.
This archive contains all four models' 13-condition full-test statistics and
A/D GSD records, preserved chunks, manifests, training logs, and source snapshots.
It contains no dataset image bytes or checkpoint tensors. Obtain the unchanged
A/C/B/D checkpoints from the documented GitHub Releases and verify their SHA256.
Extract into a NEW review directory. Do not overwrite an active experiment.
Each original run identity fixes its model, source files, data and runtime.
RECORDS_SHA256.json verifies archived evidence, not the external datasets/models.
GSD full and supplementary have identical members and are not independent trials.
'''
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('READ_FIRST.txt',instructions)
        z.writestr('RECORDS_SHA256.json',json.dumps(entries,indent=2)+'\n')
        for p in sorted(selected):z.write(p,p.as_posix())
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None,'Damaged ZIP')
        require(all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in entries.items()),'Archived bytes differ')
    receipt={'path':str(archive),'bytes':archive.stat().st_size,'files':len(selected),
             'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'all_archived_files_verified':True,
             'weights_included':False,'dataset_images_included':False}
    save(ROOT/'raw_records_archive.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
