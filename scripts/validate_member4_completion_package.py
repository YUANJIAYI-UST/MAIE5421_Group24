"""Seal the final handoff after all four rendered slides have been inspected."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.summarize_member4_completion import REPORT, ROOT, read, save, require
from scripts.package_member4_completion import OUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-review-confirmed',action='store_true',
                        help='Only after inspecting all four actual rendered slides')
    args=parser.parse_args()
    require(args.visual_review_confirmed,'Visual review is still required')
    require(not (OUT/'validation.json').exists(),'Already sealed')
    build=Path('tmp/member4_complete_ppt_20261118')
    require((build/'final_validation.json').is_file(),'Presentation finalizer receipt missing')
    require(read(REPORT/'verification.json')['passed'],'Formal evidence incomplete')
    slides=read(OUT/'presentation_data.json')
    with zipfile.ZipFile(OUT/'MEMBER4_Presentation_4slides.pptx') as archive:
        require(archive.testzip() is None,'Invalid presentation ZIP member')
        slide_names=sorted(n for n in archive.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n))
        notes_names=sorted(n for n in archive.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n))
        require(len(slide_names)==len(notes_names)==4,'Slide/notes count differs')
        visible=[' '.join(ET.fromstring(archive.read(n)).itertext()) for n in slide_names]
        require(not any(re.search(r'pending|待补充',s,re.I) for s in visible),'Old pending labels remain')
        summary=slides['summary']
        for model in 'ABCD':
            for metric in ['foreground_iou','region_f1','boundary_iou']:
                require(f"{summary['ABCD'][model][metric]['clean_percent']:.2f}" in visible[1],
                        'Clean slide differs from source')
        for model in 'AD':
            for metric in ['foreground_iou','boundary_iou','background_fpr']:
                require(f"{summary['GSD'][model][metric]*100:.2f}" in visible[3],
                        'GSD slide differs from source')
        for note_path,note in zip(notes_names,slides['slides']):
            words=' '.join(ET.fromstring(archive.read(note_path)).itertext())
            require(note['en'] in words,'Embedded English notes differ')
    with (OUT/'evidence/ABCD_condition_results.csv').open(encoding='utf-8-sig',newline='') as f:
        require(len(list(csv.DictReader(f)))==156,'Expected 156 condition/subset rows')
    require(all(x['all_prediction_hashes_match_formal_run'] for x in read(OUT/'gallery/index.json')),
            'Gallery mismatch')
    previews=OUT/'slide_previews'
    previews.mkdir(exist_ok=True)
    for i in range(1,5):
        source=build/f'slide-{i}.png'
        require(source.is_file(),f'Missing inspected slide {i}')
        shutil.copyfile(source,previews/source.name)
    shutil.copyfile(build/'final_validation.json',OUT/'presentation_validation.json')
    for folder in [REPORT,OUT/'evidence']:
        current=read(folder/'summary.json')
        current['stage7_packaging_complete']=True
        current['stage7_deliverable']=str(OUT)
        save(folder/'summary.json',current)
    slides['summary']['stage7_packaging_complete']=True
    slides['summary']['stage7_deliverable']=str(OUT)
    save(OUT/'presentation_data.json',slides)
    save(OUT/'packaging_status.json',{'writing_and_evidence_complete':True,
        'pptx_validation_pending':False,'stage7_complete':True})
    save(OUT/'validation.json',{'passed':True,'checked_at':datetime.now(timezone.utc).isoformat(),
        'slides':4,'speaker_note_sections':4,'visual_review_all_four_slides':True,
        'formal_results_verified':True,'gallery_prediction_hashes_verified':True,
        'presentation_numbers_match_source':True,'native_tables_on_slides':[2,4],
        'scientific_plot_generated_with_python':True,'stage7_complete':True,
        'limitations':['Not tested in the desktop Microsoft PowerPoint application.',
                      'Team report assembly, final PR merges and rehearsal remain team responsibilities.']})
    paths=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
    lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(OUT).as_posix() for p in paths]
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=OUT.with_suffix('.zip')
    require(not archive.exists(),'Package archive already exists')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with zipfile.ZipFile(archive) as z:require(z.testzip() is None,'Package archive damaged')
    receipt={'passed':True,'stage6_complete':True,'stage7_complete':True,'package':str(OUT),
        'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'archive_bytes':archive.stat().st_size,'checked_files':len(paths),'PR_merged':False}
    save(ROOT/'completion_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
