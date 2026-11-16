"""Run and verify the authorized A transfer experiment on the frozen GSD cohort."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from scripts.member4_gsd_review_simple import make_template, validate_decisions
from transparent_seg.data import decode_mask, read_manifest
from transparent_seg.evaluate import load_checkpoint, predict
from transparent_seg.evaluate_resumable import atomic_json, digest, read_chunk, code_identity
from transparent_seg.prepare import sha256_file

ROOT = Path('runs/member4_stage6_20261116')
RELEASE = Path('manifests/member4_gsd_reviewed_v2_20261116')
CHECKPOINT = Path('runs/member2_full_A_20261114/best.pt')
A_SHA = '3a785feb578ecbd2199684c53130737c872cd6c2f5803f221297f1e9dac0b25a'
FULL_SHA = 'cf33e8de136c5fd0be03c1e4ed70c0a6c866e734df5ca3cf1a5b5b8f6c767d82'
SUPPLEMENT_SHA = '29a75ebbf98b5aa019ad4f08a544ffbb36056cdefc512676a0f2bfad2b54b60a'
EVALUATION = ROOT / 'A_full'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def check_stage5():
    for name, expected in read(RELEASE / 'files_sha256.json').items():
        require(sha256_file(RELEASE / name) == expected, 'Frozen release changed: ' + name)
    review = read(RELEASE / 'human_review_record.json')
    validate_decisions(review, make_template())
    full, base = read_manifest(RELEASE / 'test_full.json')
    supplementary, other_base = read_manifest(RELEASE / 'test_supplementary.json')
    require(sha256_file(RELEASE / 'test_full.json') == FULL_SHA, 'Full manifest changed')
    require(sha256_file(RELEASE / 'test_supplementary.json') == SUPPLEMENT_SHA, 'Supplementary manifest changed')
    require(full['samples'] == supplementary['samples'] and base == other_base,
            'Cohorts differ; do not reuse the full-cohort score for a different subset')
    require(len(full['samples']) == 813 and len({r['id'] for r in full['samples']}) == 813,
            'GSD membership differs')
    require(full['encoding'] == 'binary_0255' and full['void_values'] == []
            and not any('difficulty' in r for r in full['samples']), 'GSD label/group protocol differs')
    require(read(RELEASE / 'exclusions.json') == [], 'Unexpected exclusions')
    require(sha256_file(CHECKPOINT) == A_SHA, 'Published A checkpoint changed')
    for path, expected in read('runs/member4_stage4_20261115/source_sha256.json').items():
        require(sha256_file(path) == expected, 'Frozen experiment implementation changed: ' + path)
    return full, base, review


def run(resume=False):
    full, _, review = check_stage5()
    require(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0), 'Expected local 5090')
    if resume:
        require((ROOT / 'authorization.json').is_file(), 'Resume requires the original authorized run')
        for path, expected in read(ROOT / 'source_sha256.json').items():
            require(sha256_file(path) == expected, 'Source changed before resume: ' + path)
    else:
        require(not ROOT.exists(), 'Use --resume or a new version; never overwrite a run')
        ROOT.mkdir(parents=True)
        # The frozen release records authorization at freeze time. This later user instruction
        # is recorded separately so the dataset hashes are preserved.
        atomic_json(ROOT / 'authorization.json', {
            'recorded_at': now(), 'instruction': '现在检查阶段五是否已经全部完成，如果是，继续阶段6',
            'stage5_verified_complete': True, 'stage6_authorized': True,
            'full_manifest_sha256': FULL_SHA, 'supplementary_manifest_sha256': SUPPLEMENT_SHA,
            'frozen_release_not_modified': True, 'stage7_authorized': False})
        snapshot = {**code_identity()['files'], 'scripts/member4_stage6_gsd.py': sha256_file(__file__)}
        atomic_json(ROOT / 'source_sha256.json', snapshot)
        for name in snapshot:
            target = ROOT / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(name, target)
        atomic_json(ROOT / 'stage5_gate.json', {'passed': True, 'reviewer': review['reviewer'],
            'reviewed_on': review['reviewed_on'], 'full_count': 813, 'supplementary_count': 813,
            'same_membership': True, 'unresolved_questions': 0, 'checked_at': now()})
        atomic_json(ROOT / 'qualitative_selection.json', {
            'selected_at': now(), 'fixed_ids': [r['gsd_id'] for r in review['label_samples']],
            'fixed_rule': 'The same 20 evenly spaced samples fixed before GSD model scoring.',
            'failure_rule': 'Three lowest per-image foreground IoU and three lowest Boundary IoU; ID breaks ties; deduplicate for display only.',
            'selection_does_not_change_evaluation_membership': True})
        atomic_json(ROOT / 'protocol.json', {
            'model': 'A', 'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': A_SHA,
            'manifest': str(RELEASE / 'test_full.json'), 'manifest_sha256': FULL_SHA,
            'supplementary_manifest_sha256': SUPPLEMENT_SHA,
            'cohort_reuse': 'Identical ordered samples and masks: one inference pass supports both cohorts.',
            'conditions': ['clean'], 'input_size': 512, 'precision': 'float32', 'threshold': 0.5,
            'logits_interpolation': 'bilinear to original resolution, align_corners=False',
            'region_aggregation': 'pooled TP/FP/FN/TN', 'boundary_aggregation': 'mean over images',
            'chunk_size': 25, 'training_or_target_tuning': False,
            'new_dataset_easy_hard_labels': False,
            'B_D_status': 'No formal B/D delivery found in local runs or current remote branches/releases; pending delivery.'})
        releases = Path('tmp/member4_stage6_remote_releases.json')
        if releases.is_file():
            atomic_json(ROOT / 'remote_releases_at_start.json', read(releases))
    logs = ROOT / 'logs'; logs.mkdir(exist_ok=True)
    args = [sys.executable, '-X', 'utf8', '-u', '-B', '-m', 'transparent_seg.evaluate_resumable',
            '--checkpoint', str(CHECKPOINT), '--manifest', str(RELEASE / 'test_full.json'),
            '--output', str(EVALUATION), '--device', 'cuda', '--conditions', 'clean', '--chunk-size', '25']
    if resume and (EVALUATION / 'identity.json').exists():
        args.append('--resume')
    started = time.perf_counter(); log = logs / f'A_{time.time_ns()}.log'
    atomic_json(ROOT / 'state.json', {'status': 'running_A', 'started_at': now(), 'log': str(log), 'arguments': args})
    try:
        with log.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            try:
                for line in process.stdout:
                    stream.write(line); stream.flush(); print(line.rstrip(), flush=True)
                code = process.wait()
            except BaseException:
                process.terminate(); process.wait(); raise
        require(code == 0, f'A evaluation failed ({code}); see {log}')
        atomic_json(ROOT / 'state.json', {'status': 'A_evaluation_complete', 'ended_at': now(),
                    'wall_seconds': time.perf_counter() - started, 'log': str(log), 'B_D_status': 'pending_delivery'})
    except BaseException as error:
        atomic_json(ROOT / 'state.json', {'status': 'failed', 'error': str(error), 'log': str(log), 'updated_at': now()})
        raise


def summarize():
    full, _, _ = check_stage5()
    result, identity = read(EVALUATION / 'result.json'), read(EVALUATION / 'identity.json')
    require(result['complete'] and result['image_count'] == 813 and result['conditions'] == ['clean'],
            'A evaluation is incomplete')
    require(identity['checkpoint_sha256'] == A_SHA and identity['manifest_sha256'] == FULL_SHA,
            'Model/data identity differs')
    require(identity['precision'] == 'float32' and identity['threshold'] == .5
            and identity['original_resolution'] and identity['size'] == 512, 'Inference rule differs')
    require(result['identity_sha256'] == digest(identity), 'Identity digest differs')
    for name, expected in read(ROOT / 'source_sha256.json').items():
        require(sha256_file(name) == expected, 'Run source changed: ' + name)
    rows = []
    for start in range(0, 813, 25):
        ids = [r['id'] for r in full['samples'][start:start+25]]
        block = read_chunk(EVALUATION / 'chunks' / f'{start:06d}.json', digest(identity), start, ids, ['clean'])
        require(block is not None, 'Invalid/missing chunk: ' + str(start))
        rows.extend(block['rows'])
    require([r['id'] for r in rows] == [r['id'] for r in full['samples']], 'Membership/order differs')
    totals = {key: sum(r['conditions']['clean']['metrics'][key] for r in rows) for key in ('tp', 'fp', 'fn', 'tn')}
    for row, sample in zip(rows, full['samples']):
        score = row['conditions']['clean']['metrics']
        require(score['images'] == score['boundary_images'] == 1 and score['skipped_all_void'] == 0,
                'Missing image/boundary score')
        require(score['tp'] + score['fn'] == sample['foreground_pixels']
                and sum(score[k] for k in totals) == sample['valid_pixels'], 'Pixel accounting differs')
    expected = {**totals, 'foreground_iou': totals['tp'] / (totals['tp'] + totals['fp'] + totals['fn']),
                'region_f1': 2*totals['tp'] / (2*totals['tp'] + totals['fp'] + totals['fn']),
                'background_fpr': totals['fp'] / (totals['fp'] + totals['tn']),
                'boundary_iou': sum(r['conditions']['clean']['metrics']['boundary_iou'] for r in rows) / 813}
    score = result['scores']['clean']['all']
    for key, value in expected.items():
        require(score[key] == value if key in totals else math.isclose(score[key], value, rel_tol=0, abs_tol=1e-12),
                'Independent aggregate differs: ' + key)
    with (EVALUATION / 'clean/per_image.csv').open(encoding='utf-8', newline='') as stream:
        exported = list(csv.DictReader(stream))
    require([r['id'] for r in exported] == [r['id'] for r in rows], 'CSV membership differs')
    for a, b in zip(exported, rows):
        require(a['prediction_sha256'] == b['conditions']['clean']['prediction_sha256'], 'Prediction hash export differs')
        for key in totals:
            require(int(a[key]) == b['conditions']['clean']['metrics'][key], 'CSV counts differ')
    selection = read(ROOT / 'qualitative_selection.json')
    failures = []
    for metric in ('foreground_iou', 'boundary_iou'):
        for row in sorted(rows, key=lambda r: (r['conditions']['clean']['metrics'][metric], r['id']))[:3]:
            failures.append({'id': row['id'], 'reason': 'three lowest per-image ' + metric,
                             'value': row['conditions']['clean']['metrics'][metric]})
    ids = list(dict.fromkeys(selection['fixed_ids'] + [r['id'] for r in failures]))
    results = ROOT / 'results'; results.mkdir(exist_ok=True)
    summary = {'status': 'A_complete_B_D_pending', 'stage5_complete': True, 'stage6_complete': False,
               'checkpoint_sha256': A_SHA, 'selected_A_epoch': 29, 'A_scores': score,
               'full_manifest_sha256': FULL_SHA, 'supplementary_manifest_sha256': SUPPLEMENT_SHA,
               'cohorts_share_identical_predictions': True, 'image_count': 813, 'unique_images': 812,
               'evaluation_seconds': result['elapsed_seconds'], 'gpu': identity['runtime']['gpu'],
               'precision': identity['precision'], 'threshold': .5,
               'A_D_comparison': 'pending_formal_D', 'B_D_perturbations': 'pending_formal_B_D',
               'unseen_category_evidence': False,
               'limitations': ['Single A model and seed; no A/D comparison yet.',
                              'Glass-scene transfer and unseen-category testing are distinct experiments.',
                              'Internal duplicate members and differing masks retained.',
                              'Fixed overlap search cannot prove no overlap; annotation scopes differ.']}
    atomic_json(results / 'summary.json', summary)
    atomic_json(results / 'gallery_selection.json', {**selection, 'failures': failures, 'all_ids': ids})
    atomic_json(results / 'verification.json', {'passed': True, 'images': 813, 'chunks': 33,
                'all_ids_unique_and_complete': True, 'region_counts_independently_verified': True,
                'boundary_mean_independently_verified': True, 'pixel_accounting_verified': True,
                'frozen_manifest_and_checkpoint_hashes_match': True, 'source_hashes_match': True,
                'full_and_supplementary_identical_membership': True})
    fields = ['model', 'cohort', 'status', 'images', 'foreground_iou_percent', 'region_f1_percent',
              'boundary_iou_percent', 'background_fpr_percent', 'manifest_sha256', 'shared_prediction_run']
    with (results / 'GSD_transfer_results.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for cohort, manifest_sha in [('full', FULL_SHA), ('supplementary', SUPPLEMENT_SHA)]:
            writer.writerow({'model': 'A', 'cohort': cohort, 'status': 'complete', 'images': 813,
                **{k+'_percent': score[k]*100 for k in ('foreground_iou', 'region_f1', 'boundary_iou', 'background_fpr')},
                'manifest_sha256': manifest_sha, 'shared_prediction_run': str(EVALUATION)})
            writer.writerow({'model': 'D', 'cohort': cohort, 'status': 'pending_formal_model',
                             'manifest_sha256': manifest_sha})
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def gallery():
    full, base, _ = check_stage5()
    selection = read(ROOT / 'results/gallery_selection.json')
    out = ROOT / 'results/gallery'; require(not out.exists(), 'Preserve existing gallery'); out.mkdir()
    samples = {r['id']: r for r in full['samples']}
    with (EVALUATION / 'clean/per_image.csv').open(encoding='utf-8', newline='') as stream:
        scores = {r['id']: r for r in csv.DictReader(stream)}
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True; torch.use_deterministic_algorithms(True)
    model, _ = load_checkpoint(CHECKPOINT, 'cuda')
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    index = []
    for n, identity in enumerate(selection['all_ids'], 1):
        sample = samples[identity]
        with Image.open(base / sample['image']) as im: image = im.convert('RGB')
        target, _ = decode_mask(base / sample['mask'], 'binary_0255', [])
        pred = predict(model, image, 512, 'cuda')
        require(hashlib.sha256(np.packbits(pred).tobytes()).hexdigest() == scores[identity]['prediction_sha256'],
                'Gallery prediction differs from complete evaluation: ' + identity)
        mask = Image.fromarray(pred.astype(np.uint8)*255); mask.save(out / f'{n:02d}_mask.png')
        rgb = np.asarray(image).astype(np.float32); rgb[pred] = .5*rgb[pred]+.5*np.array([0,220,160])
        overlay = Image.fromarray(rgb.astype(np.uint8)); overlay.save(out / f'{n:02d}_overlay.png')
        canvas = Image.new('RGB', (1600, 530), 'white'); draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f'{n:02d}  {identity}', fill='black', font=font)
        score = scores[identity]
        reason = '评分前固定样本' if identity in selection['fixed_ids'] else '按预先规则选出的低分案例'
        draw.text((12, 36), f"{reason} | A IoU={float(score['foreground_iou']):.4f}, Boundary IoU={float(score['boundary_iou']):.4f}", fill='black', font=font)
        for col, (view, title) in enumerate(zip((image, Image.fromarray((target*255).astype(np.uint8)).convert('RGB'), mask.convert('RGB'), overlay),
                                               ('原图', '原标签：白色玻璃', 'A预测：白色前景', 'A预测叠图'))):
            draw.text((col*400+10, 72), title, fill='black', font=font)
            thumb = view.copy(); thumb.thumbnail((380, 410))
            canvas.paste(thumb, (col*400+(400-thumb.width)//2, 105+(410-thumb.height)//2))
        canvas.save(out / f'{n:02d}.png')
        index.append({'index': n, 'id': identity, 'file': f'{n:02d}.png', 'fixed_before_scoring': identity in selection['fixed_ids'],
                      'failure_reasons': [r for r in selection['failures'] if r['id'] == identity],
                      'prediction_hash_matches_full_evaluation': True})
    atomic_json(out / 'index.json', index)
    print(f'Generated {len(index)} panels and original-resolution masks/overlays; all prediction hashes match.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run', 'summarize', 'gallery'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'run': run(args.resume)
    elif args.command == 'summarize': summarize()
    else: gallery()
