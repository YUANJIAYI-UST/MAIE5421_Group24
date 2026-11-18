"""Verify completed B03/E03 runs and summarize their paired effects."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from transparent_seg.prepare import sha256_file, write_json
from transparent_seg.model import TransparentSegformer
from evaluate_group24_e03 import INIT, FROZEN

AB_METRICS = ('foreground_iou', 'region_f1', 'boundary_iou', 'background_fpr')
V1_TEST = 'ecf3ad309423aa0247183cfc1e315921a7666c47d10700f381199a7dcfb41012'
V1_TRAIN = '7a3094e0c1aeacf0edd7335f196af79eacd4b551cd9f78cce2cf7c903e910ca2'
V1_VAL = '861c24027bfddcfba031e253a6518343010e31d296fb66c010f8d88ee9a90497'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def report_training(run, stage, isolated, output):
    train_count, val_count = (3230, 646) if isolated else (4959, 995)
    per_epoch = math.ceil(train_count / 8)
    config = read(run / 'config.json')
    planned = read(Path(f'configs/group24_{stage[:3].lower()}_{stage[-1]}_seed42.json'))
    environment = read(run / 'environment.json')
    train_hash, val_hash = ((FROZEN['train'], FROZEN['validation']) if isolated else (V1_TRAIN, V1_VAL))
    best_path, last_path = run / 'best.pt', run / 'last.pt'
    best, last = (torch.load(p, map_location='cpu', weights_only=True) for p in (best_path, last_path))
    rows = [json.loads(line) for line in (run / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = max(rows, key=lambda row: row['validation']['all']['foreground_iou'])
    if (config != planned or [r['epoch'] for r in rows] != list(range(1, 31)) or last['epoch'] != 30
            or environment['initial_shared_state_sha256'] != INIT or (run / 'resume.json').exists()
            or any(r['optimizer_steps_per_epoch'] != per_epoch or r['global_optimizer_step'] != r['epoch'] * per_epoch
                   or r['validation']['all']['images'] != val_count
                   or not all(math.isfinite(x) for x in r['loss'].values()) for r in rows)
            or best['epoch'] != selected['epoch']
            or best['validation_foreground_iou'] != selected['validation']['all']['foreground_iou']
            or last['global_step'] != 30 * per_epoch
            or any(c['manifest_sha256'] != {'train': train_hash, 'val': val_hash}
                   or c['config'] != config or c['initial_shared_state_sha256'] != INIT
                   or c['encoder_resolved_revision'] != config['encoder_revision'] for c in (best, last))):
        raise ValueError(f'Training controls, checkpoint or selection mismatch: {stage}')
    if config['boundary']:
        # Adam's second moment is positive only when the parameter received a nonzero gradient.
        model = TransparentSegformer(boundary=True, pretrained=False)
        names = [name for name, _ in model.named_parameters()]
        indices = last['optimizer']['param_groups'][0]['params']
        if len(names) != len(indices):
            raise ValueError('Model tensors do not map to the optimizer parameter list')
        moments = [last['optimizer']['state'][index]['exp_avg_sq'] for name, index in zip(names, indices)
                   if name.startswith('boundary.')]
        if (not moments or not any(torch.count_nonzero(t).item() for t in moments)
                or not all(r['loss']['boundary'] > 0 for r in rows)):
            raise ValueError(f'Boundary head gradients or loss missing: {stage}')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('environment.json', 'config.json', 'history.jsonl', 'pip_freeze.txt'):
        shutil.copy2(run / name, output / name)
    return {'selected_epoch': best['epoch'], 'selected_validation_iou': best['validation_foreground_iou'],
            'optimizer_steps': last['global_step'], 'train_count': train_count,
            'validation_count': val_count, 'training_loop_seconds': sum(r['seconds'] for r in rows),
            'peak_allocated_mib': max(r['peak_cuda_memory_mb'] for r in rows), 'environment': environment,
            'best_sha256': sha256_file(best_path), 'last_sha256': sha256_file(last_path),
            'boundary_nonzero_gradient_evidence': bool(config['boundary'])}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_b04(run_root, output, c_metrics):
    baseline = Path('reports/member2_full_A_20261114')
    a = read(baseline / 'summary.json')
    if a['evaluations']['test']['manifest_sha256'] != V1_TEST:
        raise ValueError('Unexpected A test membership')
    with c_metrics.open(encoding='utf-8-sig', newline='') as stream:
        c = {r['subset']: {k: float(r[k]) for k in AB_METRICS} | {'images': int(r['images'])}
             for r in csv.DictReader(stream) if r['model'] == 'C' and r['condition'] == 'clean'}
    scores = {'A': a['evaluations']['test']['scores'], 'C': c}
    training = {}
    for model in ('B', 'D'):
        stage = f'B03_{model}'
        run, target = run_root / stage, output / stage
        training[model] = report_training(run, stage, False, target)
        eval_dir = run / 'evaluation'
        identity = read(eval_dir / 'identity.json')
        result = read(eval_dir / 'clean/metrics.json')
        if (not result['complete'] or result['condition'] != 'clean' or result['manifest_sha256'] != V1_TEST
                or identity['checkpoint_sha256'] != training[model]['best_sha256']):
            raise ValueError(f'Incomplete or mismatched main test: {model}')
        for name in ('identity.json', 'clean/metrics.json', 'clean/per_image.csv'):
            shutil.copy2(eval_dir / name, target / Path(name).name)
        scores[model] = result['scores']
    result_rows, effects = [], []
    for subset, count in (('all', 4428), ('easy', 3491), ('hard', 937)):
        if any(scores[model][subset]['images'] != count for model in 'ABCD') or any(
                scores[model][subset]['boundary_images'] != count for model in 'ABD'):
            raise ValueError('Subgroup sizes differ across A/B/C/D')
        for model in 'ABCD':
            result_rows.append({'model': model, 'subset': subset, 'images': count,
                                **{metric: scores[model][subset][metric] for metric in AB_METRICS}})
        differences = {}
        for pair, base, method in [('B_minus_A', 'A', 'B'), ('D_minus_C', 'C', 'D'),
                                   ('C_minus_A', 'A', 'C'), ('D_minus_B', 'B', 'D')]:
            differences[pair] = {key: 100 * (scores[method][subset][key] - scores[base][subset][key])
                                 for key in AB_METRICS}
        differences['interaction'] = {metric: differences['D_minus_C'][metric] - differences['B_minus_A'][metric]
                                      for metric in AB_METRICS}
        effects.extend({'contrast': pair, 'subset': subset, **metrics} for pair, metrics in differences.items())
    write_csv(output / 'B04_clean_comparison.csv', result_rows)
    write_csv(output / 'B04_effects_percentage_points.csv', effects)
    write_json(output / 'B03_B04_summary.json', {'training': training, 'clean_test': result_rows,
                'paired_effects_percentage_points': effects, 'c_table_sha256': sha256_file(c_metrics),
                'c_source_commit': '3d11d48397840443d7274d65dd03c44c7b73a0bf',
                'main_manifest_sha256': V1_TEST, 'scope': 'single seed, clean test'})


def summarize_e03(run_root, output):
    result_rows, recall_rows, training = [], [], {}
    for model in ('A', 'D'):
        stage = f'E03_{model}'
        run, target = run_root / stage, output / stage
        training[model] = report_training(run, stage, True, target)
        eval_dir = run / 'evaluation'
        result = read(eval_dir / 'result.json')
        if not result['complete'] or result['identity']['checkpoint_sha256'] != training[model]['best_sha256']:
            raise ValueError(f'Mismatched E03 selected checkpoint: {model}')
        for name in ('result.json', 'per_image.csv'):
            shutil.copy2(eval_dir / name, target / name)
        for subset, count in (('all', 4428), ('seen', 2935), ('heldout', 1493)):
            metrics = result['scores'][subset]
            if metrics['images'] != count or metrics['boundary_images'] != count:
                raise ValueError('Incomplete E03 evaluation')
            result_rows.append({'model': model, 'subset': subset, **metrics})
            recall_rows.extend({'model': model, 'subset': subset, 'category_id': int(cid), **row}
                               for cid, row in result['category_recall'][subset].items())
    for subset in ('all', 'seen', 'heldout'):
        totals = [[r['total'] for r in recall_rows if r['model'] == model and r['subset'] == subset]
                  for model in ('A', 'D')]
        if totals[0] != totals[1]:
            raise ValueError(f'Category denominators differ between isolated A/D: {subset}')
    write_csv(output / 'E03_original_size_metrics.csv', result_rows)
    write_csv(output / 'E03_category_recall.csv', recall_rows)
    write_json(output / 'E03_summary.json', {'training': training, 'metrics': result_rows,
                                           'category_recall': recall_rows, 'heldout_class_ids': [5, 8]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('b04', 'e03'))
    parser.add_argument('--c-metrics', type=Path)
    args = parser.parse_args()
    if args.stage == 'b04':
        if not args.c_metrics:
            parser.error('--c-metrics must point to the verified Member4 condition_results.csv')
        summarize_b04(args.runs, args.output, args.c_metrics)
    else:
        summarize_e03(args.runs, args.output)
