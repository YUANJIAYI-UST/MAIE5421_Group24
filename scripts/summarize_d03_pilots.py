"""Export matched-budget D03 pilot evidence, excluding images and weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.runs), Path(args.output)
    rows, checkpoints, environments, configs = [], {}, {}, {}
    for name in ('A', 'D'):
        source = root / name
        history = [json.loads(line) for line in (source / 'history.jsonl').read_text().splitlines()]
        assert len(history) == 1 and history[0]['global_optimizer_step'] == 20
        assert history[0]['validation']['all']['images'] == 646
        configs[name] = json.loads((source / 'config.json').read_text())
        environments[name] = json.loads((source / 'environment.json').read_text())
        rows.append({'model': name, **history[0]})
        checkpoints[name] = {}
        for file in ('best.pt', 'last.pt'):
            digest = hashlib.sha256()
            with (source / file).open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            checkpoints[name][file] = digest.hexdigest()
    assert environments['A']['initial_shared_state_sha256'] == environments['D']['initial_shared_state_sha256']
    differences = {key for key in configs['A'].keys() | configs['D'].keys() if configs['A'].get(key) != configs['D'].get(key)}
    assert differences == {'ablation', 'boundary', 'augmentation'}
    output.mkdir(parents=True, exist_ok=False)
    for name in ('A', 'D'):
        (output / name).mkdir()
        for file in ('config.json', 'environment.json', 'history.jsonl', 'pip_freeze.txt'):
            shutil.copy2(root / name / file, output / name / file)
    for file in ('v2_import.json', 'd03_verification.json'):
        shutil.copy2(root / file, output / file)
    summary = {'purpose': 'pilot_only', 'same_budget': True, 'same_initial_shared_hash': True,
               'initial_shared_state_sha256': environments['A']['initial_shared_state_sha256'],
               'test_set_model_evaluation': 'not_run', 'rows': rows, 'checkpoint_sha256': checkpoints}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'resources.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'steps', 'train_loss', 'boundary_loss', 'validation_IoU', 'validation_F1',
                         'validation_BIoU', 'steady_step_seconds', 'train_seconds', 'validation_seconds', 'peak_allocated_MiB'])
        for r in rows:
            scores = r['validation']['all']
            writer.writerow([r['model'], r['global_optimizer_step'], r['loss']['total'], r['loss']['boundary'],
                scores['foreground_iou'], scores['region_f1'], scores['boundary_iou'], r['steady_microbatch_seconds'],
                r['train_seconds'], r['validation_seconds'], r['peak_cuda_memory_mb']])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
