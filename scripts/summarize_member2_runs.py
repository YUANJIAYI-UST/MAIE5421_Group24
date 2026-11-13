"""Publish small, traceable pilot/overfit evidence without data or model weights."""
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
    output.mkdir(parents=True, exist_ok=False)
    resources = json.loads((root / 'pilot/resources.json').read_text())
    shutil.copy2(root / 'pilot/resources.json', output / 'resources.json')
    shutil.copy2(root / 'v1_verification.json', output / 'v1_verification.json')
    provenance = {}
    for name in ('pilot/batch4', 'pilot/batch8', 'overfit'):
        source = root / name
        destination = output / name
        destination.mkdir(parents=True)
        for file in ('config.json', 'environment.json', 'pip_freeze.txt', 'history.jsonl'):
            if (source / file).exists():
                shutil.copy2(source / file, destination / file)
        provenance[name] = {}
        for file in ('best.pt', 'last.pt'):
            if (source / file).exists():
                digest = hashlib.sha256()
                with (source / file).open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                provenance[name][file] = {'sha256': digest.hexdigest(), 'bytes': (source / file).stat().st_size}
    (output / 'checkpoint_hashes.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    history = [json.loads(line) for line in (root / 'overfit/history.jsonl').read_text().splitlines()]
    with (output / 'overfit_curve.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'train_loss', 'train_iou', 'validation_iou', 'lr'])
        for row in history:
            writer.writerow([row['epoch'], row['loss']['total'], row['overfit_training']['all']['foreground_iou'],
                row['validation']['all']['foreground_iou'] if row['validation'] else '', row['lr']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot([r['epoch'] for r in history], [r['loss']['total'] for r in history])
    axes[0].set(xlabel='Epoch', ylabel='BCE + Dice', title='Four-image training loss')
    axes[1].plot([r['epoch'] for r in history], [r['overfit_training']['all']['foreground_iou'] for r in history], label='Training subset')
    validated = [r for r in history if r['validation']]
    axes[1].plot([r['epoch'] for r in validated], [r['validation']['all']['foreground_iou'] for r in validated], 'o-', label='Full validation')
    axes[1].set(xlabel='Epoch', ylabel='Foreground IoU', ylim=(0, 1), title='Diagnostic only; no test-set evaluation')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / 'overfit_curve.png', dpi=160)
    plt.close(fig)
    with (output / 'resource_table.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['batch_size', 'status', 'steady_step_seconds', 'train_seconds_20_steps', 'full_validation_seconds', 'peak_allocated_MiB', 'peak_reserved_MiB'])
        for row in resources['rows']:
            writer.writerow([row['batch_size'], row['status'], *[row.get(k) for k in (
                'steady_microbatch_seconds', 'train_seconds', 'validation_seconds', 'peak_cuda_memory_mb', 'peak_cuda_reserved_mb')]])
    print(output)


if __name__ == '__main__':
    main()
