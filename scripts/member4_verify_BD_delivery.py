"""Verify the released main-experiment B/D models before member 4 evaluation."""
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

ROOT = Path('runs/member4_BD_delivery_20261118')
SOURCE = Path('tmp/member4_completion_20261118/member2_source')
OUT = Path('runs/member4_completion_20261118')
INIT = 'fd50559254d2d5fe9b16cbaa6f7fd32452faa4a7f3b9a3ec42f1a6fb6eedc835'
EXPECTED = {
    'B': 'e219d6bf2f7f7cc57711d8df7354202efd81360108247f16dd4cc72765549d81',
    'D': '49498d53b63b13368ceafa5200842a7136c41172c9decad6000c23d1c22b93a4',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    common = read('configs/member4_full_C_20261115.json')
    comparison_keys = ['purpose', 'train_manifest', 'val_manifest', 'size', 'pretrained',
        'encoder_revision', 'seed', 'batch_size', 'epochs', 'lr', 'weight_decay', 'amp',
        'amp_dtype', 'deterministic', 'gradient_accumulation_steps', 'validation_interval',
        'validation_metrics', 'data_protocol_id', 'expected_manifest_sha256', 'expected_versions']
    hashes = {key: sha(common[file]) for key, file in [('train','train_manifest'),('val','val_manifest')]}
    summary = read(ROOT/'B03_B04_summary.json')
    assets = read(ROOT/'model_assets.json')['assets']
    for item in assets:
        file = ROOT/item['name']
        if file.exists():
            require(file.stat().st_size == item['bytes'] and sha(file) == item['sha256'],
                    'Release asset differs: '+item['name'])
    # These three files define the architecture, augmentation and training losses.
    source_equal = {}
    for name in ['transparent_seg/data.py','transparent_seg/model.py','transparent_seg/losses.py']:
        source_equal[name] = Path(name).read_text(encoding='utf-8') == (SOURCE/name).read_text(encoding='utf-8')
        require(source_equal[name], 'Shared implementation differs: '+name)
    verified = {}
    for model in ['B','D']:
        folder = ROOT/'full_runs'/('B03_'+model)
        config = read(folder/'config.json')
        history = [json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        environment = read(folder/'environment.json')
        require(sha(folder/'best.pt') == EXPECTED[model], 'Best checkpoint SHA mismatch')
        require(sha(folder/'last.pt') == summary['training'][model]['last_sha256'], 'Last checkpoint SHA mismatch')
        require(all(config.get(k) == common.get(k) for k in comparison_keys), 'A/C common controls differ')
        require(config['boundary'] is True and config['augmentation'] == (model == 'D')
                and config['boundary_radius'] == 2 and config['boundary_weight'] == 0.1
                and config['ablation'] == model, 'Wrong B/D switches or boundary settings')
        protocol_path = SOURCE/config['frozen_protocol']
        require(sha(protocol_path) == config['frozen_protocol_sha256'], 'Frozen protocol hash mismatch')
        protocol = read(protocol_path)
        require(protocol['config'] == {k:v for k,v in config.items()
                                      if k not in ['frozen_protocol','frozen_protocol_sha256']},
                'Actual config differs from frozen protocol')
        require([r['epoch'] for r in history] == list(range(1,31)), 'Incomplete epoch history')
        current_best = -math.inf
        for epoch, row in enumerate(history,1):
            val = row['validation']['all']
            require(val['images'] == 995 and val['skipped_all_void'] == 0, 'Incomplete full validation')
            require(row['global_optimizer_step'] == epoch*620 and row['optimizer_steps_per_epoch'] == 620
                    and row['effective_batch_size'] == 8, 'Incomplete training budget')
            selected = val['foreground_iou'] > current_best
            require(row['selected'] == selected, 'Selection flags do not follow validation-only best IoU')
            current_best = max(current_best,val['foreground_iou'])
        chosen = max(history, key=lambda row: row['validation']['all']['foreground_iou'])
        best = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        last = torch.load(folder/'last.pt', map_location='cpu', weights_only=True)
        for checkpoint in (best,last):
            require(checkpoint['config'] == config, 'Checkpoint config mismatch')
            require(checkpoint['manifest_sha256'] == hashes, 'Checkpoint data hashes differ')
            require(checkpoint['initial_shared_state_sha256'] == INIT, 'Shared initialization differs')
            require(checkpoint['encoder_resolved_revision'] == common['encoder_revision'], 'Encoder revision differs')
        require(last['epoch'] == 30 and last['global_step'] == 18600, 'Final checkpoint budget incomplete')
        require(best['epoch'] == chosen['epoch'] and best['global_step'] == best['epoch']*620
                and best['validation_foreground_iou'] == chosen['validation']['all']['foreground_iou'],
                'Best checkpoint was not the validation-selected epoch')
        require(environment['initial_shared_state_sha256'] == INIT, 'Environment initialization differs')
        # The same local inference architecture must load all weights strictly.
        from transparent_seg.model import TransparentSegformer
        loaded = TransparentSegformer(boundary=True,pretrained=False)
        loaded.load_state_dict(best['model'],strict=True)
        verified[model] = {'checkpoint':str(folder/'best.pt'),'sha256':EXPECTED[model],
            'selected_epoch':best['epoch'],'selected_validation_iou':best['validation_foreground_iou'],
            'last_epoch':last['epoch'],'optimizer_steps':last['global_step'],
            'shared_initialization_sha256':INIT,'manifest_sha256':hashes,
            'strict_model_load':True,'validation_selection_reconstructed':True,
            'training_gpu':environment['gpu'],'training_commit':environment['git_commit'],
            'frozen_protocol_sha256':sha(protocol_path),'passed':True}
        del loaded,best,last
    OUT.mkdir(parents=True,exist_ok=True)
    result = {'passed':True,'models':verified,'shared_source_matches':source_equal,
        'member2_head':read(ROOT/'download_receipt.json')['member2_head'],
        'release':read(ROOT/'download_receipt.json')['release'],
        'C_settings_reference':'configs/member4_full_C_20261115.json',
        'limitations':['Single seed; training hardware/Python environments differ.',
                      'Training records establish reproducibility and selection consistency; they are not independent observation of the original training.'],
        'E03_models_used':False}
    (OUT/'intake_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
