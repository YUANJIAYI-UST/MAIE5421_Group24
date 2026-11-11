"""Create one owner-complete handoff, with immutable inputs and filled workbooks."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.prepare import sha256_file, write_json
from transparent_seg.prepare_final import load


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'runs/member1_final_20261029'
    release = root / 'manifests/member1_final_20261029'
    archive = root / 'reports/Member1_Final_D01_D02_D03_20261029.zip'
    if archive.exists():
        raise FileExistsError('Preserve existing final archive')
    structural = load(run / 'structural_verification.json')
    v1 = load(run / 'v1_verification_replay.json')
    d03 = load(run / 'd03_verification.json')
    assert structural['status'] == d03['status'] == 'passed'
    assert v1['source_files_unchanged'] and v1['byte_identical_regeneration']
    assert v1['cpu_diagnostic']['finite_gradients']
    assert all(d03['byte_identical_reproduction'].values())
    assert load(run / 'tests.json')['passed'] == 61
    policy = load(release / 'policy.json')
    for name, digest in policy['input_sha256'].items():
        assert sha256_file(root / name) == digest
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'Pillow', 'scipy', 'pytest', 'pyarrow')},
                   'git_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   'delivery_includes_uncommitted_work': True,
                   'remote_checked': 'origin/main matches local base after git fetch on 2026-10-29',
                   'member1_processing_complete': True, 'member5_review': 'pending'}
    write_json(run / 'environment.json', environment)
    (run / 'pip_freeze.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    files = {}

    def include(relative):
        p = root / relative
        if p.is_dir():
            for child in sorted(p.rglob('*')):
                if child.is_file() and '__pycache__' not in child.parts and child.name != '.DS_Store':
                    files[child.relative_to(root).as_posix()] = child
        elif p.is_file():
            files[p.relative_to(root).as_posix()] = p
        else:
            raise FileNotFoundError(p)

    for name in ('transparent_seg', 'tests', 'scripts', 'configs', 'docs',
                 'manifests/member1_final_20261029', 'manifests/trans10k_v1_review_20261028',
                 'manifests/trans10k_d03_20261028', 'data/Trans10K_v1_review_20261028',
                 'requirements.txt', 'requirements-data.txt', 'pyproject.toml', 'README.txt', '.gitignore',
                 'runs/trans10k_d03_20261028/selection'):
        include(name)
    for name in policy['input_sha256']:
        include(name)
    for name in ('audit.json',):
        include('runs/trans10k_data_audit_20261022/diagnostics/' + name)
    for name in ('selection_freeze_order.json', 'verification.json', 'ai_visual_review.json'):
        include('runs/trans10k_d03_20261028/' + name)
    for name in ('structural_verification.json', 'v1_verification_replay.json', 'd03_verification.json',
                 'tests.json', 'pilot_config_verification.json', 'environment.json', 'pip_freeze.txt'):
        include('runs/member1_final_20261029/' + name)
    for name in ('D02_OWNER_CONFIRMATION_20261029.txt', 'D02_AI_SECOND_REVIEW_20261029.json',
                 'D02_AI_SECOND_REVIEW_20261029.txt', 'MEMBER1_FLAGGED_MASK_LINEAGE_20261028.json',
                 'MEMBER1_FINAL_DELIVERY_20261029.txt'):
        include('reports/' + name)
    for split in ('train', 'validation'):
        base = root / 'runs/trans10k_data_audit_20261022' / f'{split}_review'
        include((base / 'review_samples.json').relative_to(root))
        for p in base.glob('*_contact_*.jpg'):
            include(p.relative_to(root))
    gallery = root / 'runs/trans10k_member1_20261028/review_gallery'
    for pattern in ('*contact*.jpg', 'train_images_*.jpg.jpg'):
        for p in gallery.glob(pattern):
            include(p.relative_to(root))
    for p in (root / 'runs/trans10k_d03_20261028/human_review').glob('*contact*.jpg'):
        include(p.relative_to(root))
    for name in ('README.md', 'files.json', 'download_train_validation.json', 'download_test.json'):
        include('data/archives/Trans10K_v2_mirror/' + name)
    for name in ('author_revision.json', 'author_code/README.md', 'author_code/LICENSE',
                 'author_code/segmentron/data/dataloader/transparent11.py', 'author_code/segmentron/utils/visualize.py'):
        include('data/archives/Trans10K_v2/' + name)
    owner = load(root / policy['inputs']['owner_decisions'])
    for workbook in owner['source_workbooks']:
        p = Path(workbook['path'])
        assert sha256_file(p) == workbook['sha256']
        files['review_workbooks/' + p.name] = p
    source_hashes = {name: sha256_file(p) for name, p in sorted(files.items())
                     if name.startswith(('transparent_seg/', 'scripts/', 'tests/', 'configs/'))}
    write_json(run / 'source_code_hashes.json', source_hashes)
    include('runs/member1_final_20261029/source_code_hashes.json')
    start = '''成员 1 最终交付：D01 + D02 + D03（2026-10-29）
先看 reports/MEMBER1_FINAL_DELIVERY_20261029.txt，再看 docs/12_MEMBER1_FINAL_HANDOFF.txt。
成员 1 的处理和交付已完成；成员 5 的独立复核尚未发生。
当前数据清单只使用 manifests/member1_final_20261029/；旧日期文件是保留的历史证据。
本包集齐代码、清单、审计、类别统计、排除理由、人工填写 Excel、图片复核材料、测试和复现记录。
原始大数据与模型权重不在包内。仅重建清单无需大数据；验证实际标签需准备相同来源的数据。
61 项测试通过；所有 V1 原文件未变；D03 train/val 两类标注为零；完整 test 4428 张保留。
请勿把旧候选报告中的 pending 当作新的待填任务，也不要把本包当作成员 5 已签字。
V2 镜像与作者原包的一致性未证明、同场景不同照片保留、第 141 对身份不确定，均已明示。
'''
    # Verify a fresh extracted code/input tree can regenerate the release.
    # Raw datasets and the development checkout are deliberately not used.
    with tempfile.TemporaryDirectory(prefix='member1-package-') as temp:
        stage = Path(temp)
        for name, path in files.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
        build = subprocess.run([sys.executable, '-m', 'transparent_seg.prepare_final',
                                '--output', 'manifests/member1_package_replay'],
                               cwd=stage, text=True, capture_output=True, check=True)
        verified = subprocess.run([sys.executable, 'scripts/verify_member1_final.py',
                                   '--release', 'manifests/member1_final_20261029',
                                   '--replay', 'manifests/member1_package_replay',
                                   '--output', 'runs/package_replay_check.json'],
                                  cwd=stage, text=True, capture_output=True, check=True)
        check = load(stage / 'runs/package_replay_check.json')
        assert check['byte_identical_reproduction_files'] == 22
        imported = subprocess.check_output([sys.executable, '-c',
                    'import transparent_seg.prepare_final as p; print(p.__file__)'], cwd=stage, text=True).strip()
        assert Path(imported).resolve().is_relative_to(stage.resolve()), imported
        package_replay = {'status': 'passed', 'fresh_directory': True, 'raw_datasets_required_for_rebuild': False,
                          'development_source_used': False, 'regenerated_files': 22,
                          'same_interpreter_used': True, 'other_member_machine_tested': False,
                          'structural_check': check}
        write_json(run / 'package_replay_verification.json', package_replay)
    include('runs/member1_final_20261029/package_replay_verification.json')
    checks = {name: sha256_file(p) for name, p in sorted(files.items())}
    checks['START_HERE.txt'] = hashlib.sha256(start.encode()).hexdigest()
    with ZipFile(archive, 'x', ZIP_DEFLATED, compresslevel=6) as z:
        for name, path in sorted(files.items()):
            z.write(path, name)
        z.writestr('START_HERE.txt', start)
        z.writestr('CHECKSUMS.json', json.dumps(checks, ensure_ascii=False, indent=2) + '\n')
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(checks) | {'CHECKSUMS.json'}
        for name, digest in checks.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    digest = sha256_file(archive)
    archive.with_suffix('.sha256.txt').write_text(f'{digest}  {archive.name}\n')
    result = {'archive': str(archive.relative_to(root)), 'sha256': digest, 'bytes': archive.stat().st_size,
              'files': len(checks) + 1, 'zip_crc': 'passed', 'all_payload_hashes': 'passed',
              'fresh_directory_reproduction': 'passed', 'member1_processing': 'complete', 'member5_review': 'pending'}
    write_json(run / 'handoff_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
