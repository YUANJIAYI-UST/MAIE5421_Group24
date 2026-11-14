"""Validate a V2 mask supplement against frozen member-1 inventories, then copy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.prepare import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, repo = Path(args.source).resolve(), Path('.').resolve()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    policy = json.loads((repo / 'manifests/member1_final_20261029/policy.json').read_text())
    expected = {}
    for key in ('v2_train_val_inventory', 'v2_test_inventory'):
        relative = policy['inputs'][key]
        assert sha256_file(repo / relative) == policy['input_sha256'][relative]
        for row in json.loads((repo / relative).read_text())['samples']:
            name = 'data/Trans10K_v2_mirror_3445e93/' + row['v2_mask']
            assert name not in expected
            expected[name] = row['v2_mask_sha256']
    declared = json.loads((source / 'V2_MASKS_SHA256_20261114.json').read_text(encoding='utf-8'))
    assert declared == expected, 'Supplement differs from frozen inventory'
    assert {p.relative_to(source).as_posix() for p in (source / 'data').rglob('*.png')} == set(expected)
    counts = Counter(Path(name).parts[2] for name in expected)
    assert counts == {'train': 5000, 'validation': 1000, 'test': 4428}
    # Validate the entire input and existing destinations before writing any masks.
    for name, digest in expected.items():
        incoming, target = (source / name).resolve(), (repo / name).resolve()
        assert incoming.is_relative_to(source) and target.is_relative_to(repo / 'data')
        assert sha256_file(incoming) == digest, name
        if target.exists():
            assert sha256_file(target) == digest, f'Existing destination differs: {name}'
    copied = 0
    for name, digest in expected.items():
        target = repo / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with (source / name).open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            copied += 1
        assert sha256_file(target) == digest, name
    result = {'status': 'passed', 'source_package': source.name,
              'mirror_revision': '3445e93f1e507ea6c2dd36f06badfd5e818b699f',
              'counts': dict(counts), 'files_verified': len(expected), 'files_copied': copied,
              'matched_frozen_inventories': True,
              'supplement_checksum_file_sha256': sha256_file(source / 'V2_MASKS_SHA256_20261114.json'),
              'v1_binary_targets_modified': False,
              'author_archive_equivalence': 'not_verified'}
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
