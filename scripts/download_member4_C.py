"""Download the published C checkpoint and optionally its verified run records."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.request import Request, urlopen
import zipfile

CATALOG = Path(__file__).resolve().parents[1] / 'reports/member4_C_release_20261116/model_assets.json'
URL_PREFIX = 'https://github.com/YUANJIAYI-UST/MAIE5421_Group24/releases/download/member4-c-augmentation-20261116/'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset, output):
    name = asset['name']
    if Path(name).name != name or '/' in name or '\\' in name or not asset['url'].startswith(URL_PREFIX):
        raise ValueError('Unexpected published asset path or URL')
    target = output / name
    if target.exists():
        if target.is_file() and target.stat().st_size == asset['bytes'] and sha256(target) == asset['sha256']:
            print('Already verified: ' + name, flush=True)
            return target
        raise ValueError('Existing file differs; choose a new output directory: ' + str(target))
    print(f"Downloading {name} ({asset['bytes'] / 2**20:.1f} MiB)", flush=True)
    request = Request(asset['url'], headers={'User-Agent': 'MAIE5421-Group24-model-download'})
    temp = None
    try:
        with urlopen(request, timeout=60) as response, tempfile.NamedTemporaryFile(
                mode='wb', dir=output, prefix=name + '.', suffix='.part', delete=False) as stream:
            temp = Path(stream.name)
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(chunk)
        if temp.stat().st_size != asset['bytes'] or sha256(temp) != asset['sha256']:
            raise ValueError('Download hash/size mismatch: ' + name)
        if target.exists():
            raise FileExistsError('Output appeared during download: ' + str(target))
        os.rename(temp, target)
        print('Verified: ' + name, flush=True)
        return target
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def extract_records(archive, destination):
    # The ZIP is pinned by the committed catalog. Also reject paths escaping the output.
    if destination.exists():
        raise FileExistsError('Records directory already exists; do not overwrite: ' + str(destination))
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive paths')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive member')
        inventory = json.loads(bundle.read('files_sha256.json'))
        if set(names) != set(inventory) | {'files_sha256.json'}:
            raise ValueError('Archive inventory differs')
        for name, expected in inventory.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise ValueError('Archive member hash differs: ' + name)
        destination.mkdir()
        for name in names:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(bundle.read(name))
    print(f'Verified and extracted {len(inventory)} records to {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='runs/member4_C_release_20261116')
    parser.add_argument('--records', action='store_true', help='Also download and unpack the run-record archive')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    records = output / 'records'
    if args.records and records.exists():
        raise FileExistsError('Use a new output directory to avoid overwriting existing records')
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    assets = {row['name']: row for row in catalog['artifacts']}
    required = ['best.pt', 'MODEL_README.txt']
    if args.records:
        required.append('member4_C_seed42_records.zip')
    output.mkdir(parents=True, exist_ok=True)
    for name in required:
        download(assets[name], output)
    if args.records:
        extract_records(output / 'member4_C_seed42_records.zip', records)
    print('Checkpoint ready: ' + str(output / 'best.pt'))


if __name__ == '__main__':
    main()
