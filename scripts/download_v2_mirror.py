"""Pin a public V2 mirror for inspection; it is not an author-certified archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

REPO = 'rdyzakya/Trans10K-v2'
REVISION = '3445e93f1e507ea6c2dd36f06badfd5e818b699f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', nargs='+', choices=['train', 'validation', 'test'], required=True)
    args = parser.parse_args()
    root = Path('data/archives/Trans10K_v2_mirror')
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true', timeout=40)
    response.raise_for_status()
    files = response.json()
    (root/'files.json').write_text(json.dumps(files, indent=2)+'\n')
    card = requests.get(f'https://huggingface.co/datasets/{REPO}/raw/{REVISION}/README.md', timeout=40)
    card.raise_for_status()
    (root/'README.md').write_bytes(card.content)

    def get(info):
        dest = root / info['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix('.part')
        if not dest.exists():
            url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{info["path"]}'
            with requests.get(url, stream=True, timeout=(30, 90)) as r:
                r.raise_for_status()
                with temp.open('wb') as out:
                    for chunk in r.iter_content(2**20):
                        out.write(chunk)
            if temp.stat().st_size != info['size']:
                raise ValueError('Mirror download size mismatch')
            temp.rename(dest)
        digest = hashlib.sha256()
        with dest.open('rb') as f:
            for block in iter(lambda: f.read(8*2**20), b''):
                digest.update(block)
        if digest.hexdigest() != info['lfs']['oid']:
            raise ValueError('Mirror SHA256 mismatch')
        print(f'Verified {info["path"]}: {digest.hexdigest()}', flush=True)
        return {'path': info['path'], 'sha256': digest.hexdigest(), 'bytes': dest.stat().st_size}

    selected = [x for x in files if any(x['path'].startswith(f'data/{s}-') for s in args.split)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(get, selected))
    (root/('download_'+'_'.join(args.split)+'.json')).write_text(json.dumps({
        'repository': REPO, 'revision': REVISION, 'source_type': 'third_party_mirror',
        'original_author_archive_equivalence': 'unverified', 'files': records}, indent=2)+'\n')


if __name__ == '__main__':
    main()
