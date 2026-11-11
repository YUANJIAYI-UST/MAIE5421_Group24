"""Download the author-linked V2 ZIP, preserving pinned code provenance."""
import argparse
import json
from pathlib import Path

from download_trans10k_v1 import download

FILE_ID = '1YzAAMY8xfL9BMTIDU-nFC3dcGbSIBPu5'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/archives/Trans10K_v2')
    args = parser.parse_args()
    output = Path(args.output)
    source = output / 'author_code/README.md'
    if not source.is_file() or FILE_ID not in source.read_text():
        raise ValueError('Save and verify the author README and revision before downloading')
    info = download('Trans10K_v2', FILE_ID, output)
    record = {'source_repository': 'https://github.com/xieenze/Trans2Seg',
              'author_revision': json.loads((output/'author_revision.json').read_text())['revision'],
              'archive': info, 'zip_crc_and_dataset_audit': 'separate steps'}
    (output/'download_provenance.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
