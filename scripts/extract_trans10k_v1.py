"""Safely extract complete official ZIPs, preserving data files and split layout."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
from zipfile import ZipFile


def extract(split, archives, root):
    source = archives / f"{split}.zip"
    destination = root / split
    stage = root / f".extract_{split}"
    if destination.exists() or stage.exists():
        raise FileExistsError(f"Refusing to overwrite extracted/staged {split} data")
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    stage.mkdir()
    skipped, suffixes, total, size = [], Counter(), 0, 0
    with ZipFile(source) as archive:
        seen = set()
        infos = archive.infolist()
        required_space = sum(i.file_size for i in infos)
        if shutil.disk_usage(root).free < required_space + 1024**3:
            raise OSError("Insufficient disk space for extraction plus 1 GiB reserve")
        for info in infos:
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename
                    or not path.parts or stat.S_ISLNK(info.external_attr >> 16)):
                raise ValueError(f"Unsafe archive member: {info.filename}")
            if info.filename in seen:
                raise ValueError(f"Duplicate archive member: {info.filename}")
            seen.add(info.filename)
            metadata = (path.parts[0] == "__MACOSX" or path.name == ".DS_Store"
                        or path.name.startswith("._"))
            if not metadata and path.parts[0] != split:
                raise ValueError(f"Unexpected layout: {info.filename}")
            if info.is_dir():
                continue
            # Reading every file to EOF also checks its ZIP CRC, including metadata.
            if metadata:
                with archive.open(info) as stream:
                    while stream.read(1024 * 1024):
                        pass
                skipped.append(info.filename)
                continue
            target = stage.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            if target.stat().st_size != info.file_size:
                raise IOError(f"Extracted size mismatch: {info.filename}")
            total += 1
            size += info.file_size
            suffixes[path.suffix.lower()] += 1
            if total % 1000 == 0:
                print(f"{split}: extracted {total} files", flush=True)
    (stage / split).rename(destination)
    stage.rmdir()
    record = {"split": split, "archive": source.name, "archive_sha256": digest.hexdigest(),
              "zip_crc_check": "passed for every file member",
              "data_files_extracted": total, "data_bytes": size, "extensions": dict(suffixes),
              "metadata_excluded": skipped,
              "data_exclusions": [], "official_split_preserved": True}
    (archives / f"extraction_{split}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "metadata_excluded"}), flush=True)
    print(f"{split}: excluded {len(skipped)} macOS metadata entries only", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", default="data/archives/Trans10K_v1")
    parser.add_argument("--root", default="data/Trans10K")
    parser.add_argument("--split", choices=("train", "validation", "test"), nargs="+",
                        default=["train", "validation", "test"])
    args = parser.parse_args()
    for split in args.split:
        extract(split, Path(args.archives), Path(args.root))


if __name__ == "__main__":
    main()
