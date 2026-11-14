"""Verify Release downloads and reassemble original ZIP archives (does not extract/overwrite)."""
import argparse
import hashlib
import json
from pathlib import Path


def restore(assets, output):
    assets, output = Path(assets).resolve(), Path(output).resolve()
    if assets == output:
        raise ValueError("Use a separate archive output directory")
    index = json.loads((assets / "dataset_assets.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    for artifact in index["artifacts"]:
        target = (output / artifact["archive"]).resolve()
        if not target.is_relative_to(output) or target.exists():
            raise ValueError(f"Unsafe or existing output: {target}")
        whole, total = hashlib.sha256(), 0
        with target.open("xb") as sink:
            for part in artifact["parts"]:
                source = (assets / part["name"]).resolve()
                if not source.is_relative_to(assets) or source.stat().st_size != part["bytes"]:
                    raise ValueError("Invalid part path/size")
                h = hashlib.sha256()
                with source.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
                        whole.update(chunk)
                        total += len(chunk)
                        sink.write(chunk)
                if h.hexdigest() != part["sha256"]:
                    raise ValueError(f"Part hash mismatch: {source.name}; discard incomplete output")
        if whole.hexdigest() != artifact["sha256"] or total != artifact["bytes"]:
            raise ValueError("Reassembled archive mismatch")
        print(f"Verified {target.name}: {total} bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    restore(args.assets, args.output)
