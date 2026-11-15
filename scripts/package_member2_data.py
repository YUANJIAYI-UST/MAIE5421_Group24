"""Package authorized course data for GitHub Release without adding binary blobs to Git."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset_assets.json").exists():
        raise ValueError("Package already exists; verify it instead of replacing it")
    artifacts = []
    expected = {"train.zip": "5c388786af167c640f472c455a7fd1851d591632f1d419873cf4a8070eb99869",
                "validation.zip": "c2edba4ba966445272d245699848b16a1427ef5bc5140fbeb15c4fedf41a046c",
                "test.zip": "514d3788091058e41ad226a9d3a99ca29523150403f9e5a1efc99b23e1bd88bb",
                "Member1_Final_D01_D02_D03_20261029.zip": "b67ba5f5b8c3eccb3b65fe184af78bbf8febfddb6a9dafaf0f65fcef7a3521ba"}
    for name, sha in expected.items():
        path = source / name
        if digest(path) != sha:
            raise ValueError(f"Source archive hash mismatch: {name}")
        parts = []
        whole = hashlib.sha256()
        with path.open("rb") as stream:
            index = 1
            while True:
                chunk = stream.read(512 * 1024 * 1024)
                if not chunk:
                    break
                part = output / (name if path.stat().st_size <= 512 * 1024 * 1024 else f"{name}.part{index:03d}")
                if part.exists():
                    raise ValueError(f"Refusing to overwrite {part}")
                part.write_bytes(chunk)
                whole.update(chunk)
                parts.append({"name": part.name, "bytes": len(chunk), "sha256": digest(part)})
                print(f"Packaged {part.name}", flush=True)
                index += 1
        if whole.hexdigest() != sha:
            raise ValueError("Source changed while packaging")
        artifacts.append({"archive": name, "sha256": sha, "bytes": path.stat().st_size, "parts": parts})

    masks = Path("data/Trans10K_v2_mirror_3445e93")
    policy = json.loads(Path('manifests/member1_final_20261029/policy.json').read_text(encoding='utf-8'))
    mask_hashes = {}
    for key in ('v2_train_val_inventory', 'v2_test_inventory'):
        inventory = policy['inputs'][key]
        if digest(inventory) != policy['input_sha256'][inventory]:
            raise ValueError('Frozen V2 inventory changed')
        for row in json.loads(Path(inventory).read_text(encoding='utf-8'))['samples']:
            mask_hashes[masks.as_posix() + '/' + row['v2_mask']] = row['v2_mask_sha256']
    path = output / "Trans10K_v2_category_masks_3445e93.zip"
    if path.exists():
        raise ValueError("V2 package already exists")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        files = sorted(masks.rglob("*.png"))
        if len(files) != 10428:
            raise ValueError("Expected all 10428 V2 category masks")
        for item in files:
            if digest(item) != mask_hashes.get(item.as_posix()):
                raise ValueError(f'V2 source hash mismatch: {item}')
            archive.write(item, item.as_posix())
    artifacts.append({"archive": path.name, "sha256": digest(path), "bytes": path.stat().st_size,
        "parts": [{"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}]})
    manifest = {"schema_version": 1, "description": "Course reproduction assets; original authors retain rights",
        "v1_source": "https://github.com/xieenze/Segment_Transparent_Objects",
        "v2_source": "https://github.com/xieenze/Trans2Seg",
        "v2_mirror_revision": "3445e93f1e507ea6c2dd36f06badfd5e818b699f",
        "v2_author_archive_equivalence": "unverified", "artifacts": artifacts}
    (output / "dataset_assets.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(artifacts)} archives", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    package(args.source, args.output)
