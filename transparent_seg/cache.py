"""Lossless cache of PIL preprocessing, bound to source byte hashes."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from importlib.metadata import version

import numpy as np
from PIL import Image

from .data import decode_mask, read_manifest
from .prepare import sha256_file, write_json


def build_cache(manifest, output, size=512, training=False, root=None, workers=4):
    meta, base = read_manifest(manifest, root)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new cache directory")
    output.mkdir(parents=True, exist_ok=True)

    def build(pair):
        index, sample = pair
        for key in ("image", "mask"):
            expected = sample.get(key + "_sha256")
            if not expected or sha256_file(base / sample[key]) != expected:
                raise ValueError(f"Source hash mismatch: {sample['id']} {key}")
        with Image.open(base / sample["image"]) as im:
            image = im.convert("RGB")
        target, valid = decode_mask(base / sample["mask"], meta["encoding"], meta.get("void_values", []))
        if image.size != (target.shape[1], target.shape[0]):
            raise ValueError(f"Image/mask geometry differs: {sample['id']}")
        rgb = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))
        if training:
            target = np.asarray(Image.fromarray(target).resize((size, size), Image.Resampling.NEAREST))
            valid = np.asarray(Image.fromarray(valid.astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST))
        path = output / f"{index:06d}.npz"
        np.savez(path, rgb=rgb, shape=np.array(target.shape),
                 target=np.packbits(target.astype(bool)), valid=np.packbits(valid.astype(bool)))
        return {"id": sample["id"], "file": path.name, "sha256": sha256_file(path)}

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(build, enumerate(meta["samples"])):
            records.append(record)
            if len(records) % 100 == 0:
                print(f"Cache {meta['split']}: {len(records)}/{len(meta['samples'])}", flush=True)
    write_json(output / "index.json", {"schema": 1, "preprocessing": "PIL-bilinear-rgb-nearest-label-v1",
        "manifest_sha256": sha256_file(manifest), "size": size, "training": training,
        "pillow_version": version("Pillow"), "numpy_version": version("numpy"), "samples": records})


class PreprocessingCache:
    def __init__(self, directory, manifest, size, training, samples):
        self.directory = Path(directory)
        index = json.loads((self.directory / "index.json").read_text(encoding="utf-8"))
        if (index["schema"] != 1 or index["preprocessing"] != "PIL-bilinear-rgb-nearest-label-v1"
                or index["manifest_sha256"] != sha256_file(manifest)
                or index["size"] != size or index["training"] != training
                or index["pillow_version"] != version("Pillow") or index["numpy_version"] != version("numpy")
                or [x["id"] for x in index["samples"]] != [x["id"] for x in samples]):
            raise ValueError("Cache does not match manifest/preprocessing")
        self.paths = []
        for record in index["samples"]:
            path = (self.directory / record["file"]).resolve()
            if not path.is_relative_to(self.directory.resolve()) or sha256_file(path) != record["sha256"]:
                raise ValueError("Cache file hash mismatch")
            self.paths.append(path)

    def read(self, index):
        with np.load(self.paths[index], allow_pickle=False) as item:
            shape = tuple(item["shape"].tolist())
            count = int(np.prod(shape))
            target = np.unpackbits(item["target"], count=count).reshape(shape).astype(np.float32)
            valid = np.unpackbits(item["valid"], count=count).reshape(shape).astype(bool)
            return Image.fromarray(item["rgb"]), target, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build_cache(args.manifest, args.output, args.size, args.training, workers=args.workers)


if __name__ == "__main__":
    main()
