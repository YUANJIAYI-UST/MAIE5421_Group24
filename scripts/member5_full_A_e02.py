"""Evaluate full-category A on the frozen D03 test groups, resuming by image."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.data import decode_mask, image_tensor, read_manifest
from transparent_seg.data_v2 import CLASS_NAMES, read_category_mask
from transparent_seg.evaluate import load_checkpoint, predict_in_v2_frame
from transparent_seg.metrics import CategoryRecall
from transparent_seg.prepare import sha256_file


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    folder = Path("../Member1_Final_D01_D02_D03_20261029/manifests/member1_final_20261029/d03")
    checkpoint = Path("runs/member5_e02/A_best.pt")
    digest = sha256_file(checkpoint)
    if digest != "3a785feb578ecbd2199684c53130737c872cd6c2f5803f221297f1e9dac0b25a":
        raise ValueError("Unexpected A checkpoint")
    manifest, root = read_manifest(folder / "test.json")
    rows = manifest["samples"]
    ids = {r["id"] for r in rows}
    subsets = {}
    for group in ("seen", "heldout"):
        members, _ = read_manifest(folder / f"test_{group}.json")
        subsets[group] = {r["id"] for r in members["samples"]}
    if subsets["seen"] & subsets["heldout"] or subsets["seen"] | subsets["heldout"] != ids:
        raise ValueError("Invalid test partition")
    paths = [folder / name for name in ("test.json", "test_seen.json", "test_heldout.json")]
    paths += [Path(__file__), Path("transparent_seg/evaluate.py"), Path("transparent_seg/metrics.py")]
    run = checkpoint.parent / "full_A"
    run.mkdir(exist_ok=True)
    receipt = run / "inputs.json"
    inputs = {"checkpoint_sha256": digest, "torch": str(torch.__version__),
              "file_hashes": {p.name: sha256_file(p) for p in paths}}
    if receipt.exists() and json.loads(receipt.read_text()) != inputs:
        raise ValueError("Resume inputs differ from recorded run")
    receipt.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    model, ck = load_checkpoint(checkpoint, "cpu")
    stats = {g: np.zeros((2, len(CLASS_NAMES)), dtype=np.int64) for g in ("all", *subsets)}
    background = {g: np.zeros(2, dtype=np.int64) for g in stats}
    for index, row in enumerate(rows, 1):
        cache = run / f"{index:05}.json"
        if cache.exists():
            record = json.loads(cache.read_text())
        else:
            labels = read_category_mask(root / row["v2_category_mask"], "author_rgb_palette")
            with Image.open(root / row["image"]) as image:
                x = image_tensor(image.convert("RGB"), ck["config"]["size"])[None]
            logits = model(x)["logits"]
            meter = CategoryRecall()
            meter.update(predict_in_v2_frame(logits, labels.shape, row["v2_to_v1_orientation"]), labels)
            target, valid = decode_mask(root / row["mask"], manifest["encoding"], manifest["void_values"])
            bg = valid & (target == 0)
            pred = predict_in_v2_frame(logits, bg.shape, "none")
            record = dict(id=row["id"], hits=meter.hits.tolist(), total=meter.total.tolist(),
                          fp=int(np.count_nonzero(pred & bg)), bg=int(np.count_nonzero(bg)))
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps(record), encoding="utf-8")
            tmp.replace(cache)
        if record["id"] != row["id"] or record["total"] != row["v2_category_pixel_counts"]:
            raise ValueError(f"Manifest mismatch at {row['id']}")
        group = "heldout" if row["id"] in subsets["heldout"] else "seen"
        for key in ("all", group):
            stats[key] += [record["hits"], record["total"]]
            background[key] += [record["fp"], record["bg"]]
        if index == 1 or index % 50 == 0:
            print(f"{index}/{len(rows)}", flush=True)
    output = Path("reports/member5_e02_full_A.csv")
    tmp = output.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subset", "images", "metric", "class", "numerator", "denominator", "value"])
        for key in stats:
            count = len(rows) if key == "all" else len(subsets[key])
            fp, bg = background[key]
            writer.writerow([key, count, "background_fpr", "V1 background", fp, bg, fp/bg if bg else None])
            for c in range(1, len(CLASS_NAMES)):
                hits, total = stats[key][:, c]
                writer.writerow([key, count, "category_recall", CLASS_NAMES[c], hits, total, hits/total if total else None])
    tmp.replace(output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
