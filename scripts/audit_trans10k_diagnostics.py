"""Collect all data anomalies without modifying data or issuing training manifests."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.data import decode_mask
from transparent_seg.prepare import sha256_file, write_json

EXPECTED = {"train": 5000, "validation": 1000, "test": 4428}


def inspect(task):
    root, split, group, image = task
    mask = image.parent.parent / "masks" / f"{image.stem}_mask.png"
    row = {"id": image.relative_to(root).as_posix(), "split": split,
           "difficulty": group or "train", "image": image.relative_to(root).as_posix(),
           "mask": mask.relative_to(root).as_posix(), "issues": []}
    try:
        row["image_sha256"] = sha256_file(image)
        with Image.open(image) as im:
            row["image_size"] = list(im.size)
            row["image_exif_orientation"] = im.getexif().get(274)
            rgb = np.asarray(im.convert("RGB"))
            row["pixel_sha256"] = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
        del rgb
    except Exception as error:
        row["issues"].append({"type": "unreadable_image", "detail": str(error)})
    if not mask.is_file():
        row["issues"].append({"type": "missing_mask"})
        return row
    try:
        row["mask_sha256"] = sha256_file(mask)
        with Image.open(mask) as im:
            row["mask_size"] = list(im.size)
            row["mask_mode"] = im.mode
            row["mask_exif_orientation"] = im.getexif().get(274)
            codes, counts = np.unique(np.asarray(im.convert("RGB"), np.int16).sum(2), return_counts=True)
            row["mask_channel_sum_histogram"] = {str(int(k)): int(v) for k, v in zip(codes, counts)}
        if "image_size" in row and row["image_size"] != row["mask_size"]:
            row["issues"].append({"type": "shape_mismatch", "image_size": row["image_size"],
                                  "mask_size": row["mask_size"]})
        target, valid = decode_mask(mask)
        row["foreground_pixels"] = int(target.sum())
        row["valid_pixels"] = int(valid.sum())
    except ValueError as error:
        row["issues"].append({"type": "mask_decode_error", "detail": str(error)})
    except Exception as error:
        row["issues"].append({"type": "unreadable_mask", "detail": str(error)})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/Trans10K")
    parser.add_argument("--output", default="runs/trans10k_data_audit_20261022/diagnostics")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Use 1-8 workers")
    root, output = Path(args.root).resolve(), Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new diagnostics directory")
    output.mkdir(parents=True)
    tasks, unpaired, split_counts = [], [], {}
    for split, expected in EXPECTED.items():
        groups = [""] if split == "train" else ["easy", "hard"]
        group_counts = {}
        for group in groups:
            base = root / split / group
            images = sorted(p for p in (base / "images").glob("*")
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            tasks.extend((root, split, group, p) for p in images)
            expected_masks = {f"{p.stem}_mask.png" for p in images}
            unpaired.extend(p.relative_to(root).as_posix() for p in (base / "masks").glob("*.png")
                            if p.name not in expected_masks)
            group_counts[group or "train"] = len(images)
        split_counts[split] = {"images": sum(group_counts.values()), "expected": expected,
                              "count_matches": sum(group_counts.values()) == expected,
                              "difficulty": group_counts}
    started, rows = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(inspect, tasks):
            rows.append(row)
            if len(rows) % 500 == 0:
                errors = sum(bool(x["issues"]) for x in rows)
                print(f"Scanned {len(rows)}/{len(tasks)} images; {errors} image/mask pairs with issues; "
                      f"elapsed {time.monotonic()-started:.1f}s", flush=True)
    hashes, issues, histograms = defaultdict(list), [], {s: Counter() for s in EXPECTED}
    for row in rows:
        if "pixel_sha256" in row:
            hashes[row["pixel_sha256"]].append(row["id"])
        for issue in row["issues"]:
            issues.append({"image": row["id"], "mask": row["mask"], **issue})
        histograms[row["split"]].update(row.get("mask_channel_sum_histogram", {}))
    duplicate_groups = [{"pixel_sha256": key, "ids": ids} for key, ids in hashes.items() if len(ids) > 1]
    cross = [g for g in duplicate_groups if len({i.split('/')[0] for i in g['ids']}) > 1]
    within = [g for g in duplicate_groups if len({i.split('/')[0] for i in g['ids']}) == 1]
    issue_counts = dict(Counter(x["type"] for x in issues))
    count_ok = all(v["count_matches"] for v in split_counts.values())
    passed = not issues and not unpaired and not cross and count_ok
    summary = {"recorded_at": datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        "audit_type": "diagnostic_collection_not_training_manifest",
        "status": "passed_diagnostics" if passed else "blocked",
        "training_manifests_issued": False, "root": str(root), "splits": split_counts,
        "images_scanned": len(rows), "pairs_with_issues": sum(bool(r["issues"]) for r in rows),
        "issue_counts": issue_counts, "issues": issues, "unpaired_masks": unpaired,
        "cross_split_exact_duplicate_groups": cross, "within_split_exact_duplicate_groups": within,
        "mask_channel_sum_histograms": {k: dict(v) for k, v in histograms.items()},
        "empty_foreground_samples": [r["id"] for r in rows if r.get("foreground_pixels") == 0],
        "all_foreground_samples": [r["id"] for r in rows if r.get("foreground_pixels", -1) == r.get("valid_pixels", -2)],
        "near_duplicate_audit": "not performed; sampled visual candidates recorded separately",
        "data_changes": [], "seconds": time.monotonic()-started}
    write_json(output / "audit.json", summary)
    write_json(output / "diagnostic_inventory.json", {"schema": "trans10k_diagnostic_inventory_v1",
        "not_for_training": True, "root": str(root), "samples": rows})
    print(json.dumps({k: summary[k] for k in ("status", "images_scanned", "pairs_with_issues", "issue_counts", "seconds")}), flush=True)
    print(f"Cross-split exact duplicate groups: {len(cross)}; within-split: {len(within)}", flush=True)


if __name__ == "__main__":
    main()
