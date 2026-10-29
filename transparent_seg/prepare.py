"""Audit official Trans10K layout; freeze file lists without random re-splitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import decode_mask


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def audit_trans10k(root, output, allow_partial=False, synthetic=False):
    root, output = Path(root).resolve(), Path(output)
    manifests, seen, overlaps = {}, {}, []
    for split, expected in [("train", 5000), ("validation", 1000), ("test", 4428)]:
        groups = [""] if split == "train" else ["easy", "hard"]
        rows, encodings = [], Counter()
        for group in groups:
            folder = root / split / group
            images = sorted(p for p in (folder / "images").glob("*")
                            if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            paired_masks = set()
            for image in images:
                mask = folder / "masks" / f"{image.stem}_mask.png"
                if not mask.is_file():
                    raise ValueError(f"Missing label: {mask}")
                paired_masks.add(mask.resolve())
                y, v = decode_mask(mask)
                with Image.open(image) as im:
                    rgb = np.asarray(im.convert("RGB"))
                    if rgb.shape[:2] != y.shape:
                        raise ValueError(f"Shape mismatch: {image}")
                    pixel_hash = hashlib.sha256(str(rgb.shape).encode() + rgb.tobytes()).hexdigest()
                with Image.open(mask) as im:
                    codes, counts = np.unique(np.asarray(im.convert("RGB"), np.int16).sum(2), return_counts=True)
                    encodings.update({str(int(k)): int(n) for k, n in zip(codes, counts)})
                if pixel_hash in seen and seen[pixel_hash][0] != split:
                    overlaps.append([seen[pixel_hash], [split, image.relative_to(root).as_posix()]])
                seen[pixel_hash] = (split, image.relative_to(root).as_posix())
                rows.append({"id": image.relative_to(root).as_posix(),
                             "image": image.relative_to(root).as_posix(),
                             "mask": mask.relative_to(root).as_posix(),
                             "difficulty": group or "train", "height": y.shape[0], "width": y.shape[1],
                             "image_sha256": sha256_file(image), "mask_sha256": sha256_file(mask),
                             "pixel_sha256": pixel_hash, "foreground_pixels": int(y.sum()),
                             "valid_pixels": int(v.sum())})
            extras = set(p.resolve() for p in (folder / "masks").glob("*.png")) - paired_masks
            if extras:
                raise ValueError(f"Unpaired masks: {sorted(map(str, extras))[:3]}")
        if not rows or (not allow_partial and len(rows) != expected):
            raise ValueError(f"{split}: found {len(rows)}, expected {expected}. Check extraction/layout.")
        manifests[split] = {"schema_version": 1, "dataset": "synthetic_smoke" if synthetic else "trans10k_v1",
                            "root": str(root), "split": split, "encoding": "trans10k_v1_rgb",
                            "void_values": [], "official_count": expected,
                            "complete_official_split": len(rows) == expected and not synthetic,
                            "synthetic": synthetic, "mask_channel_sum_histogram": dict(encodings),
                            "samples": rows}
    if overlaps:
        write_json(output / "overlap_errors.json", overlaps)
        raise ValueError("Cross-split pixel-identical images found; see overlap_errors.json. Audit before training.")
    for split, manifest in manifests.items():
        write_json(output / f"{split}.json", manifest)
    summary = {split: {"images": len(m["samples"]), "complete": m["complete_official_split"],
                       "difficulty": dict(Counter(x["difficulty"] for x in m["samples"]))}
               for split, m in manifests.items()}
    write_json(output / "audit.json", {"splits": summary, "cross_split_exact_overlap": [],
                                       "near_duplicate_audit": "pending manual/perceptual review"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", default="manifests/trans10k")
    parser.add_argument("--allow-partial", action="store_true", help="Pilot only; records incomplete split")
    parser.add_argument("--review-policy", help="Explicit local derivative policy; source data stays immutable")
    parser.add_argument("--source-inventory", help="Pinned full diagnostic inventory for a derivative")
    parser.add_argument("--source-audit", help="Pinned original diagnostic audit for a derivative")
    args = parser.parse_args()
    if args.review_policy:
        if args.allow_partial or not args.source_inventory or not args.source_audit:
            parser.error("A review policy needs --source-inventory and --source-audit; do not use --allow-partial")
        from .prepare_reviewed import prepare_reviewed
        prepare_reviewed(args.root, args.output, args.review_policy, args.source_inventory, args.source_audit)
        return
    if args.source_inventory or args.source_audit:
        parser.error("Source evidence arguments require --review-policy")
    print(json.dumps(audit_trans10k(args.root, args.output, args.allow_partial), indent=2))


if __name__ == "__main__":
    main()
