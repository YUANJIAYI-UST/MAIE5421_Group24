"""Create reproducible train/validation GT overlays for a data-quality review."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.data import decode_mask


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def sample_images(root, split, count, seed):
    groups = [""] if split == "train" else ["easy", "hard"]
    rng = random.Random(seed)
    selected = []
    for index, group in enumerate(groups):
        paths = sorted(p for p in (root / split / group / "images").iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        amount = count // len(groups) + (index < count % len(groups))
        if len(paths) < amount:
            raise ValueError(f"Not enough samples in {split}/{group}")
        selected.extend(sorted(rng.sample(paths, amount)))
    return selected


def make_panel(image, target, overlay, caption):
    panel = Image.new("RGB", (1232, 366), "#f1f5f9")
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), caption, fill="#0f172a", font=font(18))
    for index, (label, tile) in enumerate((("Original RGB", image),
            ("Binary ground truth", target), ("GT overlay (green)", overlay))):
        x = 12 + index * 408
        draw.text((x, 35), label, fill="#334155", font=font(16))
        tile = tile.copy()
        tile.thumbnail((400, 300), Image.Resampling.LANCZOS)
        panel.paste(tile, (x + (400 - tile.width) // 2, 60 + (300 - tile.height) // 2))
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/Trans10K")
    parser.add_argument("--output", default="runs/trans10k_data_audit_20261022")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--count", type=int, default=20, help="Samples per split")
    parser.add_argument("--splits", choices=("train", "validation"), nargs="+",
                        default=["train", "validation"])
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be positive")
    root, output = Path(args.root).resolve(), Path(args.output)
    if output.exists():
        raise ValueError("Use a new output directory to preserve review evidence")
    output.mkdir(parents=True)
    records = []
    for split in args.splits:
        panels = []
        for index, path in enumerate(sample_images(root, split, args.count, args.seed)):
            mask = path.parent.parent / "masks" / f"{path.stem}_mask.png"
            y, valid = decode_mask(mask)
            with Image.open(path) as im:
                image = im.convert("RGB")
            assert image.size == (y.shape[1], y.shape[0]) and valid.all()
            foreground = y.astype(bool)
            rgb = np.asarray(image)
            tinted = rgb.astype(np.float32).copy()
            tinted[foreground] = 0.55 * tinted[foreground] + 0.45 * np.array([0, 220, 160])
            boundary = foreground & ~binary_erosion(foreground)
            tinted[boundary] = [255, 220, 0]
            overlay = Image.fromarray(tinted.astype(np.uint8))
            target = Image.fromarray(y.astype(np.uint8) * 255).convert("RGB")
            sample_id = path.relative_to(root).as_posix()
            name = f"{index + 1:02d}_{path.stem}.png"
            overlay_path = output / "overlays" / split / name
            panel_path = output / "panels" / split / name
            for p in (overlay_path, panel_path):
                p.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(overlay_path)
            caption = f"{index + 1:02d} | {sample_id} | {image.width}x{image.height} | FG {foreground.mean():.1%}"
            panel = make_panel(image, target, overlay, caption)
            panel.save(panel_path)
            panels.append(panel)
            records.append({"split": split, "id": sample_id,
                "mask": mask.relative_to(root).as_posix(), "size": list(image.size),
                "image_sha256": sha256(path), "mask_sha256": sha256(mask),
                "foreground_fraction": float(foreground.mean()),
                "overlay": overlay_path.relative_to(output).as_posix(),
                "panel": panel_path.relative_to(output).as_posix(),
                "visual_review": "pending"})
        for start in range(0, len(panels), 5):
            subset = panels[start:start + 5]
            sheet = Image.new("RGB", (1232, 366 * len(subset) + 36), "white")
            ImageDraw.Draw(sheet).text((12, 8),
                f"Trans10K-v1 {split} | seed {args.seed} | samples {start + 1}-{start + len(subset)}",
                fill="#0f172a", font=font(20))
            for row, panel in enumerate(subset):
                sheet.paste(panel, (0, 36 + row * 366))
            sheet.save(output / f"{split}_contact_{start // 5 + 1:02d}.jpg", quality=94)
    (output / "review_samples.json").write_text(json.dumps({"seed": args.seed,
        "selection": {"count_per_split": args.count, "splits": args.splits,
                      "method": "uniform without replacement; validation stratified equally by easy/hard"},
        "samples": records}, indent=2) + "\n")
    print(f"Created {len(records)} original-resolution GT overlays and matching panels in {output}")


if __name__ == "__main__":
    main()
