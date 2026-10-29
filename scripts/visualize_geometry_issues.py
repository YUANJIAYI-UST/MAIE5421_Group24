"""Render rotation hypotheses for geometry issues without changing dataset files."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_trans10k_audit import decode_mask, font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/Trans10K")
    parser.add_argument("--audit", required=True, help="JSON with an issues array")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.root).resolve(), Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new diagnostic visualization directory")
    output.mkdir(parents=True)
    panels, records = [], []
    for issue in json.loads(Path(args.audit).read_text())["issues"]:
        if "image_size" not in issue or "mask_size" not in issue:
            continue
        paths = [root / issue[k] for k in ("image", "mask")]
        if not all(p.resolve().is_relative_to(root) for p in paths):
            raise ValueError("Image or mask path escapes dataset root")
        with Image.open(paths[0]) as im:
            image = im.convert("RGB")
        y, _ = decode_mask(paths[1])
        mask = Image.fromarray(y.astype(np.uint8) * 255)
        canvas = Image.new("RGB", (1648, 440), "#f1f5f9")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f"{issue['image']} | image {image.size}, mask {mask.size} | candidates only, NOT APPLIED",
                  fill="#0f172a", font=font(18))
        tiles = [("Original image", image), ("Original mask", mask.convert("RGB"))]
        image = image.copy()
        image.thumbnail((400, 350), Image.Resampling.LANCZOS)
        for name, op in (("Mask rotated 90 CW", Image.Transpose.ROTATE_270),
                         ("Mask rotated 90 CCW", Image.Transpose.ROTATE_90)):
            rotated = mask.transpose(op)
            if rotated.size != tuple(issue["image_size"]):
                continue
            foreground = np.asarray(rotated.resize(image.size, Image.Resampling.NEAREST)) > 0
            rgb = np.asarray(image).astype(np.float32).copy()
            rgb[foreground] = 0.55 * rgb[foreground] + 0.45 * np.array([0, 220, 160])
            tiles.append((name, Image.fromarray(rgb.astype(np.uint8))))
        for index, (name, tile) in enumerate(tiles):
            x = 12 + index * 408
            draw.text((x, 45), name, fill="#334155", font=font(18))
            tile = tile.copy()
            tile.thumbnail((400, 350), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x + (400 - tile.width) // 2, 78 + (350 - tile.height) // 2))
        name = issue["image"].replace("/", "_") + ".jpg"
        canvas.save(output / name, quality=95)
        panels.append(canvas)
        records.append({**issue, "comparison": name, "decision": "pending visual review", "applied": False})
    for start in range(0, len(panels), 4):
        subset = panels[start:start + 4]
        sheet = Image.new("RGB", (1648, 440 * len(subset)), "white")
        for index, panel in enumerate(subset):
            sheet.paste(panel, (0, 440 * index))
        sheet.save(output / f"geometry_contact_{start // 4 + 1:02d}.jpg", quality=95)
    (output / "review_decisions.json").write_text(json.dumps({"data_modified": False,
        "purpose": "Visual hypotheses only; no training labels were changed", "cases": records}, indent=2) + "\n")
    print(f"Created {len(records)} geometry comparison panels; dataset files untouched.")


if __name__ == "__main__":
    main()
