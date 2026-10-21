"""Synthetic shapes for plumbing tests ONLY. Not transparent-object evidence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw
from transparent_seg.prepare import audit_trans10k


def main():
    root = Path("data/smoke")
    if root.exists():
        raise SystemExit("data/smoke already exists; use the existing fixtures or choose a fresh checkout")
    rng = np.random.default_rng(20261021)
    for split, count in [("train", 12), ("validation", 4), ("test", 4)]:
        for i in range(count):
            group = "" if split == "train" else ("easy" if i % 2 == 0 else "hard")
            folder = root / split / group
            (folder / "images").mkdir(parents=True, exist_ok=True)
            (folder / "masks").mkdir(parents=True, exist_ok=True)
            w, h = 112 + 8 * (i % 3), 80 + 8 * (i % 2)
            mask = Image.new("RGB", (w, h))
            draw = ImageDraw.Draw(mask)
            draw.ellipse((int(rng.integers(5, 20)), 10, w - 15, h - 8), fill=(255, 0, 0))
            draw.rectangle((w - 30, 4, w - 8, 24), fill=(255, 255, 255))
            foreground = np.asarray(mask).sum(2) > 0
            rgb = rng.uniform(0.05, 0.3, size=(h, w, 3))
            rgb[foreground] += [0.4, 0.5, 0.3]
            Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(folder / "images" / f"{i:03d}.jpg")
            mask.save(folder / "masks" / f"{i:03d}_mask.png")
    print(audit_trans10k(root, "manifests/smoke", allow_partial=True, synthetic=True))
    print("SYNTHETIC FIXTURE ONLY: do not report these scores as Trans10K results.")


if __name__ == "__main__":
    main()
