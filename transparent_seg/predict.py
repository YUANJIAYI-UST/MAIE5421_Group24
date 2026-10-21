"""RGB image to original-size binary PNG and overlay (no auxiliary inference)."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .evaluate import load_checkpoint, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as im:
        image = im.convert("RGB")
    p = predict(model, image, checkpoint["config"]["size"], device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(p.astype(np.uint8) * 255).save(output / "mask.png")
    overlay = np.asarray(image).copy().astype(np.float32)
    overlay[p] = 0.5 * overlay[p] + 0.5 * np.array([0, 220, 160])
    Image.fromarray(overlay.astype(np.uint8)).save(output / "overlay.png")
    print(f"Saved {image.width}x{image.height} mask and overlay to {output}")


if __name__ == "__main__":
    main()
