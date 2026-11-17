"""Check pooled category recall for the first five D03 test images."""
import argparse
from pathlib import Path
import sys

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.data import image_tensor, read_manifest
from transparent_seg.data_v2 import read_category_mask
from transparent_seg.evaluate import load_checkpoint, predict_in_v2_frame
from transparent_seg.metrics import CategoryRecall


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    manifest, root = read_manifest(args.manifest)
    rows = manifest["samples"][:5]
    if manifest["split"] != "test" or len(rows) != 5:
        raise ValueError("Expected a test manifest with at least five images")

    model, checkpoint = load_checkpoint(args.checkpoint, "cpu")
    metric = CategoryRecall()
    for row in rows:
        labels = read_category_mask(root / row["v2_category_mask"], "author_rgb_palette")
        with Image.open(root / row["image"]) as image:
            x = image_tensor(image.convert("RGB"), checkpoint["config"]["size"])[None]
        logits = model(x)["logits"]
        prediction = predict_in_v2_frame(logits, labels.shape, row["v2_to_v1_orientation"])
        metric.update(prediction=prediction, category_ids=labels)
        print("Evaluated:", row["id"], flush=True)

    for class_id, counts in metric.compute().items():
        expected = sum(row["v2_category_pixel_counts"][class_id] for row in rows)
        if counts["total"] != expected:
            raise ValueError(f"Pixel count mismatch for class {class_id}")
        if expected:
            print(f"{counts['class_name']}: {counts['hits']}/{expected} = {counts['recall']:.2%}")
    print("All foreground pixel counts match the manifest.")


if __name__ == "__main__":
    main()
