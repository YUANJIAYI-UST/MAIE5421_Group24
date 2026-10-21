"""Original-resolution evaluation and paired fixed photometric stress tests."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import PERTURBATIONS, decode_mask, image_tensor, read_manifest
from .metrics import SegmentationMetrics
from .model import TransparentSegformer
from .prepare import sha256_file, write_json


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TransparentSegformer(boundary=checkpoint["config"]["boundary"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict(model, image, size, device, condition=None):
    tensor = image_tensor(image, size, condition)[None].to(device)
    logits = model(tensor)["logits"]
    # Interpolate LOGITS before sigmoid/threshold, never resize binary predictions.
    logits = F.interpolate(logits.float(), size=(image.height, image.width),
                           mode="bilinear", align_corners=False)
    return logits.sigmoid()[0, 0].cpu().numpy() >= 0.5


def save_panel(image, target, prediction, path):
    base = np.asarray(image.convert("RGB"))
    overlay = base.copy().astype(np.float32)
    overlay[prediction] = 0.5 * overlay[prediction] + 0.5 * np.array([0, 220, 160])
    gt = np.repeat((target.astype(np.uint8) * 255)[..., None], 3, axis=2)
    pred = np.repeat((prediction.astype(np.uint8) * 255)[..., None], 3, axis=2)
    panel = Image.fromarray(np.concatenate([base, gt, pred, overlay.astype(np.uint8)], axis=1))
    panel.thumbnail((1600, 600))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def evaluate(model, manifest_path, size, device, root=None, condition=None, output=None, visuals=0):
    meta, base = read_manifest(manifest_path, root)
    model.eval()
    meters = {"all": SegmentationMetrics(), "easy": SegmentationMetrics(), "hard": SegmentationMetrics()}
    per_image = []
    for i, sample in enumerate(meta["samples"]):
        with Image.open(base / sample["image"]) as im:
            image = im.convert("RGB")
        y, valid = decode_mask(base / sample["mask"], meta["encoding"], meta.get("void_values", []))
        p = predict(model, image, size, device, condition)
        meters["all"].update(p, y, valid)
        if sample.get("difficulty") in ("easy", "hard"):
            meters[sample["difficulty"]].update(p, y, valid)
        single = SegmentationMetrics()
        single.update(p, y, valid)
        per_image.append({"id": sample["id"], **single.compute()})
        if output and i < visuals:
            save_panel(image, y, p, Path(output) / "visuals" / f"{i:04d}.png")
    result = {"split": meta["split"], "dataset": meta["dataset"],
              "synthetic": meta.get("synthetic", False),
              "complete_official_split": meta.get("complete_official_split", False),
              "manifest_sha256": sha256_file(manifest_path),
              "condition": list(condition) if condition else "clean", "threshold": 0.5,
              "resolution": "original", "scores": {k: v.compute() for k, v in meters.items()}}
    if output:
        write_json(Path(output) / "metrics.json", result)
        with (Path(output) / "per_image.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
            writer.writeheader()
            writer.writerows(per_image)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--visuals", type=int, default=12)
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new evaluation directory to preserve previous evidence")
    size = checkpoint["config"]["size"]
    clean = evaluate(model, args.manifest, size, device, args.root,
                     output=output / "clean", visuals=args.visuals)
    write_json(output / "provenance.json", {"checkpoint_sha256": sha256_file(args.checkpoint),
               "training_config": checkpoint["config"], "device": device,
               "epoch": checkpoint["epoch"], "selection": "validation foreground IoU"})
    print(clean["scores"])
    if args.robustness:
        results = []
        for condition in PERTURBATIONS:
            folder = output / f"{condition[0]}_{condition[1]:g}"
            results.append(evaluate(model, args.manifest, size, device, args.root,
                                    condition, output=folder))
        summary = {}
        for subset in ("all", "easy", "hard"):
            summary[subset] = {}
            for metric in ("foreground_iou", "region_f1", "boundary_iou"):
                values = [r["scores"][subset][metric] for r in results]
                mean = float(np.mean(values)) if all(v is not None for v in values) else None
                original = clean["scores"][subset][metric]
                summary[subset][metric] = {"clean": original, "perturbed_mean": mean,
                    "drop": original - mean if original is not None and mean is not None else None}
        write_json(output / "robustness.json", {"conditions": results, "summary": summary,
                                                "units": "fraction; multiply drop by 100 for percentage points"})


if __name__ == "__main__":
    main()
