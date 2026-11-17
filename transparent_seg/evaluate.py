"""Original-resolution evaluation and paired fixed photometric stress tests."""
import argparse
import csv
import subprocess
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

def predict_in_v2_frame(logits, v2_shape, v2_to_v1_orientation):
    """Map one image's foreground logits to the native V2 mask grid."""
    if logits.ndim != 4 or logits.shape[:2] != (1, 1):
        raise ValueError("Expected foreground logits with shape [1, 1, H, W]")

    # torch.rot90: k=1 is counterclockwise, k=-1 is clockwise.
    inverse_turns = {"none": 0, "rotate_90_cw": 1, "rotate_90_ccw": -1}
    if v2_to_v1_orientation not in inverse_turns:
        raise ValueError(f"Unknown orientation: {v2_to_v1_orientation}")

    aligned = torch.rot90(logits.float(), inverse_turns[v2_to_v1_orientation], (-2, -1))
    aligned = F.interpolate(aligned, size=v2_shape, mode="bilinear", align_corners=False)
    return (aligned[0, 0].sigmoid() >= 0.5).detach().cpu().numpy()


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


def evaluate_selection(model, dataset, device, progress_every=0):
    """Original-size region counts; boundary metrics deferred to final evaluation."""
    model.eval()
    meters = {k: SegmentationMetrics() for k in ("all", "easy", "hard")}
    for i in range(len(dataset)):
        if progress_every and i % progress_every == 0:
            print(f"Validation selection: {i}/{len(dataset)}", flush=True)
        sample = dataset[i]
        y, v = sample["target"][0].to(device).bool(), sample["valid"][0].to(device)
        logits = model(sample["image"][None].to(device))["logits"]
        logits = F.interpolate(logits.float(), size=y.shape, mode="bilinear", align_corners=False)
        p = logits.sigmoid()[0, 0] >= 0.5
        counts = [int(x.item()) for x in ((p & y & v).sum(), (p & ~y & v).sum(),
                                          (~p & y & v).sum(), (~p & ~y & v).sum())]
        groups = ["all"] + ([sample["difficulty"]] if sample["difficulty"] in ("easy", "hard") else [])
        for group in groups:
            meter = meters[group]
            meter.images += int(sum(counts) > 0)
            meter.skipped += int(sum(counts) == 0)
            for key, value in zip(("tp", "fp", "fn", "tn"), counts):
                setattr(meter, key, getattr(meter, key) + value)
    return {"scores": {k: v.compute() for k, v in meters.items()},
            "resolution": "original", "boundary_metrics": "deferred_to_selected_checkpoint"}


@torch.inference_mode()
def evaluate(model, manifest_path, size, device, root=None, condition=None, output=None, visuals=0, progress_every=0):
    meta, base = read_manifest(manifest_path, root)
    model.eval()
    meters = {"all": SegmentationMetrics(), "easy": SegmentationMetrics(), "hard": SegmentationMetrics()}
    per_image = []
    for i, sample in enumerate(meta["samples"]):
        if progress_every and i % progress_every == 0:
            print(f"Evaluation {meta['split']}: {i}/{len(meta['samples'])}", flush=True)
        with Image.open(base / sample["image"]) as im:
            image = im.convert("RGB")
        y, valid = decode_mask(base / sample["mask"], meta["encoding"], meta.get("void_values", []))
        p = predict(model, image, size, device, condition)
        single = SegmentationMetrics()
        single.update(p, y, valid)
        # Reuse the image's sufficient statistics. In particular, original-size
        # boundary morphology must not run again for all/easy/hard aggregation.
        groups = ["all"]
        if sample.get("difficulty") in ("easy", "hard"):
            groups.append(sample["difficulty"])
        for group in groups:
            meter = meters[group]
            for key in ("tp", "fp", "fn", "tn", "images", "skipped"):
                setattr(meter, key, getattr(meter, key) + getattr(single, key))
            meter.boundaries.extend(single.boundaries)
        if output:
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
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(checkpoint["config"].get("deterministic", False))
    try:
        evaluation_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        evaluation_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        evaluation_commit, evaluation_dirty = None, None
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new evaluation directory to preserve previous evidence")
    size = checkpoint["config"]["size"]
    clean = evaluate(model, args.manifest, size, device, args.root,
                     output=output / "clean", visuals=args.visuals, progress_every=args.progress_every)
    write_json(output / "provenance.json", {"checkpoint_sha256": sha256_file(args.checkpoint),
               "training_config": checkpoint["config"], "device": device,
               "epoch": checkpoint["epoch"], "selection": "validation foreground IoU",
               "evaluation_git_commit": evaluation_commit, "evaluation_git_dirty": evaluation_dirty})
    print(clean["scores"])
    if args.robustness:
        results = []
        for condition in PERTURBATIONS:
            folder = output / f"{condition[0]}_{condition[1]:g}"
            results.append(evaluate(model, args.manifest, size, device, args.root,
                                    condition, output=folder, progress_every=args.progress_every))
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
