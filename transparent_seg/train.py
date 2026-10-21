"""Single-device training with validation-only selection and reproducible logs."""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SegmentationDataset, read_manifest
from .evaluate import evaluate
from .losses import segmentation_loss
from .model import TransparentSegformer
from .prepare import sha256_file, write_json


def load_config(path):
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    if "extends" in config:
        base = load_config(path.parent / config.pop("extends"))
        base.update(config)
        return base
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shared_parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("boundary."):
            digest.update(name.encode())
            digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def train(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new run directory; existing experiments are never overwritten")
    if config["size"] < 32 or config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Invalid training dimensions/budget")
    if config.get("purpose") != "smoke" and not config["pretrained"]:
        raise ValueError("Research runs require ImageNet initialization per proposal")
    root = config.get("root")
    training = SegmentationDataset(config["train_manifest"], config["size"], True,
                                   config["augmentation"], config["seed"], root)
    validation, _ = read_manifest(config["val_manifest"], root)
    if validation["split"] != "validation":
        raise ValueError("Model selection requires validation split, never test")
    train_hashes = {x["pixel_sha256"] for x in training.samples}
    val_hashes = {x["pixel_sha256"] for x in validation["samples"]}
    if train_hashes & val_hashes:
        raise ValueError("Train/validation image leakage detected")
    if training.meta["dataset"] != validation["dataset"]:
        raise ValueError("Training and validation datasets must match")
    if config.get("purpose") != "smoke" and (training.meta.get("synthetic") or validation.get("synthetic")):
        raise ValueError("Synthetic smoke data cannot be used for a research run")
    if config.get("purpose") == "research" and not all(
            m.get("complete_official_split") for m in (training.meta, validation)):
        raise ValueError("Main research runs require complete official train/validation splits; use pilot for partial data")
    seed_everything(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install CUDA PyTorch via scripts/setup.ps1")
    model = TransparentSegformer(config["boundary"], config["pretrained"],
                                 config.get("encoder_revision", "main"), ".cache/huggingface").to(device)
    initial_hash = shared_parameter_hash(model)
    # Optional head initialization must not shift the shared dropout/drop-path RNG.
    seed_everything(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True,
                        num_workers=config.get("workers", 0), generator=generator,
                        pin_memory=device.startswith("cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    steps_per_epoch = min(len(loader), config.get("max_steps_per_epoch") or len(loader))
    total_steps = steps_per_epoch * config["epochs"]
    warmup = max(1, int(total_steps * 0.05))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, 1 - (step - warmup) / max(1, total_steps - warmup)) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    amp = bool(config.get("amp")) and device.startswith("cuda")
    # BF16 retains the FP32 exponent range; avoids FP16 loss-scale overflow on
    # weak-boundary batches. Disable AMP for GPUs without BF16 support.
    if amp and (config.get("amp_dtype", "bfloat16") != "bfloat16" or not torch.cuda.is_bf16_supported()):
        raise ValueError("This training protocol uses BF16 AMP; set amp=false for GPUs without BF16 support")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    write_json(output / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
        "device": device, "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "git_commit": commit, "git_dirty": dirty, "initial_shared_state_sha256": initial_hash,
        "encoder_resolved_revision": model.encoder_revision,
        "train_manifest_sha256": sha256_file(config["train_manifest"]),
        "val_manifest_sha256": sha256_file(config["val_manifest"]),
        "synthetic": training.meta.get("synthetic", False),
        "full_official_train": training.meta.get("complete_official_split", False)})
    (output / "pip_freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    best = -1.0
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(config["epochs"]):
        started = time.perf_counter()
        training.epoch = epoch
        model.train()
        totals = {key: 0.0 for key in ("total", "bce", "dice", "boundary")}
        for step, batch in enumerate(loader):
            if step >= steps_per_epoch:
                break
            optimizer.zero_grad(set_to_none=True)
            x, y, valid = (batch[key].to(device) for key in ("image", "target", "valid"))
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                                dtype=torch.bfloat16, enabled=amp):
                losses = segmentation_loss(model(x, return_boundary=config["boundary"]), y, valid,
                                            config["boundary_weight"], config["boundary_radius"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Non-finite loss")
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            for key, value in losses.items():
                totals[key] += value.item()
        scores = evaluate(model, config["val_manifest"], config["size"], device, root)["scores"]
        score = scores["all"]["foreground_iou"]
        if score is None or not math.isfinite(score):
            raise ValueError("Validation contains no valid pixels")
        selected = score > best
        best = max(best, score)
        checkpoint = {"model": model.state_dict(), "config": config, "epoch": epoch + 1,
                      "validation_foreground_iou": score, "best_validation_foreground_iou": best}
        torch.save(checkpoint, output / "last.pt")
        if selected:
            torch.save(checkpoint, output / "best.pt")
        record = {"epoch": epoch + 1, "loss": {k: v / steps_per_epoch for k, v in totals.items()},
                  "validation": scores, "selected": selected, "lr": scheduler.get_last_lr()[0],
                  "seconds": time.perf_counter() - started,
                  "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.startswith("cuda") else None}
        with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
    return output / "best.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--root")
    args = parser.parse_args()
    config = load_config(args.config)
    for key in ("device", "seed", "root"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    train(config, args.output)


if __name__ == "__main__":
    main()
