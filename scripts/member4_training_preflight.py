"""Independent 20-step C resource check; never a formal C checkpoint."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch

import transparent_seg.train as trainer
from transparent_seg.data import SegmentationDataset, read_manifest
from transparent_seg.evaluate import load_checkpoint, predict
from transparent_seg.prepare import write_json, sha256_file
from transparent_seg.protocol import validate_frozen_protocol, SHARED_INITIAL_SHA256


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = trainer.load_config("configs/member4_full_C_20261115.json")
    train_meta, _ = read_manifest(config["train_manifest"])
    val_meta, val_root = read_manifest(config["val_manifest"])
    validate_frozen_protocol(config, train_meta, val_meta)
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "BF16 CUDA is required")
    selected = json.loads(Path("runs/member4_stage2_20261115_v2/selection.json").read_text(encoding="utf-8"))
    cache_checks = []
    for split, manifest, cache in (("train", config["train_manifest"], config["train_cache"]),
                                  ("validation", config["val_manifest"], config["validation_cache"])):
        training = split == "train"
        raw = SegmentationDataset(manifest, 512, training, training, 42)
        cached = SegmentationDataset(manifest, 512, training, training, 42, cache_dir=cache)
        indices = {s["id"]: i for i, s in enumerate(raw.samples)}
        for sample in selected[split]:
            index = indices[sample["id"]]
            a, b = raw[index], cached[index]
            require(all(torch.equal(a[k], b[k]) for k in ("image", "target", "valid")),
                    "Full-cache tensor mismatch: " + sample["id"])
        cache_checks.append({"split": split, "cache_members": len(cached),
                             "real_input_pairs_exact": len(selected[split]),
                             "index_sha256": sha256_file(Path(cache) / "index.json")})
        print(f"Full {split} cache verified: {len(cached)} entries, 20 exact real input pairs", flush=True)
    write_json(output / "full_cache_checks.json", cache_checks)
    pilot = {k: v for k, v in config.items() if k not in ("frozen_protocol", "frozen_protocol_sha256")}
    pilot.update(purpose="pilot", epochs=1, max_steps_per_epoch=20, train_progress_every=5)
    write_json(output / "pilot_config.json", pilot)
    observations, gradients, live_models = [], [], []
    original_model, original_clip = trainer.TransparentSegformer, torch.nn.utils.clip_grad_norm_

    def observe(model, inputs, outputs):
        if model.training:
            entry = {"batch_shape": list(inputs[0].shape), "logits_dtype": str(outputs["logits"].dtype),
                     "cuda_autocast": torch.is_autocast_enabled("cuda"),
                     "requires_grad": outputs["logits"].requires_grad}
            require(entry["batch_shape"] == [8, 3, 512, 512], "Unexpected preflight batch shape")
            require(entry["logits_dtype"] == "torch.bfloat16" and entry["cuda_autocast"],
                    "Preflight did not execute BF16 autocast")
            observations.append(entry)

    def factory(*args, **kwargs):
        model = original_model(*args, **kwargs)
        model.register_forward_hook(observe)
        live_models.append(model)
        return model

    def clip(parameters, *args, **kwargs):
        params = list(parameters)
        require(all(p.grad is not None for p in params if p.requires_grad), "Missing parameter gradients")
        norm = original_clip(params, *args, **kwargs)
        require(bool(torch.isfinite(norm)), "Non-finite gradient norm")
        gradients.append(float(norm.item()))
        return norm

    training_start = time.perf_counter()
    with patch.object(trainer, "TransparentSegformer", side_effect=factory), patch.object(
            torch.nn.utils, "clip_grad_norm_", side_effect=clip):
        best = trainer.train(pilot, output / "pilot_C_20_steps")
    pilot_wall_seconds = time.perf_counter() - training_start
    history = [json.loads(s) for s in (best.parent / "history.jsonl").read_text().splitlines()]
    require(len(history) == 1 and history[0]["global_optimizer_step"] == 20, "Preflight budget mismatch")
    record = history[0]
    require(record["validation"]["all"]["images"] == 995, "Preflight validation incomplete")
    require(len(observations) == len(gradients) == 20, "Incomplete forward/gradient observations")
    environment = json.loads((best.parent / "environment.json").read_text())
    require(environment["initial_shared_state_sha256"] == SHARED_INITIAL_SHA256, "Initialization differs from A")
    model, checkpoint = load_checkpoint(best, "cuda")
    require(checkpoint["config"]["purpose"] == "pilot", "Pilot checkpoint must be labelled")
    require(all(bool(torch.isfinite(v).all()) for v in checkpoint["model"].values()), "Non-finite saved weights")
    sample = selected["validation"][0]
    with Image.open(val_root / sample["image"]) as im:
        image = im.convert("RGB")
    in_memory = predict(live_models[0], image, 512, "cuda")
    reloaded = predict(model, image, 512, "cuda")
    require(np.array_equal(in_memory, reloaded), "Reloaded model predictions differ")
    last = torch.load(best.parent / "last.pt", map_location="cpu", weights_only=True)
    require(all(torch.equal(last["model"][k], v) for k, v in checkpoint["model"].items()),
            "One-epoch preflight best/last disagree")
    report = {"status": "passed", "purpose": "independent resource/gradient/reload diagnostic",
        "formal_training_started": False, "steps": 20, "batch_size": 8, "size": 512,
        "precision": "bfloat16 autocast, float32 master parameters", "augmentation": True, "boundary": False,
        "initial_shared_state_sha256": environment["initial_shared_state_sha256"],
        "full_validation_images": 995, "forward_observations": observations, "gradient_norms": gradients,
        "reload_exact": True, "reload_sample_id": sample["id"], "cache_checks": cache_checks,
        "train_seconds": record["train_seconds"], "validation_seconds": record["validation_seconds"],
        "steady_microbatch_seconds": record["steady_microbatch_seconds"],
        "microbatch_seconds": record["microbatch_seconds"],
        "peak_cuda_memory_mb": record["peak_cuda_memory_mb"],
        "peak_cuda_reserved_mb": record["peak_cuda_reserved_mb"], "gpu": torch.cuda.get_device_name(),
        "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 2**20,
        "pilot_wall_seconds": pilot_wall_seconds, "total_seconds": time.perf_counter() - started,
        "pilot_best_sha256": sha256_file(best), "pilot_last_sha256": sha256_file(best.parent / "last.pt"),
        "formal_config_sha256": sha256_file("configs/member4_full_C_20261115.json"),
        "script_sha256": sha256_file(__file__),
        "limitations": [
            "Pilot uses an independent 20-step LR schedule; formal C remains 18600 steps, warmup930.",
            "No pilot weights may be used for formal C or reported as formal benchmark results.",
            "20-step timing cannot guarantee sustained laptop thermals or overnight throughput.",
        ]}
    write_json(output / "preflight_report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "steps", "reload_exact", "train_seconds",
          "validation_seconds", "steady_microbatch_seconds", "peak_cuda_memory_mb", "total_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        main(args.output)
    except Exception as exc:
        # Retain partial run evidence, and do not retry with changed batch/precision.
        path = Path(args.output)
        if path.exists() and not (path / "preflight_report.json").exists():
            write_json(path / "preflight_failure.json", {"status": "failed", "error": str(exc),
                "formal_training_started": False})
        raise
