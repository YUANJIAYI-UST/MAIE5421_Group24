"""Explicit hash-pinned acceptance of a complete derivative experiment."""
import json
from pathlib import Path

from .prepare import sha256_file


def validate_frozen_protocol(config, training, validation):
    path = config.get("frozen_protocol")
    if not path or sha256_file(path) != config.get("frozen_protocol_sha256"):
        raise ValueError("Research derivative requires a hash-pinned frozen protocol")
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = {k: v for k, v in config.items() if k not in ("frozen_protocol", "frozen_protocol_sha256")}
    scope = protocol.get("scope")
    if protocol.get("config") != actual or scope not in {
            "member2_full_A_baseline", "member3_B02_boundary_grid"}:
        raise ValueError("Configuration differs from a supported frozen protocol")
    if (config.get("purpose") != "research"
            or config.get("max_steps_per_epoch") is not None
            or config.get("overfit_samples") is not None
            or config.get("validation_interval", 1) != 1):
        raise ValueError("Frozen research protocol requires full training and validation")
    if scope == "member2_full_A_baseline":
        if config.get("boundary") or config.get("augmentation") or config.get("ablation") != "A":
            raise ValueError("Frozen A protocol cannot authorize boundary or augmentation settings")
    else:
        expected_grid = {(1, 0.1), (1, 0.3), (2, 0.1), (2, 0.3)}
        cell = (config.get("boundary_radius"), config.get("boundary_weight"))
        fixed = {"ablation": "B", "boundary": True, "augmentation": False,
                 "seed": 42, "size": 512, "batch_size": 8, "epochs": 30,
                 "lr": 6e-5, "weight_decay": 0.01,
                 "gradient_accumulation_steps": 1, "validation_interval": 1}
        device = config.get("device")
        expected_torch = {"cuda": "2.7.1+cu128", "mps": "2.7.1"}.get(device)
        if (cell not in expected_grid
                or any(config.get(key) != value for key, value in fixed.items())
                or device not in ("cuda", "mps")
                or not config.get("amp")
                or config.get("amp_dtype") != "bfloat16"
                or config.get("expected_versions", {}).get("torch") != expected_torch):
            raise ValueError("B02 protocol only authorizes its four frozen grid cells and shared controls")
    for split, meta, key in (("train", training, "train_manifest"), ("validation", validation, "val_manifest")):
        expected = protocol.get("data", {}).get(split, {})
        if (expected.get("sha256") != sha256_file(config[key]) or expected.get("count") != len(meta["samples"])
                or meta.get("synthetic") or meta.get("split") != split
                or meta.get("data_protocol", {}).get("id") != protocol.get("data_protocol_id")):
            raise ValueError("Frozen protocol data identity/count mismatch")
    return True
