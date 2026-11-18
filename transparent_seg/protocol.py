"""Explicit hash-pinned acceptance of a complete derivative experiment."""
import json
from pathlib import Path

from .prepare import sha256_file

FOLLOWUP_SCOPES = {
    "group24_b03_full_B": ("B", "member1_final_20261029_v1", 4959, 995),
    "group24_b03_full_D": ("D", "member1_final_20261029_v1", 4959, 995),
    "group24_e03_isolated_A": ("A", "member1_final_20261029_d03", 3230, 646),
    "group24_e03_isolated_D": ("D", "member1_final_20261029_d03", 3230, 646),
}


def validate_frozen_protocol(config, training, validation):
    path = config.get("frozen_protocol")
    if not path or sha256_file(path) != config.get("frozen_protocol_sha256"):
        raise ValueError("Research derivative requires a hash-pinned frozen protocol")
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = {k: v for k, v in config.items() if k not in ("frozen_protocol", "frozen_protocol_sha256")}
    scope = protocol.get("scope")
    if protocol.get("config") != actual or scope not in {"member2_full_A_baseline", *FOLLOWUP_SCOPES}:
        raise ValueError("Configuration differs from frozen protocol")
    if (config.get("purpose") != "research"
            or config.get("max_steps_per_epoch") is not None or config.get("overfit_samples") is not None
            or config.get("validation_interval", 1) != 1):
        raise ValueError("Frozen protocol requires full training and validation")
    if scope == "member2_full_A_baseline":
        if config.get("boundary") or config.get("augmentation"):
            raise ValueError("Frozen A protocol requires both ablation switches off")
    else:
        ablation, protocol_id, train_count, val_count = FOLLOWUP_SCOPES[scope]
        fixed = {"ablation": ablation, "boundary": ablation in {"B", "D"}, "augmentation": ablation == "D",
                 "boundary_radius": 2, "boundary_weight": 0.1, "pretrained": True,
                 "size": 512, "seed": 42, "batch_size": 8, "epochs": 30, "lr": 6e-5,
                 "weight_decay": 0.01, "gradient_accumulation_steps": 1, "amp": True,
                 "amp_dtype": "bfloat16", "device": "cuda", "deterministic": True,
                 "validation_metrics": "region_only",
                 "encoder_revision": "80983a413c30d36a39c20203974ae7807835e2b4"}
        if (any(config.get(k) != v for k, v in fixed.items())
                or protocol.get("data_protocol_id") != protocol_id
                or config.get("data_protocol_id") != protocol_id
                or (len(training["samples"]), len(validation["samples"])) != (train_count, val_count)):
            raise ValueError("B03/E03 training controls or split differ from the frozen experiment")
    for split, meta, key in (("train", training, "train_manifest"), ("validation", validation, "val_manifest")):
        expected = protocol.get("data", {}).get(split, {})
        if (expected.get("sha256") != sha256_file(config[key]) or expected.get("count") != len(meta["samples"])
                or meta.get("synthetic") or meta.get("split") != split
                or meta.get("data_protocol", {}).get("id") != protocol.get("data_protocol_id")):
            raise ValueError("Frozen protocol data identity/count mismatch")
    return True
