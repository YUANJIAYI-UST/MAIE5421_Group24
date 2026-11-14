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
    if protocol.get("config") != actual or protocol.get("scope") != "member2_full_A_baseline":
        raise ValueError("Configuration differs from frozen protocol")
    if (config.get("purpose") != "research" or config.get("boundary") or config.get("augmentation")
            or config.get("max_steps_per_epoch") is not None or config.get("overfit_samples") is not None
            or config.get("validation_interval", 1) != 1):
        raise ValueError("Frozen A protocol requires full training and validation")
    for split, meta, key in (("train", training, "train_manifest"), ("validation", validation, "val_manifest")):
        expected = protocol.get("data", {}).get(split, {})
        if (expected.get("sha256") != sha256_file(config[key]) or expected.get("count") != len(meta["samples"])
                or meta.get("synthetic") or meta.get("split") != split
                or meta.get("data_protocol", {}).get("id") != protocol.get("data_protocol_id")):
            raise ValueError("Frozen protocol data identity/count mismatch")
    return True
