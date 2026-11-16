"""Explicit hash-pinned acceptance of a complete derivative experiment."""
import json
import math
from pathlib import Path

from .prepare import sha256_file

C_SCOPE = "member4_full_C_augmentation"
SHARED_INITIAL_SHA256 = "fd50559254d2d5fe9b16cbaa6f7fd32452faa4a7f3b9a3ec42f1a6fb6eedc835"
A_CONFIG_SHA256 = "5205af206df59eb2a436d46e8b24883be24be4296adf067f480fbd3f0eb9b9bc"
A_PROTOCOL_SHA256 = "baa62eb2c3ba5a7358caf0020a3740460ac5d885868acd113391d5bdb08e9eaf"


def validate_c_comparability(config, protocol):
    reference = protocol.get("reference_A", {})
    if (reference.get("config_sha256") != A_CONFIG_SHA256
            or reference.get("protocol_sha256") != A_PROTOCOL_SHA256
            or sha256_file(reference["config_path"]) != A_CONFIG_SHA256
            or sha256_file(reference["protocol_path"]) != A_PROTOCOL_SHA256):
        raise ValueError("C requires the original published A configuration/protocol")
    baseline = json.loads(Path(reference["config_path"]).read_text(encoding="utf-8"))
    common = {k: v for k, v in baseline.items() if k not in ("frozen_protocol", "frozen_protocol_sha256")}
    common.update(ablation="C", augmentation=True)
    actual = {k: v for k, v in config.items() if k not in ("frozen_protocol", "frozen_protocol_sha256")}
    if actual != common:
        raise ValueError("C may only change A's ablation name and augmentation flag")
    budget = {"epochs": 30, "batches_per_epoch": 620, "optimizer_steps": 18600, "warmup_steps": 930}
    if (protocol.get("budget") != budget
            or math.ceil(protocol["data"]["train"]["count"] / config["batch_size"]) != 620
            or protocol.get("initial_shared_state_sha256") != SHARED_INITIAL_SHA256
            or protocol.get("selection") !=
                "Every epoch: all 995 validation images, original-resolution pooled foreground IoU, threshold 0.5; strict improvement; earliest tie."):
        raise ValueError("C frozen budget, initialization or model selection differs")


def validate_initialization(config, initial_hash):
    """Apply the new C initialization guard without changing legacy A behavior."""
    if config.get("frozen_protocol"):
        protocol = json.loads(Path(config["frozen_protocol"]).read_text(encoding="utf-8"))
        if protocol.get("scope") == C_SCOPE and initial_hash != SHARED_INITIAL_SHA256:
            raise ValueError("C shared initialization does not match the published A run")


def validate_frozen_protocol(config, training, validation):
    path = config.get("frozen_protocol")
    if not path or sha256_file(path) != config.get("frozen_protocol_sha256"):
        raise ValueError("Research derivative requires a hash-pinned frozen protocol")
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    actual = {k: v for k, v in config.items() if k not in ("frozen_protocol", "frozen_protocol_sha256")}
    scope = protocol.get("scope")
    if protocol.get("config") != actual or scope not in ("member2_full_A_baseline", C_SCOPE):
        raise ValueError("Configuration differs from frozen protocol")
    if (config.get("purpose") != "research" or config.get("boundary")
            or bool(config.get("augmentation")) != (scope == C_SCOPE)
            or config.get("max_steps_per_epoch") is not None or config.get("overfit_samples") is not None
            or config.get("validation_interval", 1) != 1):
        raise ValueError("Frozen protocol requires full training and validation")
    for split, meta, key in (("train", training, "train_manifest"), ("validation", validation, "val_manifest")):
        expected = protocol.get("data", {}).get(split, {})
        if (expected.get("sha256") != sha256_file(config[key]) or expected.get("count") != len(meta["samples"])
                or meta.get("synthetic") or meta.get("split") != split
                or meta.get("data_protocol", {}).get("id") != protocol.get("data_protocol_id")):
            raise ValueError("Frozen protocol data identity/count mismatch")
    if scope == C_SCOPE:
        validate_c_comparability(config, protocol)
    return True
