import copy
import json
from pathlib import Path

import pytest

import transparent_seg.protocol as guard
from transparent_seg.prepare import write_json, sha256_file


@pytest.fixture
def frozen_c(monkeypatch):
    config = json.loads(Path("configs/member4_full_C_20261115.json").read_text())
    meta = {s: {"split": s, "samples": [{}] * n, "synthetic": False,
                "data_protocol": {"id": config["data_protocol_id"]}}
            for s, n in (("train", 4959), ("validation", 995))}
    actual_hash = guard.sha256_file
    fake_hashes = {config["train_manifest"]: config["expected_manifest_sha256"]["train"],
                   config["val_manifest"]: config["expected_manifest_sha256"]["validation"]}
    monkeypatch.setattr(guard, "sha256_file", lambda p: fake_hashes.get(str(p)) or actual_hash(p))
    return config, meta


def repin(tmp_path, config, **updates):
    protocol = json.loads(Path(config["frozen_protocol"]).read_text())
    changed = copy.deepcopy(config)
    changed.update(updates)
    protocol["config"] = {k: v for k, v in changed.items()
                          if k not in ("frozen_protocol", "frozen_protocol_sha256")}
    path = tmp_path / "changed_protocol.json"
    write_json(path, protocol)
    changed.update(frozen_protocol=str(path), frozen_protocol_sha256=sha256_file(path))
    return changed


def test_explicit_c_and_initialization(frozen_c):
    config, meta = frozen_c
    assert guard.validate_frozen_protocol(config, meta["train"], meta["validation"])
    guard.validate_initialization(config, guard.SHARED_INITIAL_SHA256)
    with pytest.raises(ValueError, match="initialization"):
        guard.validate_initialization(config, "wrong")
    guard.validate_initialization({}, "legacy")  # No behavior change for unfrozen pilots.


@pytest.mark.parametrize("change", [{"batch_size": 4}, {"epochs": 1}, {"size": 256},
    {"seed": 43}, {"augmentation": False}, {"boundary": True},
    {"max_steps_per_epoch": 20}, {"validation_interval": 2}, {"lr": 1e-4}])
def test_repinning_c_cannot_change_comparison(tmp_path, frozen_c, change):
    config, meta = frozen_c
    changed = repin(tmp_path, config, **change)
    with pytest.raises(ValueError):
        guard.validate_frozen_protocol(changed, meta["train"], meta["validation"])


def test_c_requires_full_membership_and_budget(tmp_path, frozen_c):
    config, meta = frozen_c
    shorter = copy.deepcopy(meta["validation"])
    shorter["samples"].pop()
    with pytest.raises(ValueError, match="identity/count"):
        guard.validate_frozen_protocol(config, meta["train"], shorter)
    protocol = json.loads(Path(config["frozen_protocol"]).read_text())
    protocol["budget"]["optimizer_steps"] = 20
    path = tmp_path / "bad_budget.json"
    write_json(path, protocol)
    config.update(frozen_protocol=str(path), frozen_protocol_sha256=sha256_file(path))
    with pytest.raises(ValueError, match="budget"):
        guard.validate_frozen_protocol(config, meta["train"], meta["validation"])
