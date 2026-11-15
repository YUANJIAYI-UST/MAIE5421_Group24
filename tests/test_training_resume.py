import json
import random

import numpy as np
import pytest
import torch

import transparent_seg.train as trainer
from scripts.make_smoke_data import main as make_data
from transparent_seg.checkpoint import capture_rng, restore_rng


class TinyModel(torch.nn.Module):
    def __init__(self, boundary=False, *args):
        super().__init__()
        self.encoder_revision = "test"
        self.conv = torch.nn.Conv2d(3, 1, 1)
        self.dropout = torch.nn.Dropout(0.2)
        self.boundary = torch.nn.Conv2d(3, 1, 1) if boundary else None

    def forward(self, x, return_boundary=False):
        result = {"logits": self.conv(self.dropout(x))}
        if return_boundary:
            result["boundary_logits"] = self.boundary(x)
        return result


@pytest.fixture
def run_config(tmp_path, monkeypatch):
    config = trainer.load_config("configs/smoke.json")
    monkeypatch.chdir(tmp_path)
    make_data()
    monkeypatch.setattr(trainer, "TransparentSegformer", TinyModel)
    config.update(epochs=3, batch_size=5, size=32, cpu_threads=1)
    return config


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_resume_matches_uninterrupted(run_config, accumulation):
    run_config["gradient_accumulation_steps"] = accumulation
    trainer.train(run_config, "full")
    trainer.train(run_config, "partial", stop_after_epoch=1)
    trainer.train(run_config, "resumed", resume="partial/last.pt")
    full = torch.load("full/last.pt", weights_only=True)
    resumed = torch.load("resumed/last.pt", weights_only=True)
    assert_equal(full, resumed)
    assert full["global_step"] == 3 * ((3 + accumulation - 1) // accumulation)
    full_history = [json.loads(x) for x in open("full/history.jsonl")]
    resumed_history = [json.loads(x) for x in open("resumed/history.jsonl")]
    for a, b in zip(full_history[1:], resumed_history):
        for key in ("loss", "validation", "lr", "selected", "global_optimizer_step"):
            assert a[key] == b[key]
    assert_equal(torch.load("full/best.pt", weights_only=True),
                 torch.load("resumed/best.pt", weights_only=True))


def test_resume_rejects_budget_and_manifest_changes(run_config):
    trainer.train(run_config, "partial", stop_after_epoch=1)
    with pytest.raises(ValueError, match="config"):
        trainer.train({**run_config, "epochs": 4}, "bad", resume="partial/last.pt")
    path = run_config["train_manifest"]
    with open(path, "a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hashes"):
        trainer.train(run_config, "bad", resume="partial/last.pt")


def test_rng_roundtrip():
    generator = torch.Generator().manual_seed(18)
    state = capture_rng(generator)
    expected = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    restore_rng(state, generator)
    actual = (random.random(), np.random.rand(), torch.rand(2), torch.rand(2, generator=generator))
    assert_equal(expected, actual)


def test_resume_keeps_earlier_best_and_rejects_legacy(run_config, monkeypatch):
    monkeypatch.setattr(trainer, "evaluate", lambda *a, **k: {
        "scores": {"all": {"foreground_iou": 0.5}}})
    trainer.train(run_config, "partial", stop_after_epoch=1)
    trainer.train(run_config, "resumed", resume="partial/last.pt")
    assert torch.load("resumed/best.pt", weights_only=True)["epoch"] == 1
    checkpoint = torch.load("partial/last.pt", weights_only=True)
    checkpoint.pop("checkpoint_version")
    torch.save(checkpoint, "legacy.pt")
    with pytest.raises(ValueError, match="lacks resumable"):
        trainer.train(run_config, "bad", resume="legacy.pt")


def test_overfit_subset_stays_training_only(run_config):
    trainer.train({**run_config, "overfit_samples": 2}, "overfit", stop_after_epoch=1)
    subset = json.load(open("overfit/overfit_train.json"))
    assert subset["split"] == "train" and len(subset["samples"]) == 2
    record = json.loads(open("overfit/history.jsonl").readline())
    assert record["overfit_training"] is not None
    assert record["validation"]["all"]["images"] == 4


def test_sparse_validation_is_diagnostic_only(run_config):
    with pytest.raises(ValueError, match="only allowed"):
        trainer.train({**run_config, "validation_interval": 2}, "bad")
    trainer.train({**run_config, "overfit_samples": 2, "validation_interval": 3}, "sparse")
    records = [json.loads(line) for line in open("sparse/history.jsonl")]
    assert records[0]["validation"] is not None
    assert records[1]["validation"] is None and records[1]["selected"] is False
    assert records[2]["validation"] is not None
    assert all(r["overfit_training"] is not None for r in records)
