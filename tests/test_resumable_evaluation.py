import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import transparent_seg.evaluate_resumable as runner
from transparent_seg.evaluate import evaluate, predict
from transparent_seg.metrics import SegmentationMetrics
from transparent_seg.prepare import sha256_file, write_json


class PixelModel(torch.nn.Module):
    def forward(self, x):
        return {"logits": torch.nn.functional.avg_pool2d(x[:, :1] - x[:, 1:2] * 0.4, 4)}


@pytest.mark.parametrize("case", ["empty", "full", "random", "void_band", "all_void", "no_safe"])
def test_prepared_target_matches_original(case):
    rng = np.random.default_rng(123)
    y = rng.random((61, 87)) > 0.4
    v = np.ones_like(y)
    if case == "empty": y[:] = False
    if case == "full": y[:] = True
    if case == "void_band": v[12:30] = False
    if case == "all_void": v[:] = False
    if case == "no_safe": v[:] = False; v[20, 20] = True
    prepared = runner.PreparedTarget(y, v)
    for p in (y, ~y, rng.random(y.shape) > 0.5):
        original = SegmentationMetrics()
        original.update(p, y, v)
        assert prepared.score(p) == original.compute()


def test_resized_predictions_equal_every_pixel():
    im = Image.fromarray(np.random.default_rng(9).integers(0, 256, (83, 111, 3), dtype=np.uint8))
    resized = im.resize((32, 32), Image.Resampling.BILINEAR)
    model = PixelModel().eval()
    for condition in runner.CONDITIONS.values():
        assert np.array_equal(predict(model, im, 32, "cpu", condition),
            runner.predict_resized(model, resized, (83, 111), 32, "cpu", condition))


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    samples = []
    rng = np.random.default_rng(91)
    for i in range(27):
        image, mask = tmp_path / f"{i}.png", tmp_path / f"{i}_mask.png"
        Image.fromarray(rng.integers(0, 256, (43 + i % 2, 57, 3), dtype=np.uint8)).save(image)
        y = rng.integers(0, 2, (43 + i % 2, 57), dtype=np.uint8)
        y[:3] = 2
        if i == 0: y[:] = 2
        Image.fromarray(y).save(mask)
        samples.append({"id": str(i), "image": image.name, "mask": mask.name,
            "image_sha256": sha256_file(image), "mask_sha256": sha256_file(mask),
            "difficulty": ("easy", "hard", "unspecified")[i % 3]})
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"schema_version": 1, "root": str(tmp_path), "split": "validation",
        "dataset": "test", "encoding": "binary_01", "void_values": [2], "samples": samples, "synthetic": True})
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    monkeypatch.setattr(runner, "load_checkpoint", lambda *args: (
        PixelModel().eval(), {"config": {"size": 32, "deterministic": True}}))
    return checkpoint, manifest


def strip_timing(path):
    rows = []
    for chunk in sorted((path / "chunks").glob("*.json")):
        rows.extend(json.loads(chunk.read_text())["body"]["rows"])
    return [{k: v for k, v in row.items() if k != "seconds"} for row in rows]


def test_resume_matches_continuous_and_legacy(tmp_path, experiment):
    checkpoint, manifest = experiment
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    a = runner.run(checkpoint, manifest, full, "cpu")
    paused = runner.run(checkpoint, manifest, resumed, "cpu", stop_after_chunks=1)
    assert paused["image_count"] == 25 and not paused["complete"]
    assert not (resumed / "clean/metrics.json").exists()
    (resumed / "chunks/000025.json.tmp").write_text('{"unfinished":')
    b = runner.run(checkpoint, manifest, resumed, "cpu", resume=True)
    assert a["scores"] == b["scores"] and strip_timing(full) == strip_timing(resumed)
    assert b["reused_chunks"] == 1 and b["computed_chunks"] == 1 and b["image_count"] == 27
    for name, condition in runner.CONDITIONS.items():
        reference = evaluate(PixelModel().eval(), manifest, 32, "cpu", condition=condition)
        assert a["scores"][name] == reference["scores"]
    again = runner.run(checkpoint, manifest, resumed, "cpu", resume=True)
    assert again["computed_chunks"] == 0 and again["scores"] == a["scores"]


def test_corrupt_chunks_are_recomputed(tmp_path, experiment):
    checkpoint, manifest = experiment
    out = tmp_path / "run"
    original = runner.run(checkpoint, manifest, out, "cpu", names=["clean"])
    (out / "chunks/000000.json").write_text('{"truncated":')
    recovered = runner.run(checkpoint, manifest, out, "cpu", names=["clean"], resume=True)
    assert recovered["computed_chunks"] == recovered["reused_chunks"] == 1
    assert recovered["scores"] == original["scores"]
    assert len(list((out / "invalid_chunks").glob("*.json"))) == 1
    path = out / "chunks/000025.json"
    envelope = json.loads(path.read_text())
    envelope["body"]["rows"][0]["id"] = "wrong"
    envelope["sha256"] = runner.digest(envelope["body"])
    path.write_text(json.dumps(envelope))
    recovered = runner.run(checkpoint, manifest, out, "cpu", names=["clean"], resume=True)
    assert recovered["computed_chunks"] == 1 and recovered["scores"] == original["scores"]


@pytest.mark.parametrize("change", ["weights", "manifest", "source", "conditions", "code", "chunk_size"])
def test_resume_refuses_changed_identity(tmp_path, monkeypatch, experiment, change):
    checkpoint, manifest = experiment
    out = tmp_path / "run"
    runner.run(checkpoint, manifest, out, "cpu", names=["clean"])
    kwargs = {"names": ["clean"], "resume": True}
    if change == "weights": checkpoint.write_bytes(b"other-weights")
    if change == "manifest": manifest.write_bytes(manifest.read_bytes() + b"\n")
    if change == "source": (tmp_path / "0.png").write_bytes(b"changed")
    if change == "conditions": kwargs["names"] = ["gamma_0.8"]
    if change == "code": monkeypatch.setattr(runner, "code_identity", lambda: {"changed": True})
    if change == "chunk_size": kwargs["chunk_size"] = 5
    with pytest.raises(ValueError, match="mismatch|differs"):
        runner.run(checkpoint, manifest, out, "cpu", **kwargs)


def test_exclusive_lock_releases(tmp_path):
    with runner.exclusive_run(tmp_path):
        with pytest.raises(RuntimeError, match="Another"):
            with runner.exclusive_run(tmp_path):
                pass
    with runner.exclusive_run(tmp_path):
        pass


def test_repair_pause_does_not_leave_stale_complete_exports(tmp_path, experiment):
    checkpoint, manifest = experiment
    out = tmp_path / "run"
    original = runner.run(checkpoint, manifest, out, "cpu", names=["clean"])
    (out / "chunks/000000.json").write_text("damaged")
    paused = runner.run(checkpoint, manifest, out, "cpu", names=["clean"],
                        resume=True, stop_after_chunks=1)
    assert not paused["complete"]
    assert not json.loads((out / "progress.json").read_text())["complete"]
    assert not (out / "clean/metrics.json").exists()
    assert list((out / "previous_exports").glob("*/clean/metrics.json"))
    recovered = runner.run(checkpoint, manifest, out, "cpu", names=["clean"], resume=True)
    assert recovered["scores"] == original["scores"]
