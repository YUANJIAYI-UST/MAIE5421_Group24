import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from transparent_seg.cache import build_cache
from transparent_seg.data import SegmentationDataset, validate_derivative_use
from transparent_seg.evaluate import evaluate, evaluate_selection
from transparent_seg.prepare import sha256_file
from transparent_seg.protocol import validate_frozen_protocol


def fixture_manifest(tmp_path, split="train"):
    samples = []
    for i in range(3):
        rgb = np.random.default_rng(i).integers(0, 256, (39, 57, 3), dtype=np.uint8)
        mask = np.zeros((39, 57), np.uint8)
        mask[8:30, 4:40] = 1
        mask[:4] = 2  # Explicit void, including all-void sample.
        if i == 2:
            mask[:] = 2
        image, label = tmp_path / f"image{i}.png", tmp_path / f"mask{i}.png"
        Image.fromarray(rgb).save(image)
        Image.fromarray(mask).save(label)
        samples.append({"id": str(i), "image": image.name, "mask": label.name,
            "image_sha256": sha256_file(image), "mask_sha256": sha256_file(label),
            "difficulty": "easy" if i == 0 else "hard"})
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "root": str(tmp_path), "split": split,
        "dataset": "fixture", "encoding": "binary_01", "void_values": [2], "samples": samples}), encoding="utf-8")
    return path


@pytest.mark.parametrize("training", [True, False])
def test_cache_exact_tensors_and_corruption(tmp_path, training):
    path = fixture_manifest(tmp_path, "train" if training else "validation")
    cache = tmp_path / "cache"
    build_cache(path, cache, size=32, training=training, workers=2)
    plain = SegmentationDataset(path, 32, training, augmentation=True)
    cached = SegmentationDataset(path, 32, training, augmentation=True, cache_dir=cache)
    for epoch in range(6):
        plain.epoch = cached.epoch = epoch
        for i in range(len(plain)):
            a, b = plain[i], cached[i]
            for key in ("image", "target", "valid"):
                assert torch.equal(a[key], b[key])
    with pytest.raises(ValueError, match="match manifest"):
        SegmentationDataset(path, 64, training, cache_dir=cache)
    with (cache / "000000.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        SegmentationDataset(path, 32, training, cache_dir=cache)


def test_cache_rejects_changed_source(tmp_path):
    path = fixture_manifest(tmp_path)
    (tmp_path / "image0.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        build_cache(path, tmp_path / "cache", 32, True)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_selection_exact_original_region_counts(tmp_path, device):
    path = fixture_manifest(tmp_path, "validation")
    cache = tmp_path / "cache"
    build_cache(path, cache, 32)
    class Model(torch.nn.Module):
        def forward(self, x):
            return {"logits": torch.nn.functional.avg_pool2d(x[:, :1], 4)}
    model = Model().to(device)
    expected = evaluate(model, path, 32, device)["scores"]
    actual = evaluate_selection(model, SegmentationDataset(path, 32, cache_dir=cache), device)["scores"]
    for group in expected:
        for key, value in expected[group].items():
            if key not in ("boundary_iou", "boundary_images"):
                assert actual[group][key] == value
        assert actual[group]["boundary_iou"] is None


def test_frozen_derivative_guard(tmp_path):
    train, val = {}, {}
    config = {"purpose": "research", "data_protocol_id": "reviewed", "boundary": False,
              "augmentation": False, "epochs": 30, "expected_manifest_sha256": {}}
    for split, meta, key in (("train", train, "train_manifest"), ("validation", val, "val_manifest")):
        meta.update(split=split, samples=[{"id": "a"}], data_protocol={"id": "reviewed"})
        path = tmp_path / f"{split}.json"
        path.write_text(json.dumps(meta), encoding="utf-8")
        config[key] = str(path)
        config["expected_manifest_sha256"][split] = sha256_file(path)
    with pytest.raises(ValueError, match="pilot-only"):
        validate_derivative_use(config, train, val)
    protocol = {"scope": "member2_full_A_baseline", "config": dict(config), "data_protocol_id": "reviewed",
                "data": {k: {"count": 1, "sha256": v} for k, v in config["expected_manifest_sha256"].items()}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    config.update(frozen_protocol=str(path), frozen_protocol_sha256=sha256_file(path))
    validate_derivative_use(config, train, val)
    assert validate_frozen_protocol(config, train, val)
    with pytest.raises(ValueError, match="Configuration differs"):
        validate_frozen_protocol({**config, "epochs": 1}, train, val)
    with path.open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash-pinned"):
        validate_frozen_protocol(config, train, val)
