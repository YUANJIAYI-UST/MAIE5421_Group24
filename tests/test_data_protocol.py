import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from transparent_seg.data import SegmentationDataset, read_manifest
from transparent_seg.prepare import audit_trans10k
from transparent_seg.train import load_config, train


def make_fixture(tmp_path):
    root = tmp_path / "data"
    for index, split in enumerate(("train", "validation", "test")):
        groups = [""] if split == "train" else ["easy", "hard"]
        for j, group in enumerate(groups):
            folder = root / split / group
            (folder / "images").mkdir(parents=True)
            (folder / "masks").mkdir()
            # Deliberately unique decoded RGB across splits.
            Image.fromarray(np.full((40, 64, 3), index * 50 + j, np.uint8)).save(folder / "images/001.jpg")
            mask = np.zeros((40, 64, 3), np.uint8)
            mask[10:30, 10:50] = 255
            Image.fromarray(mask).save(folder / "masks/001_mask.png")
    output = tmp_path / "manifests"
    audit_trans10k(root, output, allow_partial=True)
    return root, output


def test_manifest_keeps_original_eval_resolution_and_nearest_labels(tmp_path):
    _, output = make_fixture(tmp_path)
    training = SegmentationDataset(output / "train.json", size=32, training=True)
    validation = SegmentationDataset(output / "validation.json", size=32)
    assert training[0]["target"].shape == (1, 32, 32)
    assert validation[0]["target"].shape == (1, 40, 64)
    assert set(training[0]["target"].unique().tolist()) == {0, 1}
    with pytest.raises(ValueError, match="Training requires"):
        SegmentationDataset(output / "test.json", training=True)


def test_audit_rejects_partial_and_cross_split_duplicates(tmp_path):
    root, output = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="expected 5000"):
        audit_trans10k(root, output)
    source = root / "train/images/001.jpg"
    target = root / "validation/easy/images/001.jpg"
    target.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="Cross-split"):
        audit_trans10k(root, output, allow_partial=True)


def test_manifest_blocks_outside_root(tmp_path):
    _, output = make_fixture(tmp_path)
    path = output / "train.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["samples"][0]["image"] = "../outside.jpg"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside-root"):
        read_manifest(path)


def test_training_rejects_test_selection_and_partial_research(tmp_path):
    _, output = make_fixture(tmp_path)
    config = load_config("configs/A_baseline.json")
    config["train_manifest"] = str(output / "train.json")
    config["val_manifest"] = str(output / "test.json")
    with pytest.raises(ValueError, match="never test"):
        train(config, tmp_path / "run")
    config["val_manifest"] = str(output / "validation.json")
    with pytest.raises(ValueError, match="complete official"):
        train(config, tmp_path / "run")


def test_augmentation_reproducible_and_masks_unchanged(tmp_path):
    _, output = make_fixture(tmp_path)
    augmented = SegmentationDataset(output / "train.json", size=32, training=True, augmentation=True)
    plain = SegmentationDataset(output / "train.json", size=32, training=True)
    for epoch in range(4):
        augmented.epoch = epoch
        first, second = augmented[0], augmented[0]
        assert first["image"].equal(second["image"])
        assert first["target"].equal(plain[0]["target"])
        assert first["valid"].equal(plain[0]["valid"])
