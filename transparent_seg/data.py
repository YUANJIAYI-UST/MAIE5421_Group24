"""Strict label decoding and reproducible, geometry-aligned preprocessing."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PERTURBATIONS = [(op, value) for op, values in {
    "brightness": [0.6, 0.8, 1.2, 1.4],
    "contrast": [0.5, 0.75, 1.25, 1.5],
    "gamma": [0.6, 0.8, 1.2, 1.6],
}.items() for value in values]


def decode_mask(path, encoding="trans10k_v1_rgb", void_values=()):
    """Return foreground and validity separately. 255 is NEVER implicit void.

    RGB channel sums 0/255/765 correspond to official means 0/85/255.
    Do not convert RGB masks to luminance: e.g. red also encodes class 1.
    void_values are documented channel MEANS, not channel sums.
    """
    with Image.open(path) as im:
        if encoding == "trans10k_v1_rgb":
            code = np.asarray(im.convert("RGB"), dtype=np.int16).sum(axis=2)
            mapping = {0: 0, 255: 1, 765: 1}
            void_codes = {int(v * 3) for v in void_values}
        elif encoding in ("binary_0255", "binary_01"):
            arr = np.asarray(im)
            if arr.ndim != 2:
                raise ValueError(f"Binary encoding requires a single-channel mask: {path}")
            code = arr.astype(np.int16)
            mapping = {0: 0, (255 if encoding == "binary_0255" else 1): 1}
            void_codes = set(void_values)
        else:
            raise ValueError(f"Unsupported mask encoding: {encoding}")
    if void_codes & mapping.keys():
        raise ValueError("Void values collide with background/foreground codes")
    unknown = set(np.unique(code).tolist()) - mapping.keys() - void_codes
    if unknown:
        raise ValueError(f"Unexpected mask codes in {path}: {sorted(unknown)}")
    valid = ~np.isin(code, list(void_codes))
    target = np.zeros(code.shape, dtype=np.float32)
    for value, label in mapping.items():
        target[code == value] = label
    return target, valid


def perturb(image, operation, value):
    """Float RGB in [0,1]. Contrast is around per-image, per-channel mean."""
    x = np.asarray(image, dtype=np.float32)
    if operation == "brightness":
        result = x * value
    elif operation == "contrast":
        center = x.mean(axis=(0, 1), keepdims=True)
        result = center + value * (x - center)
    elif operation == "gamma":
        result = np.power(x, value)
    else:
        raise ValueError(operation)
    return np.clip(result, 0, 1).astype(np.float32)


def image_tensor(image, size, condition=None):
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    if condition:
        x = perturb(x, *condition)
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())


def read_manifest(path, root=None):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("samples"):
        raise ValueError(f"Empty/unsupported manifest: {path}")
    ids = [sample["id"] for sample in manifest["samples"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    base = Path(root or manifest["root"]).resolve()
    for sample in manifest["samples"]:
        for key in ("image", "mask"):
            p = (base / sample[key]).resolve()
            if not p.is_relative_to(base) or not p.is_file():
                raise ValueError(f"Missing/outside-root {key}: {p}")
    return manifest, base


class SegmentationDataset(Dataset):
    def __init__(self, manifest, size=512, training=False, augmentation=False,
                 seed=42, root=None):
        self.meta, self.root = read_manifest(manifest, root)
        if training and self.meta["split"] != "train":
            raise ValueError("Training requires a train manifest")
        self.samples = self.meta["samples"]
        self.size, self.training, self.augmentation = size, training, augmentation
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        with Image.open(self.root / item["image"]) as im:
            image = im.convert("RGB")
        target, valid = decode_mask(self.root / item["mask"], self.meta["encoding"],
                                    self.meta.get("void_values", []))
        if image.size != (target.shape[1], target.shape[0]):
            raise ValueError(f"Image/mask geometry differs: {item['id']}")
        # Stateless augmentation: independent of loader workers and model RNG.
        token = f"{self.seed}/{self.epoch}/{item['id']}".encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "big"))
        condition = None
        if self.training and self.augmentation and rng.random() < 0.5:
            condition = (rng.choice(["brightness", "contrast", "gamma"]), rng.uniform(0.8, 1.2))
        tensor = image_tensor(image, self.size, condition)
        if self.training:
            target = np.asarray(Image.fromarray(target).resize(
                (self.size, self.size), Image.Resampling.NEAREST)).copy()
            valid = np.asarray(Image.fromarray(valid.astype(np.uint8)).resize(
                (self.size, self.size), Image.Resampling.NEAREST)).astype(bool)
        return {"image": tensor, "target": torch.from_numpy(target.copy())[None],
                "valid": torch.from_numpy(valid.copy())[None], "id": item["id"],
                "difficulty": item.get("difficulty", "unspecified")}
