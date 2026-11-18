"""Audit the existing augmentation path on fixed real images; never train a model.

Run from the repository root. The output and diagnostic caches must be new.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import transparent_seg.data as data
from transparent_seg.cache import build_cache
from transparent_seg.prepare import sha256_file, write_json


SIZE, SEED = 512, 42
OPS = ("brightness", "contrast", "gamma")
ENDPOINTS = [(op, value) for op in OPS for value in (0.8, 1.2)]
RAW_EPOCHS = (0, 1, 2, 9, 19, 29)  # Zero-based, fixed before any audit.
STAGE1 = Path("runs/member4_stage1_20261115")
MANIFESTS = Path("manifests/member1_final_20261029/v1")
EXPECTED = {
    "train": (4959, "7a3094e0c1aeacf0edd7335f196af79eacd4b551cd9f78cce2cf7c903e910ca2"),
    "validation": (995, "861c24027bfddcfba031e253a6518343010e31d296fb66c010f8d88ee9a90497"),
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def digest_tensor(tensor):
    a = tensor.numpy()
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


class Audit:
    def __init__(self):
        self.counts = Counter()

    def check(self, name, ok, detail=""):
        if not ok:
            raise ValueError(f"{name}: {detail}")
        self.counts[name] += 1


class ForcedRNG:
    """Audit-only injection to exercise endpoints and the probability boundary."""
    def __init__(self, audit, operation="brightness", value=1.1, draw=0.0):
        self.audit, self.operation, self.value, self.draw = audit, operation, value, draw

    def random(self):
        return self.draw

    def choice(self, choices, *args, **kwargs):
        self.audit.check("uniform_three_way_choice", tuple(choices) == OPS and not args and not kwargs)
        return self.operation

    def uniform(self, low, high):
        self.audit.check("uniform_parameter_bounds", (low, high) == (0.8, 1.2))
        return self.value


def traced_item(dataset, index, rgb, audit, forced=None):
    """Observe the production Dataset and perturb function without replacing their math."""
    calls = []
    photo = data.perturb
    transformed = [rgb]

    def observe(x, operation, value):
        audit.check("resize_before_augmentation", x.shape == (SIZE, SIZE, 3))
        audit.check("photometric_input_is_unnormalized_rgb",
                    x.dtype == np.float32 and np.array_equal(x, rgb))
        audit.check("only_declared_operation_and_range",
                    operation in OPS and np.isfinite(value) and 0.8 <= value <= 1.2)
        y = photo(x, operation, value)
        audit.check("photo_output_finite_range_shape",
                    y.shape == x.shape and np.isfinite(y).all() and y.min() >= 0 and y.max() <= 1)
        transformed[0] = y
        calls.append((str(operation), float(value)))
        return y

    injection = (patch.object(data.np.random, "default_rng", return_value=forced)
                 if forced is not None else nullcontext())
    with injection, patch.object(data, "perturb", side_effect=observe):
        item = dataset[index]
    audit.check("at_most_one_augmentation", len(calls) <= 1)
    expected = torch.from_numpy(((transformed[0] - MEAN) / STD).transpose(2, 0, 1).copy())
    audit.check("normalization_after_augmentation_exact", torch.equal(item["image"], expected))
    audit.check("finite_float32_input", item["image"].dtype == torch.float32
                and bool(torch.isfinite(item["image"]).all()))
    return item, (calls[0] if calls else None)


def same_items(a, b, audit, prefix):
    audit.check(prefix + "_id", a["id"] == b["id"])
    for key in ("image", "target", "valid"):
        audit.check(prefix + "_" + key, torch.equal(a[key], b[key]))


def normalized_image(rgb):
    # Quantization is only for PNG presentation, never fed back into the audit.
    return Image.fromarray(np.rint(rgb * 255).clip(0, 255).astype(np.uint8))


def font(size=17):
    path = Path("C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def make_visuals(output, index, sample, rgb, target):
    views = {f"{op}/{value}": normalized_image(data.perturb(rgb, op, value))
             for op, value in ENDPOINTS}
    views["clean"] = normalized_image(rgb)
    views["target"] = Image.fromarray(target[0].numpy().astype(np.uint8) * 255).convert("RGB")
    labels = {"clean": "原图（缩放后）", "target": "共同标签：白色为前景",
              "brightness/0.8": "亮度 ×0.8", "brightness/1.2": "亮度 ×1.2",
              "contrast/0.8": "对比度 ×0.8", "contrast/1.2": "对比度 ×1.2",
              "gamma/0.8": "Gamma 0.8（偏亮）", "gamma/1.2": "Gamma 1.2（偏暗）"}
    keys = ("clean", "brightness/0.8", "contrast/0.8", "gamma/0.8",
            "target", "brightness/1.2", "contrast/1.2", "gamma/1.2")
    full = Image.new("RGB", (1040, 660), "white")
    draw = ImageDraw.Draw(full)
    draw.text((10, 8), f"{index + 1:02d}  {sample['id']}", fill="black", font=font(20))
    draw.text((10, 38), "固定强度独立展示；训练时 p=0.5、最多一种，标签和有效区域不变。", fill="black", font=font())
    for k, key in enumerate(keys):
        x, y = (k % 4) * 260, 74 + (k // 4) * 290
        draw.text((x + 7, y), labels[key], fill="black", font=font())
        resample = Image.Resampling.NEAREST if key == "target" else Image.Resampling.LANCZOS
        full.paste(views[key].resize((252, 252), resample), (x + 4, y + 29))
    full.save(output / "visuals" / f"{index + 1:02d}.png")
    compact = Image.new("RGB", (1300, 308), "white")
    draw = ImageDraw.Draw(compact)
    draw.text((9, 3), f"{index + 1:02d}  {sample['id']}  |  同一 512×512 输入，独立光照变化", fill="black", font=font())
    for col, key in enumerate(("clean", "brightness/0.8", "contrast/0.8", "gamma/1.2", "target")):
        draw.text((col * 260 + 7, 27), labels[key], fill="black", font=font())
        resample = Image.Resampling.NEAREST if key == "target" else Image.Resampling.LANCZOS
        compact.paste(views[key].resize((250, 250), resample), (col * 260 + 5, 55))
    return compact


def diagnostic_manifest(path, original, base, samples, source):
    # Diagnostic-only selection. Never change the frozen formal manifests.
    meta = {key: original[key] for key in ("schema_version", "dataset", "split", "encoding", "void_values")}
    meta.update(root=str(base), root_relative_to_manifest=False, samples=samples,
                synthetic=False, complete_official_split=False,
                purpose="member4_stage2_diagnostic_only_never_train",
                source_manifest=str(source), source_manifest_sha256=sha256_file(source))
    write_json(path, meta)


def run(output, audit):
    started = time.perf_counter()
    torch.set_num_threads(4)
    previous = read_json(STAGE1 / "stage1_report.json")
    audit.check("stage1_automatic_checks_passed", previous["automatic_checks"] == "passed")
    protected = {**previous["protected_file_sha256"],
                 "transparent_seg/cache.py": sha256_file("transparent_seg/cache.py")}
    for path, digest in protected.items():
        audit.check("protected_source_intact", sha256_file(path) == digest, path)
    base_cache = subprocess.check_output(["git", "show", previous["baseline_commit"] + ":transparent_seg/cache.py"])
    audit.check("cache_matches_baseline",
                base_cache.replace(b"\r\n", b"\n") == Path("transparent_seg/cache.py").read_bytes().replace(b"\r\n", b"\n"))
    config = read_json("configs/member2_full_A_20261114.json")
    for package, expected in config["expected_versions"].items():
        audit.check("dependency_version_matches", version(package) == expected, package)
    sources = {}
    for split, (count, digest) in EXPECTED.items():
        path = MANIFESTS / f"{split}.json"
        audit.check("formal_manifest_hash_matches", sha256_file(path) == digest, split)
        meta, base = data.read_manifest(path)
        audit.check("formal_manifest_count_matches", len(meta["samples"]) == count, split)
        sources[split] = (meta, base, path)
    train = sorted(sources["train"][0]["samples"], key=lambda r: hashlib.sha256(
        ("member4-stage2/42/" + r["id"]).encode()).hexdigest())[:20]
    validation = read_json(STAGE1 / "selection.json")["samples"]
    audit.check("fixed_unique_membership", len(train) == len({s["id"] for s in train}) == 20
                and len(validation) == len({s["id"] for s in validation}) == 20)
    write_json(output / "selection.json", {
        "selection_rule": "lowest SHA256(member4-stage2/42/<id>), 20 training images; validation reuses stage1 IDs",
        "formal_manifest_sha256": {s: h for s, (_, h) in EXPECTED.items()},
        "seed": SEED, "epochs_zero_based": list(range(30)), "uncached_comparison_epochs": list(RAW_EPOCHS),
        "fixed_display_conditions": ENDPOINTS, "train": train, "validation": validation,
        "purpose": "implementation audit only; not model training or benchmark evaluation",
    })
    write_json(output / "stage1_acceptance.json", {
        "stage": 1, "accepted_by": "user", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "instruction": "阶段1通过，请继续实施阶段2的增强核验，完成后给我看对照图和检查结果，等我验收后再进入下一阶段。",
        "original_report_sha256": sha256_file(STAGE1 / "stage1_report.json"),
        "note": "User acceptance only; does not claim other members signed off. Historical report is preserved.",
    })
    manifests = {}
    for split, selected in (("train", train), ("validation", validation)):
        meta, base, source = sources[split]
        path = output / "diagnostic_manifests" / f"{split}.json"
        diagnostic_manifest(path, meta, base, selected, source)
        manifests[split] = path
        print(f"Building diagnostic cache: {split}, {len(selected)} real images", flush=True)
        build_cache(path, output / "diagnostic_cache" / split, SIZE, split == "train", workers=2)

    def dataset(split, augmented=False, cached=False, training=None):
        return data.SegmentationDataset(manifests[split], SIZE,
            split == "train" if training is None else training, augmented, SEED,
            cache_dir=output / "diagnostic_cache" / split if cached else None)

    raw, augmented = dataset("train", True), dataset("train", True, True)
    plain = dataset("train", False, True)
    clean, rgbs, panels, endpoint_rows = [], [], [], []
    for i, sample in enumerate(train):
        baseline = plain[i]
        rgb = np.asarray(plain.cache.read(i)[0], dtype=np.float32) / 255.0
        clean.append(baseline)
        rgbs.append(rgb)
        for op, value in ENDPOINTS:
            actual, condition = traced_item(augmented, i, rgb, audit, ForcedRNG(audit, op, value))
            from_raw, raw_condition = traced_item(raw, i, rgb, audit, ForcedRNG(audit, op, value))
            same_items(actual, from_raw, audit, "endpoint_cache_exact")
            audit.check("forced_endpoint_selected", condition == raw_condition == (op, value))
            audit.check("endpoint_target_unchanged", torch.equal(actual["target"], baseline["target"]))
            audit.check("endpoint_valid_unchanged", torch.equal(actual["valid"], baseline["valid"]))
            audit.check("display_effect_nonzero", not torch.equal(actual["image"], baseline["image"]))
            changed = data.perturb(rgb, op, value)
            # Check pixel arithmetic in float64, retaining the production float32
            # contrast center. A float64 mean would define a different transform.
            x = rgb.astype(np.float64)
            if op == "brightness":
                reference = x * value
            elif op == "contrast":
                center = rgb.mean((0, 1), keepdims=True).astype(np.float64)
                reference = center + value * (x - center)
            else:
                reference = x ** value
            error = float(np.abs(changed.astype(np.float64) - np.clip(reference, 0, 1)).max())
            audit.check("photometric_formula_float32_tolerance", error <= 2e-7,
                        f"{sample['id']} {op} {value}: {error}")
            endpoint_rows.append({"id": sample["id"], "operation": op, "value": value,
                "mean_absolute_rgb_change": float(np.abs(changed - rgb).mean()),
                "formula_max_abs_error": error, "target_changed_pixels": 0, "valid_changed_pixels": 0})
        panels.append(make_visuals(output, i, sample, rgb, baseline["target"]))
        print(f"Endpoint/cache/visual audit {i + 1}/20: {sample['id']}", flush=True)
    for page in range(4):
        sheet = Image.new("RGB", (1300, 1580), "white")
        ImageDraw.Draw(sheet).text((9, 8), "阶段2：真实训练图增强对照 | 展示固定强度，各列独立，非同时叠加", fill="black", font=font(20))
        for row, panel in enumerate(panels[page * 5:page * 5 + 5]):
            sheet.paste(panel, (0, 40 + row * 308))
        sheet.save(output / f"preview_{page + 1:02d}.png")
    write_json(output / "endpoint_checks.json", endpoint_rows)

    probability_rows = []
    for draw in (0.0, 0.499999, 0.5, 0.500001, 0.999999):
        item, condition = traced_item(augmented, 0, rgbs[0], audit, ForcedRNG(audit, draw=draw))
        audit.check("probability_threshold_exact", (condition is not None) == (draw < 0.5))
        probability_rows.append({"injected_uniform_draw": draw, "augmented": condition is not None})
    write_json(output / "probability_boundary_checks.json", probability_rows)

    rows, operations, signatures, values = [], Counter(), {}, []
    for epoch in range(30):
        raw.epoch = augmented.epoch = epoch
        for i, sample in enumerate(train):
            actual, condition = traced_item(augmented, i, rgbs[i], audit)
            # Disturb unrelated RNG state and verify stateless replay still matches.
            random.seed(7000 + epoch * 20 + i)
            np.random.seed(8000 + epoch * 20 + i)
            torch.manual_seed(9000 + epoch * 20 + i)
            random.random()
            np.random.random(3)
            torch.rand(3)
            again, replay_condition = traced_item(augmented, i, rgbs[i], audit)
            same_items(actual, again, audit, "same_seed_epoch_id_exact")
            audit.check("same_seed_epoch_id_condition", replay_condition == condition)
            for key in ("target", "valid"):
                audit.check("random_" + key + "_unchanged", torch.equal(actual[key], clean[i][key]))
            if epoch in RAW_EPOCHS:
                uncached, uncached_condition = traced_item(raw, i, rgbs[i], audit)
                same_items(actual, uncached, audit, "random_cache_exact")
                audit.check("random_cache_condition_exact", condition == uncached_condition)
            changed = not torch.equal(actual["image"], clean[i]["image"])
            audit.check("random_effect_matches_gate", changed == (condition is not None))
            op, value = condition if condition else ("none", None)
            operations[op] += 1
            if value is not None:
                values.append(value)
            signature = digest_tensor(actual["image"])
            signatures[epoch, i] = signature
            rows.append({"epoch_zero_based": epoch, "id": sample["id"], "operation": op,
                         "value": value, "image_changed": changed,
                         "target_changed_pixels": 0, "valid_changed_pixels": 0,
                         "repeat_exact": True, "raw_cache_checked": epoch in RAW_EPOCHS,
                         "image_sha256": signature})
        print(f"Stateless augmentation audit epoch {epoch + 1}/30 (no model training)", flush=True)
    audit.check("all_operations_and_clean_exercised", set(operations) == {"none", *OPS})
    for i in range(20):
        audit.check("augmentation_varies_across_epochs", len({signatures[e, i] for e in range(30)}) > 1)
    # Reorder sample access and construct a new dataset instance.
    replay = dataset("train", True, True)
    replay.epoch = 0
    for i in reversed(range(20)):
        audit.check("access_order_and_new_instance_independent", digest_tensor(replay[i]["image"]) == signatures[0, i])
    alternate = dataset("train", True, True)
    alternate.seed = 43
    seed_changes = sum(digest_tensor(alternate[i]["image"]) != signatures[0, i] for i in range(20))
    audit.check("seed_changes_schedule", seed_changes > 0)
    with (output / "augmentation_trace.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    val_plain, val_cached = dataset("validation"), dataset("validation", cached=True)
    val_guard, val_guard_cached = dataset("validation", True), dataset("validation", True, True)
    validation_rows = []
    for i, sample in enumerate(validation):
        baseline = val_plain[i]
        rgb = np.asarray(val_cached.cache.read(i)[0], dtype=np.float32) / 255.0
        cached, condition = traced_item(val_cached, i, rgb, audit)
        same_items(baseline, cached, audit, "validation_default_cache_exact")
        audit.check("validation_default_clean", condition is None)
        audit.check("validation_label_original_resolution",
                    tuple(baseline["target"].shape) == (1, sample["height"], sample["width"]))
        for epoch in (0, 29):
            for ds in (val_guard, val_guard_cached):
                ds.epoch = epoch
                item, condition = traced_item(ds, i, rgb, audit)
                same_items(baseline, item, audit, "validation_augmentation_disabled")
                audit.check("validation_never_calls_perturb", condition is None)
        validation_rows.append({"id": sample["id"], "default_clean": True,
                                "explicit_augmentation_flag_ignored_when_not_training": True,
                                "cache_exact": True, "target_shape": list(baseline["target"].shape)})
        print(f"Validation guard/cache audit {i + 1}/20", flush=True)
    write_json(output / "validation_checks.json", validation_rows)
    # Verify neither source code nor frozen inputs were changed while auditing.
    for path, digest in protected.items():
        audit.check("protected_source_intact_after", sha256_file(path) == digest, path)
    for split, selected in (("train", train), ("validation", validation)):
        base = sources[split][1]
        for sample in selected:
            for key in ("image", "mask"):
                audit.check("source_file_hash_intact_after",
                            sha256_file(base / sample[key]) == sample[key + "_sha256"])
        audit.check("formal_manifest_intact_after", sha256_file(sources[split][2]) == EXPECTED[split][1])
    report = {
        "stage": 2, "automatic_checks": "passed", "human_acceptance": "pending", "next_stage_started": False,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
        "audit_script_sha256": sha256_file(__file__), "protected_source_sha256": protected,
        "dependencies": {p: version(p) for p in config["expected_versions"]},
        "train_images": 20, "validation_images": 20, "epochs_observed": 30,
        "random_cases": len(rows), "endpoint_cases": len(endpoint_rows),
        "random_uncached_cache_pairs": len(train) * len(RAW_EPOCHS),
        "endpoint_uncached_cache_pairs": len(endpoint_rows),
        "operation_counts": dict(operations), "observed_augmented_fraction": 1 - operations["none"] / len(rows),
        "configured_probability": 0.5, "parameter_min_observed": min(values), "parameter_max_observed": max(values),
        "formula_reference": "float64 pixel arithmetic with production float32 contrast center",
        "formula_max_abs_tolerance": 2e-7,
        "changed_seed_different_images": seed_changes,
        "checks": dict(sorted(audit.counts.items())), "failures": [],
        "target_changed_pixels": 0, "valid_changed_pixels": 0,
        "seconds": time.perf_counter() - started,
        "limitations": [
            "20 real training and 20 validation images; not a full-data cache acceptance.",
            "Real selected masks have no void labels; existing synthetic void/cache tests are checked separately.",
            "Fixed display endpoint values illustrate individual operations, not the random training schedule.",
            "Observed frequency need not equal 0.5 exactly; probability is checked at the RNG branch boundary.",
            "No C training, performance claim, test-set or GSD model evaluation; stage3 has not started.",
        ],
    }
    write_json(output / "stage2_report.json", report)
    summary = [
        "成员4 阶段2：增强核验验收说明", "",
        "自动检查通过；你的人工验收待确认；阶段3尚未开始。",
        "阶段1的用户确认已单独记录，原始报告保留。", "",
        "先看图：preview_01.png 至 preview_04.png，共20张真实训练图。",
        "每行：原图（512缩放后）/亮度0.8/对比度0.8/gamma1.2/共同标签。",
        "visuals/01.png 至 20.png 展示每张图三类增强各自0.8和1.2两个端点。",
        "图上各增强独立作用于原图，不是依次叠加；训练仍然50%概率、最多一种。",
        "Gamma小于1偏亮、大于1偏暗；白色为透明物体标签，黑色为背景。", "",
        f"自动核验：20张×30个epoch索引={len(rows)}次真实读取，不是训练30轮。",
        f"本次实际增强{len(rows)-operations['none']}次（{report['observed_augmented_fraction']:.2%}），",
        "有限样本比例不必恰好50%；另已核验0.5概率分支的精确边界和三选一/参数范围。",
        f"各类型次数：{dict(operations)}。",
        "标签和有效区域改变像素数：均为0。",
        "同seed/epoch/id重复、打乱访问顺序、改变无关随机数状态：输入逐元素一致。",
        "增强位于resize之后、归一化之前；最多一种，输入输出范围和公式检查通过。",
        "缓存一致性：120组实际随机条件+120组固定端点，图片/标签/有效区域完全相同。",
        "20张验证图：默认无增强，即使误设augmentation=True也仍无增强；缓存前后完全一致。",
        "验证标签保持原图尺寸，训练标签仅做原有nearest缩放。", "",
        "证据：stage2_report.json；augmentation_trace.csv；endpoint_checks.json；",
        "probability_boundary_checks.json；validation_checks.json；selection.json。",
        "diagnostic_cache/只包含核验小样本；阶段3仍须建立正式全量缓存。",
        "本阶段没有加载或训练C，也不能据此宣称增强提高模型分数。", "",
        "请验收：[ ] 光照变化清楚且物体位置未移动；[ ] 理解标签保持不变；",
        "[ ] 自动检查没有失败；[ ] 确认阶段2通过后进入阶段3。",
    ]
    (output / "验收说明.txt").write_text("\n".join(summary) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("automatic_checks", "random_cases", "operation_counts",
          "observed_augmented_fraction", "target_changed_pixels", "valid_changed_pixels", "seconds")},
          ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new output directory; never overwrite an audit package")
    output.mkdir(parents=True)
    (output / "visuals").mkdir()
    audit = Audit()
    try:
        run(output, audit)
    except Exception as exc:
        write_json(output / "stage2_failure.json", {"stage": 2, "automatic_checks": "failed",
            "error": f"{type(exc).__name__}: {exc}", "completed_checks": dict(audit.counts),
            "next_stage_started": False})
        raise


if __name__ == "__main__":
    main()
