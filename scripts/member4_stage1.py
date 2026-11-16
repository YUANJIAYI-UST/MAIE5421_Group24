"""Freeze the member-4 starting point and recheck 20 fixed validation images.

This produces a review package, not a C training configuration or a new benchmark.
Run from the repository root; output must be a new directory.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from importlib.metadata import version

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from transparent_seg.data import decode_mask, read_manifest
from transparent_seg.evaluate import load_checkpoint, predict
from transparent_seg.metrics import SegmentationMetrics
from transparent_seg.model import TransparentSegformer
from transparent_seg.prepare import sha256_file, write_json
from transparent_seg.protocol import validate_frozen_protocol
from transparent_seg.train import seed_everything, shared_parameter_hash


BASE_COMMIT = "9cfc5436bdf5c4a7b0da899c1a57450ccaed96d7"
A_CONFIG = Path("configs/member2_full_A_20261114.json")
A_RUN = Path("runs/member2_full_A_20261114")
MANIFESTS = Path("manifests/member1_final_20261029/v1")
EXPECTED = {
    "train": (4959, "7a3094e0c1aeacf0edd7335f196af79eacd4b551cd9f78cce2cf7c903e910ca2"),
    "validation": (995, "861c24027bfddcfba031e253a6518343010e31d296fb66c010f8d88ee9a90497"),
    "test": (4428, "ecf3ad309423aa0247183cfc1e315921a7666c47d10700f381199a7dcfb41012"),
}
CHECKPOINT_SHA = "3a785feb578ecbd2199684c53130737c872cd6c2f5803f221297f1e9dac0b25a"
INITIAL_SHA = "fd50559254d2d5fe9b16cbaa6f7fd32452faa4a7f3b9a3ec42f1a6fb6eedc835"
# Absolute fractions, fixed before re-evaluation; accommodates different GPU hardware.
# These are software reproducibility tolerances, not a required model performance.
TOLERANCES = {"foreground_iou": 0.0001, "region_f1": 0.0001, "boundary_iou": 0.0005}
COUNTS = ("tp", "fp", "fn", "tn")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8").strip()


def fixed_selection(samples):
    selected = []
    for difficulty in ("easy", "hard"):
        group = [r for r in samples if r["difficulty"] == difficulty]
        group.sort(key=lambda r: hashlib.sha256(
            ("member4-stage1/42/" + r["id"]).encode()).hexdigest())
        require(len(group) >= 10, "Insufficient samples in validation group")
        selected.extend(group[:10])
    return selected


def merge_row(meter, row):
    for key in COUNTS:
        setattr(meter, key, getattr(meter, key) + int(row[key]))
    meter.images += int(row["images"])
    meter.skipped += int(row["skipped_all_void"])
    if int(row["boundary_images"]):
        meter.boundaries.append(float(row["boundary_iou"]))


def panel(image, target, prediction, sample, current, reference):
    cell, top = 360, 72
    canvas = Image.new("RGB", (4 * cell, cell + top), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    draw.text((12, 5), f"{sample['id']}  |  {sample['difficulty']}", fill="black", font=font)
    draw.text((12, 30), f"IoU local / reference: {current['foreground_iou']:.6f} / "
              f"{float(reference['foreground_iou']):.6f}", fill="black", font=font)
    # Visualization only. Metrics are calculated on the original-resolution arrays.
    original = image.copy()
    original.thumbnail((cell - 8, cell - 8))
    target_view = Image.fromarray(target.astype(np.uint8) * 255)
    prediction_view = Image.fromarray(prediction.astype(np.uint8) * 255)
    target_view = target_view.resize(original.size, Image.Resampling.NEAREST).convert("RGB")
    prediction_view = prediction_view.resize(original.size, Image.Resampling.NEAREST).convert("RGB")
    rgb = np.array(original, dtype=np.float32)
    small_pred = np.array(prediction_view)[..., 0] > 0
    rgb[small_pred] = 0.5 * rgb[small_pred] + 0.5 * np.array([0, 220, 160])
    views = [original, target_view, prediction_view, Image.fromarray(rgb.astype(np.uint8))]
    for col, (label, view) in enumerate(zip(
            ("原图", "人工标准答案", "A 模型预测", "预测叠图"), views)):
        draw.text((col * cell + 12, 53), label, fill="black", font=font)
        canvas.paste(view, (col * cell + (cell - view.width) // 2,
                           top + (cell - view.height) // 2))
    return canvas


def main(output):
    output = Path(output)
    require(not output.exists(), "Use a new output directory to preserve evidence")
    require(torch.cuda.is_available(), "CUDA is required for the local baseline recheck")
    # Verify the inference implementation still matches the reviewed starting commit.
    protected = [A_CONFIG, Path("configs/member2_full_A_protocol_20261114.json")]
    protected += [Path("transparent_seg") / (n + ".py") for n in
                  ("model", "data", "train", "losses", "metrics", "evaluate", "protocol")]
    before = {str(p): sha256_file(p) for p in protected}
    for p in protected:
        original = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{p.as_posix()}"])
        require(original.replace(b"\r\n", b"\n") == p.read_bytes().replace(b"\r\n", b"\n"),
                f"Protected starting-point file differs: {p}")
    config = read_json(A_CONFIG)
    require(config == read_json(A_RUN / "config.json"), "Published and local A configs differ")
    require(config["ablation"] == "A" and not config["boundary"] and not config["augmentation"],
            "Unexpected baseline configuration")
    dependencies = {p: version(p) for p in config["expected_versions"]}
    require(dependencies == config["expected_versions"], "Core dependency versions differ")
    require(sha256_file(A_RUN / "best.pt") == CHECKPOINT_SHA, "A checkpoint hash mismatch")
    metadata = {}
    for split, (count, digest) in EXPECTED.items():
        path = MANIFESTS / f"{split}.json"
        require(sha256_file(path) == digest, f"{split} manifest hash mismatch")
        meta = read_json(path)
        require(len(meta["samples"]) == count and meta["split"] == split, "Split/count mismatch")
        metadata[split] = meta
    validate_frozen_protocol(config, metadata["train"], metadata["validation"])
    require((4959 + config["batch_size"] - 1) // config["batch_size"] * config["epochs"] == 18600,
            "Training budget differs")
    validation, data_root = read_manifest(MANIFESTS / "validation.json")
    selected = fixed_selection(validation["samples"])
    provenance = read_json(A_RUN / "validation_evaluation/provenance.json")
    require(provenance["checkpoint_sha256"] == CHECKPOINT_SHA, "Reference checkpoint differs")
    with (A_RUN / "validation_evaluation/clean/per_image.csv").open(
            encoding="utf-8", newline="") as stream:
        reference_rows = list(csv.DictReader(stream))
    reference = {r["id"]: r for r in reference_rows}
    require(len(reference) == len(reference_rows) == 995, "Invalid reference CSV membership")
    require(set(reference) == {r["id"] for r in validation["samples"]},
            "Reference does not cover exactly the frozen validation set")
    output.mkdir(parents=True)
    (output / "visuals").mkdir()
    # Save sample membership and thresholds before any prediction is performed.
    write_json(output / "selection.json", {
        "purpose": "software recheck, not full validation performance",
        "selection": "lowest SHA256(member4-stage1/42/<id>), 10 easy + 10 hard",
        "manifest_sha256": EXPECTED["validation"][1], "tolerances": TOLERANCES,
        "samples": selected,
    })
    intended_c = {k: v for k, v in config.items()
                  if k not in ("frozen_protocol", "frozen_protocol_sha256")}
    intended_c.update(ablation="C", augmentation=True)
    write_json(output / "C_design_only.json", {
        "status": "design_only_not_executable; stage_3_protocol_pending",
        "source_A_config_sha256": sha256_file(A_CONFIG), "proposed_config": intended_c,
        "training_start": "fresh pinned ImageNet encoder and shared fresh decoder, never resume A",
        "human_acceptance": "pending",
    })
    differences = []
    for key in sorted(intended_c):
        differences.append({"setting": key, "A": config[key], "planned_C": intended_c[key],
                            "changed": config[key] != intended_c[key]})
    require({r["setting"] for r in differences if r["changed"]} == {"ablation", "augmentation"},
            "Unplanned experiment difference")
    with (output / "A_vs_C_config.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["setting", "A", "planned_C", "changed"])
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False, sort_keys=True)
                          if isinstance(v, (dict, list)) else v for k, v in r.items()}
                         for r in differences)
    torch.set_num_threads(4)
    seed_everything(config["seed"])
    torch.use_deterministic_algorithms(True)
    initial = TransparentSegformer(False, True, config["encoder_revision"], ".cache/huggingface")
    initialization = shared_parameter_hash(initial)
    require(initialization == INITIAL_SHA, "Fresh initialization differs from A")
    del initial
    model, checkpoint = load_checkpoint(A_RUN / "best.pt", "cuda")
    require(checkpoint["config"] == config and checkpoint["epoch"] == 29,
            "Unexpected A checkpoint metadata")
    local_meters = {k: SegmentationMetrics() for k in ("all", "easy", "hard")}
    reference_meters = {k: SegmentationMetrics() for k in local_meters}
    results, panels, failures = [], [], []
    for index, sample in enumerate(selected):
        for key in ("image", "mask"):
            require(sha256_file(data_root / sample[key]) == sample[key + "_sha256"],
                    f"Selected source changed: {sample['id']} {key}")
        with Image.open(data_root / sample["image"]) as im:
            image = im.convert("RGB")
        target, valid = decode_mask(data_root / sample["mask"], validation["encoding"],
                                    validation.get("void_values", []))
        prediction = predict(model, image, config["size"], "cuda")
        meter = SegmentationMetrics()
        meter.update(prediction, target, valid)
        current, ref = meter.compute(), reference[sample["id"]]
        require(prediction.shape == target.shape == (image.height, image.width),
                "Prediction/label geometry mismatch")
        delta = {key: abs(current[key] - float(ref[key])) for key in TOLERANCES}
        failed = [key for key in TOLERANCES
                  if not np.isfinite(current[key]) or delta[key] > TOLERANCES[key]]
        if failed:
            failures.append({"id": sample["id"], "metrics": failed, "absolute_delta": delta})
        for group in ("all", sample["difficulty"]):
            merge_row(local_meters[group], current)
            merge_row(reference_meters[group], ref)
        view = panel(image, target, prediction, sample, current, ref)
        view.save(output / "visuals" / f"{index + 1:02d}.png")
        panels.append(view)
        results.append({"id": sample["id"], "difficulty": sample["difficulty"],
                        "scores": current, "reference_scores": {
                            k: int(ref[k]) if k in COUNTS else float(ref[k])
                            for k in (*COUNTS, *TOLERANCES)},
                        "absolute_metric_delta": delta,
                        "count_deltas": {k: current[k] - int(ref[k]) for k in COUNTS},
                        "passed": not failed})
        print(f"Checked {index + 1}/20: {sample['id']} {'PASS' if not failed else 'REVIEW'}",
              flush=True)
    for index in range(4):
        sheet = Image.new("RGB", (1440, 432 * 5), "#e5e7eb")
        for row, view in enumerate(panels[index * 5:index * 5 + 5]):
            sheet.paste(view, (0, row * 432))
        sheet.save(output / f"preview_{index + 1:02d}.png")
    aggregate = {}
    for group in local_meters:
        local, ref = local_meters[group].compute(), reference_meters[group].compute()
        delta = {k: abs(local[k] - ref[k]) for k in TOLERANCES}
        aggregate[group] = {"local": local, "reference_same_selected_ids": ref,
                            "absolute_metric_delta": delta}
        if any(delta[k] > TOLERANCES[k] for k in TOLERANCES):
            failures.append({"group": group, "absolute_delta": delta})
    require(before == {str(p): sha256_file(p) for p in protected}, "Protected files changed")
    baseline_environment = read_json(A_RUN / "environment.json")
    report = {
        "stage": 1, "automatic_checks": "passed" if not failures else "failed",
        "human_acceptance": "pending", "next_stage_started": False,
        "baseline_commit": BASE_COMMIT, "current_commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "worktree_status": git("status", "--short"),
        "audit_script_sha256": sha256_file(__file__), "protected_file_sha256": before,
        "input_provenance_sha256": sha256_file(A_RUN / "validation_evaluation/provenance.json"),
        "reference_csv_sha256": sha256_file(A_RUN / "validation_evaluation/clean/per_image.csv"),
        "checkpoint_sha256": CHECKPOINT_SHA, "fresh_initial_shared_sha256": initialization,
        "data_counts": {s: n for s, (n, _) in EXPECTED.items()}, "selection_count": len(selected),
        "tolerances": TOLERANCES, "scores": aggregate, "per_image": results, "failures": failures,
        "all_selected_region_counts_exact": all(
            all(v == 0 for v in r["count_deltas"].values()) for r in results),
        "local_environment": {"python": platform.python_version(),
                              "gpu": torch.cuda.get_device_name(0), "versions": dependencies},
        "reference_environment": {"python": baseline_environment["python"],
                                  "gpu": baseline_environment["gpu"]},
        "limitations": [
            "Only 20 validation samples; deliberately balanced easy/hard, not all 995 images.",
            "Cross-GPU validation checks use predeclared numerical tolerances.",
            "No C training, no augmentation tuning, no test/GSD model evaluation.",
            "Original member review metadata is preserved; no human sign-off is fabricated.",
        ],
    }
    write_json(output / "stage1_report.json", report)
    with (output / "A_recheck.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ["id", "difficulty", "passed"] + [
            f"{prefix}_{metric}" for metric in TOLERANCES for prefix in ("local", "reference", "delta")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {k: row[k] for k in ("id", "difficulty", "passed")}
            for metric in TOLERANCES:
                flat.update({f"local_{metric}": row["scores"][metric],
                             f"reference_{metric}": row["reference_scores"][metric],
                             f"delta_{metric}": row["absolute_metric_delta"][metric]})
            writer.writerow(flat)
    status = "通过" if not failures else "未通过：先查看 stage1_report.json 的 failures，不能进入下一阶段"
    summary = [
        "成员 4 阶段 1：验收说明", "", f"自动检查：{status}；你的人工验收：待确认。",
        "本阶段只固定比较规则并复查 A，没有训练 C，也没有开始阶段 2。",
        "", "请依次查看：",
        "1. preview_01.png 至 preview_04.png：20 张真实验证图；visuals/ 中可单独放大。",
        "   每行从左到右为原图、人工标准答案、A 模型预测、预测叠图。",
        "   预测不是标准答案，不要求每个轮廓都完全正确；请注意输入、标签是否对应。",
        "2. A_vs_C_config.csv：唯一的实验设置变化是 augmentation=False -> True；",
        "   ablation=A -> C 只是名称变化。正式 C 协议要在阶段 3 单独建立。",
        "3. A_recheck.csv：与成员 2 对同一批图片的原始逐图结果比较，而非与完整验证集分数比较。",
        "4. stage1_report.json：数据、权重、环境、初始化哈希及逐图检查详情。",
        "", "固定规则：",
        "train/validation/test=4959/995/4428；512、batch8、seed42、30epochs、18600次更新。",
        "AdamW lr=6e-5、weight_decay=0.01、BF16、完整验证选最佳模型。",
        "C 从相同 ImageNet 初始权重和全新分割头开始，不从 A 成品权重继续训练。",
        "增强概率0.5，亮度/对比度/gamma三选一，参数[0.8,1.2]；不增加其他增强。",
        "", "数值复查规则（运行之前固定）：",
        "逐图及分组 IoU/F1 绝对差 <=0.0001；Boundary IoU 绝对差 <=0.0005。",
        "即最多0.01/0.05个百分点。这只是不同 GPU 的复现容差，不是性能提升目标。",
        "本批10 easy+10 hard，仅用于验收；不能当作995张验证集或4428张测试集的正式成绩。",
        "", "人工验收清单（请实际查看后确认）：",
        "[ ] 知道原图、标签、预测三者的区别，输入与标签没有明显错配。",
        "[ ] 同意 C 与 A 保持相同预算，只增加指定增强。",
        "[ ] 已查看数值复查结论；没有未处理的失败项。",
        "[ ] 同意进入阶段 2：真实图片增强核验。",
    ]
    (output / "验收说明.txt").write_text("\n".join(summary) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["automatic_checks"], "images": len(selected),
                      "region_counts_exact": report["all_selected_region_counts_exact"],
                      "aggregate_deltas": {k: v["absolute_metric_delta"] for k, v in aggregate.items()},
                      "output": str(output)}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
