"""Summarize measured stage3 evidence and explicitly estimated stage4 runtime."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.optimize import nnls

from transparent_seg.evaluate_resumable import code_identity, CONDITIONS
from transparent_seg.prepare import sha256_file, write_json


ROOT = Path("runs/member4_stage3_20261115")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    destination = ROOT / "acceptance"
    destination.mkdir(exist_ok=False)
    training = read(ROOT / "training_preflight/preflight_report.json")
    evaluation = read(ROOT / "evaluation_preflight/evaluation_preflight_report.json")
    clean = read(ROOT / "evaluation_preflight/clean40/result.json")
    cache = read(ROOT / "cache_build.json")
    assert training["status"] == evaluation["status"] == "passed" and cache["complete"]
    assert evaluation["code_identity"] == code_identity()
    assert clean["complete"] and clean["image_count"] == 40
    assert clean["scores"]["clean"] == evaluation["continuous"]["scores"]["clean"]
    for name in ("member4_training_preflight", "member4_evaluation_preflight"):
        record = training if "training" in name else evaluation
        assert sha256_file(Path("scripts") / (name + ".py")) == record["script_sha256"]
    suite = ET.parse(ROOT / "pytest_02.xml").getroot().find("testsuite")
    assert int(suite.attrib["tests"]) == 105 and int(suite.attrib["failures"]) == int(suite.attrib["errors"]) == 0
    config_a = read("configs/member2_full_A_20261114.json")
    config_c = read("configs/member4_full_C_20261115.json")
    differences = [{"setting": k, "A": config_a.get(k), "C": config_c.get(k),
                    "changed": config_a.get(k) != config_c.get(k)} for k in sorted(set(config_a) | set(config_c))]
    assert {r["setting"] for r in differences if r["changed"]} == {
        "ablation", "augmentation", "frozen_protocol", "frozen_protocol_sha256"}
    with (destination / "A_vs_C_config.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(differences[0]))
        writer.writeheader()
        for row in differences:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, dict) else v
                             for k, v in row.items()})
    rows = []
    for path in sorted((ROOT / "evaluation_preflight/continuous40/chunks").glob("*.json")):
        rows.extend(read(path)["body"]["rows"])
    test = read("manifests/member1_final_20261029/v1/test.json")["samples"]
    assert len(rows) == len({r["id"] for r in rows}) == 40 and len(test) == 4428
    X = np.array([[1, r["pixels"] / 1e6] for r in rows])
    y = np.array([r["seconds"] for r in rows])
    beta, residual = nnls(X, y)
    counts = Counter(r["difficulty"] for r in test)
    groups = {g: float(np.mean([r["seconds"] for r in rows if r["difficulty"] == g])) for g in counts}
    estimates = {
        "same_mean_per_image": 2 * 4428 * evaluation["continuous"]["processing_seconds"] / 40,
        "difficulty_weighted": 2 * sum(counts[g] * groups[g] for g in counts),
        "pixel_count_fit": 2 * (4428 * beta[0] + sum(s["height"] * s["width"] / 1e6 for s in test) * beta[1]),
    }
    history = read(ROOT / "training_preflight/pilot_C_20_steps/config.json")  # Explicitly distinguish pilot config.
    assert history["purpose"] == "pilot" and history["max_steps_per_epoch"] == 20
    record = json.loads((ROOT / "training_preflight/pilot_C_20_steps/history.jsonl").read_text().splitlines()[0])
    checkpoint_seconds = max(0, record["seconds"] - training["train_seconds"] - training["validation_seconds"])
    setup_seconds = max(0, training["pilot_wall_seconds"] - record["seconds"])
    train_estimate = (training["train_seconds"] / 20 * 18600
                      + evaluation["isolated_full_validation_seconds"] * 30
                      + checkpoint_seconds * 30 + setup_seconds)
    eval_estimate = max(estimates.values()) + evaluation["continuous"]["startup_seconds"] * 2
    selected_validation_estimate = clean["processing_seconds"] / 40 * 995 + clean["startup_seconds"]
    total = train_estimate + eval_estimate + selected_validation_estimate
    generated = datetime.now(timezone(timedelta(hours=8)))
    budget = {"generated_at": generated.isoformat(), "phase4_started": False,
        "measured": {"training_20_steps_seconds": training["train_seconds"],
            "steady_gpu_step_seconds_excluding_data": training["steady_microbatch_seconds"],
            "full_995_validation_seconds_isolated": evaluation["isolated_full_validation_seconds"],
            "40_images_13_conditions_processing_seconds": evaluation["continuous"]["processing_seconds"],
            "40_images_13_conditions_total_seconds": evaluation["continuous"]["elapsed_seconds"],
            "40_images_clean_total_seconds": clean["elapsed_seconds"],
            "peak_cuda_allocated_mb": training["peak_cuda_memory_mb"],
            "peak_cuda_reserved_mb": training["peak_cuda_reserved_mb"],
            "cache_build_seconds_already_completed": sum(r["seconds"] for r in cache["rows"])},
        "estimated_seconds": {"C_training_including_30_full_validations": train_estimate,
            "C_best_full_clean_validation": selected_validation_estimate,
            "A_and_C_4428_images_each_13_conditions": eval_estimate, "phase4_total": total,
            "planning_allowance_25_percent": total * 1.25, "planning_allowance_50_percent": total * 1.5},
        "evaluation_estimation_crosschecks_seconds": estimates,
        "pixel_fit": {"seconds_intercept": float(beta[0]), "seconds_per_megapixel": float(beta[1]),
                      "rmse_seconds": float(residual / np.sqrt(len(rows)))},
        "test_counts_from_manifest_only": dict(counts),
        "method": "Training uses measured wall time including loading, not GPU-only step time. A/C estimate takes the largest of sample-mean, difficulty-weighted and nonnegative pixel-fit estimates. Adds one full best-model validation.",
        "limits": ["25%-50% is an engineering planning allowance, not a statistical confidence interval.",
            "Timing pilot is 20 steps and validation sample is 40 images; full execution has not occurred.",
            "Laptop temperature, power, competing processes, I/O and C's predicted mask density can change throughput.",
            "Pilot uses a separate short LR schedule; formal C uses the frozen 18600-step schedule.",
            "Excludes GSD audit/evaluation, B/D, report authoring and any user acceptance delay.",
            "Full cache construction is already completed and is not charged again in phase4."]}
    write_json(destination / "runtime_budget.json", budget)
    source_paths = list(Path("transparent_seg").glob("*.py")) + [
        Path("configs") / name for name in (
            "member2_full_A_20261114.json", "member2_full_A_protocol_20261114.json",
            "member4_full_C_20261115.json", "member4_full_C_protocol_20261115.json")]
    source_paths += [Path("scripts") / name for name in (
        "member4_training_preflight.py", "member4_evaluation_preflight.py",
        "summarize_member4_stage3.py", "run_member4_C.ps1")]
    source_hashes = {}
    for source in source_paths:
        target = destination / "source_snapshot" / source
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        source_hashes[source.as_posix()] = sha256_file(source)
    assert source_hashes["configs/member2_full_A_20261114.json"] == "5205af206df59eb2a436d46e8b24883be24be4296adf067f480fbd3f0eb9b9bc"
    assert source_hashes["configs/member2_full_A_protocol_20261114.json"] == "baa62eb2c3ba5a7358caf0020a3740460ac5d885868acd113391d5bdb08e9eaf"
    assert sha256_file("runs/member2_full_A_20261114/best.pt") == "3a785feb578ecbd2199684c53130737c872cd6c2f5803f221297f1e9dac0b25a"
    write_json(destination / "source_sha256.json", source_hashes)
    report = {"stage": 3, "automatic_checks": "passed", "human_acceptance": "pending", "phase4_started": False,
        "core_and_regression_tests": 105, "test_failures": 0, "formal_data_counts": [4959, 995, 4428],
        "initialization_matches_A": True, "full_caches_verified": True, "pilot_steps": 20,
        "pilot_gradients_and_bf16_passed": True, "pilot_reloaded_995_validation_exact": True,
        "legacy_equivalence": {"images": 8, "conditions": 13, "every_pixel_and_metric_exact": True},
        "hard_crash_recovery": {"images": 40, "conditions": 13, "exit_code": 86,
            "reused_images": 25, "remaining_images": 15, "all_per_image_csv_bytes_exact": True},
        "runtime_budget": "runtime_budget.json", "source_snapshot": "source_snapshot/",
        "code_identity": code_identity(), "formal_C_config_sha256": sha256_file("configs/member4_full_C_20261115.json"),
        "A_original_artifacts_preserved": True, "formal_model_scores_produced": False}
    write_json(destination / "stage3_report.json", report)
    hours = {k: v / 3600 for k, v in budget["estimated_seconds"].items()}
    lines = [
        "成员4 阶段3：验收说明", "",
        "自动核验通过；用户验收待确认；没有开始阶段4正式训练或全量测试。",
        "C配置与协议已独立冻结；公共设置与A一致，只增加指定光照增强。",
        "A原始配置、协议和已发布best权重哈希未变。", "",
        "检查结果",
        "105项测试通过；新旧评估8张真实验证图×13条件逐像素和全部指标完全一致。",
        "全量缓存4959 train + 995 validation建好并校验；20+20真实输入对比完全一致。",
        "C独立20步pilot通过：batch8、512、BF16、有限且完整梯度、共享初始化一致。",
        "重载pilot最佳权重后，完整995张验证结果与训练记录完全一致。",
        "40张×13条件：进程完成25张后强制退出，恢复余下15张，结果与连续运行一致。",
        "中断演练退出码86是预先设定的诊断行为，不是未处理的程序失败。",
        "损坏块、身份变化、并发写入和旧完成标志的保护也已测试。", "",
        "实测",
        f"20步训练（含数据读取）：{training['train_seconds']:.2f}秒。",
        f"稳定GPU计算：{training['steady_microbatch_seconds']:.3f}秒/步，估算总训练时另计数据读取。",
        f"完整995张验证：{evaluation['isolated_full_validation_seconds']:.2f}秒。",
        f"40张×13条件：{evaluation['continuous']['elapsed_seconds']:.2f}秒。",
        f"40张clean：{clean['elapsed_seconds']:.2f}秒。",
        f"峰值显存已分配{training['peak_cuda_memory_mb']/1024:.2f} GiB，预留{training['peak_cuda_reserved_mb']/1024:.2f} GiB。", "",
        "阶段4估算（尚未执行）",
        f"C正式训练含30次完整验证：约{hours['C_training_including_30_full_validations']:.2f}小时。",
        f"C最佳权重完整clean验证：约{selected_validation_estimate/60:.1f}分钟。",
        f"A/C各4428张×13条件：约{hours['A_and_C_4428_images_each_13_conditions']:.2f}小时。",
        f"总计约{hours['phase4_total']:.2f}小时；另留25%-50%余量后约",
        f"{hours['planning_allowance_25_percent']:.1f}-{hours['planning_allowance_50_percent']:.1f}小时。",
        "建议预留约11-13小时连续运行；这不是完成时间保证，不包含GSD及B/D。",
        "实际测试全量不缩减；不会把未完成结果标成完成。", "",
        "请查看A_vs_C_config.csv、stage3_report.json、runtime_budget.json。",
        "详细复现入口：docs/22_MEMBER4_STAGE3_OPERATIONS.txt。",
        "所有pilot和40张诊断成绩都不是正式C或全量测试成绩。", "",
        "人工验收：[ ] 配置公平；[ ] 自动检查通过；[ ] 明白pilot不进入正式实验；",
        "[ ] 接受实际预算；[ ] 确认启动整个阶段4队列。",
        "确认前不执行正式C训练与全量A/C；确认后队列内部无需逐条确认命令。",
    ]
    (destination / "验收说明.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"automatic_checks": "passed", "estimated_hours": hours}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
