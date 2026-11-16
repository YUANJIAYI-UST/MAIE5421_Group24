"""Finish authorized stage4 reporting once the sequential experiments succeed."""
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("runs/member4_stage4_20261115")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")
    temp.replace(path)


def main():
    status_path = ROOT / "reporting_state.json"
    if status_path.exists():
        raise ValueError("Reporting worker already initialized; inspect before restarting")
    state = {"status": "waiting_for_experiments", "pid": os.getpid(), "stage5_started": False}
    save(status_path, state)
    awake = False
    try:
        if os.name == "nt":
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        while True:
            queue = read(ROOT / "state.json")
            if queue["status"] == "experiments_complete":
                break
            if queue["status"] == "stopped_requires_inspection":
                raise RuntimeError("Experiment queue stopped; reporting must not claim completion")
            time.sleep(30)
        for filename, digest in read(ROOT / "reporting_source_sha256.json").items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise ValueError("Reporting source changed: " + filename)
        commands = [
            [sys.executable, "-X", "utf8", "-B", "scripts/summarize_member4_stage4.py"],
            ["D:/app/miniconda/python.exe", "-X", "utf8", "-B", "scripts/plot_member4_results.py",
             "--root", str(ROOT / "results")],
            [sys.executable, "-X", "utf8", "-B", "scripts/member4_result_gallery.py"],
        ]
        for i, command in enumerate(commands):
            state.update(status="assembling", step=i + 1, command=command)
            save(status_path, state)
            subprocess.run(command, check=True)
        result = ROOT / "results"
        report = read(result / "summary.json")
        rows = [r for r in report["comparison"] if r["subset"] == "all"]
        labels = {"foreground_iou": "前景IoU", "region_f1": "区域F1", "boundary_iou": "Boundary IoU"}
        lines = [
            "成员4 阶段4：正式C训练及全量A/C验收说明", "",
            "自动核验通过；用户人工验收待确认；阶段5尚未开始。",
            "本机RTX5090完成冻结配置：seed42、512、batch8、BF16、30轮、18600次更新。",
            f"C最佳权重来自第{report['C_training']['selected_epoch']}轮，仅按全部995张验证图的前景IoU选择。",
            "最佳权重重载后完整验证的区域统计与选模记录完全一致。",
            "A/C各13个条件、每条件4428张，共115128次图像-条件评估全部完成。",
            "逐条件ID无遗漏/重复；IoU/F1按累计像素计算，Boundary IoU逐图平均。", "",
            "全体测试集结果：clean及12条件平均为百分数，drop为百分点。",
        ]
        for row in rows:
            lines += [
                labels[row["metric"]] + "：",
                f"  clean：A {row['A_clean_percent']:.4f}% / C {row['C_clean_percent']:.4f}% / C-A {row['C_minus_A_clean_pp']:+.4f}个百分点",
                f"  12条件平均：A {row['A_perturbed_mean_percent']:.4f}% / C {row['C_perturbed_mean_percent']:.4f}% / C-A {row['C_minus_A_perturbed_mean_pp']:+.4f}个百分点",
                f"  drop=clean-12条件平均：A {row['A_drop_pp']:.4f} / C {row['C_drop_pp']:.4f}个百分点",
            ]
        lines += [
            "", "如何看结果",
            "clean衡量正常图片，12条件平均衡量这组固定光照扰动下的表现，drop衡量相对自身clean的下降。",
            "比较平均分和下降时要同时看clean起点，不能只看某个最好看的指标。",
            "这是单seed A/C实验，不能据此声称在GSD或所有新场景上更好；GSD尚未开始。",
            "A/C的数据、初始化、训练预算和核心计算库版本相同；GPU型号、Python小版本及部分辅助依赖不同。",
            "完整环境差异见environment_comparison.json；不能将两台机器的环境称为完全相同。",
            "无论C是否提高，完整保留全部26组结果，不通过测试分数改参数或重选模型。", "",
            "交付入口",
            "A_C_comparison.csv：all/easy/hard的正常、扰动平均及drop对比。",
            "condition_results.csv：26条件×3分组的完整统计。",
            "robustness_summary.csv：均分与drop，列名明确fraction/pp。",
            "robustness_all.png、robustness_easy.png、robustness_hard.png：三类扰动对比曲线。",
            "../figures/training_curves.png：完整训练与验证曲线。",
            "gallery/：固定12张及按预先规则选择的诊断案例；索引说明每张图的选择依据。",
            "failure_selection.csv：低分/相对退步案例的具体规则和分数。",
            "summary.json：机器验收、权重位置与哈希；原始逐图记录留在各正式评估目录。", "",
            "模型位置",
            "C：runs/member4_full_C_seed42_20261115/best.pt（最佳）及last.pt（第30轮续训状态）。",
            "A：runs/member2_full_A_20261114/best.pt，原始权重和结果保留。",
            "加载C预测：",
            ".\\.venv\\Scripts\\python.exe -X utf8 -B -m transparent_seg.predict --checkpoint runs/member4_full_C_seed42_20261115/best.pt --image 图片路径 --output 新预测目录",
            "", "请验收",
            "[ ] 理解C的增强只改变光照，标签和几何位置保持不变。",
            "[ ] 查看完整预算、选模及26组完整覆盖检查。",
            "[ ] 查看clean、扰动均分和drop，而非只挑一项指标。",
            "[ ] 查看固定案例与明确标注选择规则的失败/相对薄弱案例。",
            "[ ] 确认阶段4通过后，再进入阶段5的GSD准备与重叠审计。",
        ]
        (result / "验收说明.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        shutil.copy2(ROOT / "environment_comparison.json", result / "environment_comparison.json")
        small = Path("reports/member4_stage4_20261115")
        small.mkdir(exist_ok=False)
        for filename in ("summary.json", "condition_results.csv", "robustness_summary.csv", "A_C_comparison.csv",
                         "failure_selection.csv", "验收说明.txt", "environment_comparison.json", "robustness_all.png", "robustness_easy.png", "robustness_hard.png"):
            shutil.copy2(result / filename, small / filename)
        hashes = {p.relative_to(result).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in result.rglob("*") if p.is_file()}
        save(result / "files_sha256.json", hashes)
        state.update(status="complete_pending_user_acceptance", ended_at=datetime.now(timezone.utc).isoformat(),
                     results=str(result), human_acceptance="pending", stage5_started=False)
        save(status_path, state)
        print("Stage4 results, figures and case gallery are ready for inspection; stage5 has not started.", flush=True)
    except BaseException as exc:
        state.update(status="failed_requires_inspection", error=f"{type(exc).__name__}: {exc}")
        save(status_path, state)
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
