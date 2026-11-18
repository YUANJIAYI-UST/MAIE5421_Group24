"""Independently audit all 26 full-test outputs and prepare report-ready tables."""
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transparent_seg.data import read_manifest
from transparent_seg.evaluate_resumable import CONDITIONS, code_identity
from transparent_seg.prepare import sha256_file, write_json

ROOT = Path("runs/member4_stage4_20261115")
TRAIN = Path("runs/member4_full_C_seed42_20261115")
MODELS = {"A": (Path("runs/member2_full_A_20261114"), Path("runs/member4_A_full_test_20261115")),
          "C": (TRAIN, Path("runs/member4_C_full_test_20261115"))}
METRICS = ("foreground_iou", "region_f1", "boundary_iou")
COUNTS = ("tp", "fp", "fn", "tn")
GROUPS = ("all", "easy", "hard")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_condition(folder, name, samples):
    metrics = read(folder / name / "metrics.json")
    with (folder / name / "per_image.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 4428 and [r["id"] for r in rows] == [s["id"] for s in samples],
            f"Incomplete/duplicate/wrong membership: {folder}/{name}")
    for row, sample in zip(rows, samples):
        counts = {k: int(row[k]) for k in COUNTS}
        require(all(v >= 0 for v in counts.values())
                and sum(counts.values()) == sample["height"] * sample["width"]
                and counts["tp"] + counts["fn"] == sample["foreground_pixels"]
                and int(row["images"]) == int(row["boundary_images"]) == 1
                and int(row["skipped_all_void"]) == 0,
                "Invalid per-image sufficient statistics: " + row["id"])
        for key in METRICS:
            require(math.isfinite(float(row[key])) and 0 <= float(row[key]) <= 1, "Invalid metric")
    for group in GROUPS:
        chosen = [r for r, s in zip(rows, samples) if group == "all" or s["difficulty"] == group]
        counts = {k: sum(int(r[k]) for r in chosen) for k in COUNTS}
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        reference = {"foreground_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "region_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "boundary_iou": math.fsum(float(r["boundary_iou"]) for r in chosen) / len(chosen)}
        actual = metrics["scores"][group]
        require(actual["images"] == actual["boundary_images"] == len(chosen), "Group membership mismatch")
        require(all(actual[k] == counts[k] for k in COUNTS), "Pooled pixel statistics mismatch")
        require(all(abs(actual[k] - reference[k]) <= 1e-12 for k in METRICS), "Aggregation rule mismatch")
    return metrics, {r["id"]: r for r in rows}


def main():
    output = ROOT / "results"
    require(not output.exists(), "Use a new report directory; preserve prior results")
    state = read(ROOT / "state.json")
    require(state["status"] == "experiments_complete", "Experiments have not completed")
    train_check = read(ROOT / "training_verification.json")
    require(train_check["passed"] and train_check["epochs"] == 30 and train_check["optimizer_steps"] == 18600,
            "C training budget not verified")
    meta, _ = read_manifest("manifests/member1_final_20261029/v1/test.json")
    samples = meta["samples"]
    by_condition, per_image, table, audits = {}, {}, [], []
    for model_name, (run, folder) in MODELS.items():
        result, identity = read(folder / "result.json"), read(folder / "identity.json")
        require(result["complete"] and result["image_count"] == 4428
                and result["conditions"] == list(CONDITIONS), "Full conditions missing")
        require(identity["checkpoint_sha256"] == sha256_file(run / "best.pt")
                and identity["code"] == code_identity(), "Model/code provenance mismatch")
        by_condition[model_name], per_image[model_name] = {}, {}
        for name in CONDITIONS:
            metrics, rows = audit_condition(folder, name, samples)
            by_condition[model_name][name] = metrics
            per_image[model_name][name] = rows
            audits.append({"model": model_name, "condition": name, "images": 4428,
                           "membership_exact": True, "pooled_region_metrics": True,
                           "boundary_image_mean": True, "nonfinite_values": 0})
            for group in GROUPS:
                table.append({"model": model_name, "condition": name, "subset": group,
                              **metrics["scores"][group]})
    reference_a = read("runs/member2_full_A_20261114/test_evaluation/clean/metrics.json")
    baseline_recheck = {}
    for group in GROUPS:
        actual, old = by_condition["A"]["clean"]["scores"][group], reference_a["scores"][group]
        deltas = {k: actual[k] - old[k] for k in METRICS}
        require(all(abs(deltas[k]) <= (0.0005 if k == "boundary_iou" else 0.0001) for k in METRICS),
                "Full A clean result differs materially from published A")
        baseline_recheck[group] = {"metric_deltas": deltas, "count_deltas": {k: actual[k] - old[k] for k in COUNTS}}
    robustness = []
    for model in MODELS:
        original_summary = read(MODELS[model][1] / "robustness.json")["summary"]
        for group in GROUPS:
            row = {"model": model, "subset": group, "images_per_condition": 4428 if group == "all"
                   else sum(s["difficulty"] == group for s in samples)}
            for metric in METRICS:
                clean = by_condition[model]["clean"]["scores"][group][metric]
                mean = math.fsum(by_condition[model][n]["scores"][group][metric] for n in CONDITIONS if n != "clean") / 12
                drop = clean - mean
                require(abs(mean - original_summary[group][metric]["perturbed_mean"]) <= 1e-12
                        and abs(drop - original_summary[group][metric]["drop"]) <= 1e-12, "Robustness summary mismatch")
                row.update({metric + "_clean": clean, metric + "_perturbed_mean": mean,
                            metric + "_drop": drop, metric + "_drop_pp": drop * 100})
            robustness.append(row)
    comparisons = []
    for group in GROUPS:
        a = next(r for r in robustness if r["model"] == "A" and r["subset"] == group)
        c = next(r for r in robustness if r["model"] == "C" and r["subset"] == group)
        for metric in METRICS:
            comparisons.append({"subset": group, "metric": metric,
                "A_clean_percent": a[metric + "_clean"] * 100, "C_clean_percent": c[metric + "_clean"] * 100,
                "C_minus_A_clean_pp": (c[metric + "_clean"] - a[metric + "_clean"]) * 100,
                "A_perturbed_mean_percent": a[metric + "_perturbed_mean"] * 100,
                "C_perturbed_mean_percent": c[metric + "_perturbed_mean"] * 100,
                "C_minus_A_perturbed_mean_pp": (c[metric + "_perturbed_mean"] - a[metric + "_perturbed_mean"]) * 100,
                "A_drop_pp": a[metric + "_drop_pp"], "C_drop_pp": c[metric + "_drop_pp"],
                "C_minus_A_drop_pp": c[metric + "_drop_pp"] - a[metric + "_drop_pp"]})
    output.mkdir()
    write_csv(output / "condition_results.csv", table)
    write_csv(output / "robustness_summary.csv", robustness)
    write_csv(output / "A_C_comparison.csv", comparisons)
    selection = read(ROOT / "qualitative_selection.json")
    failures = []
    worst_clean = sorted(per_image["C"]["clean"].values(), key=lambda r: (float(r["foreground_iou"]), r["id"]))[:3]
    worst_delta = sorted(per_image["C"]["gamma_1.6"].values(), key=lambda r: (
        float(r["foreground_iou"]) - float(per_image["A"]["gamma_1.6"][r["id"]]["foreground_iou"]), r["id"]))[:3]
    for reason, chosen in (("three lowest C clean IoU", worst_clean),
                            ("three smallest C-minus-A IoU under gamma_1.6", worst_delta)):
        condition = "clean" if "clean" in reason else "gamma_1.6"
        for row in chosen:
            failures.append({"id": row["id"], "selection_reason": reason, "condition": condition,
                "A_iou": float(per_image["A"][condition][row["id"]]["foreground_iou"]),
                "C_iou": float(row["foreground_iou"])})
    write_csv(output / "failure_selection.csv", failures)
    gallery = {"fixed_ids": selection["fixed_ids"], "conditions": selection["conditions"], "failures": failures,
               "all_ids": list(dict.fromkeys(selection["fixed_ids"] + [f["id"] for f in failures]))}
    write_json(output / "gallery_selection.json", gallery)
    summary = {"stage": 4, "automatic_checks": "passed", "human_acceptance": "pending",
        "stage5_started": False, "generated_at": datetime.now(timezone.utc).isoformat(),
        "C_training": train_check, "A_clean_published_recheck": baseline_recheck,
        "condition_audits": audits, "conditions_per_model": 13, "images_per_condition": 4428,
        "total_image_condition_evaluations": 2 * 13 * 4428,
        "comparison": comparisons, "robustness_summary": robustness,
        "checkpoint_paths": {k: str(run / "best.pt") for k, (run, _) in MODELS.items()},
        "checkpoint_sha256": {k: sha256_file(run / "best.pt") for k, (run, _) in MODELS.items()},
        "code_identity": code_identity(),
        "limitations": ["one seed", "A/C core computation libraries and budget match; GPU, Python and some auxiliary dependencies differ (see environment_comparison.json)",
                       "reviewed Trans10K derivative train/validation, original test membership preserved",
                       "no GSD/transfer conclusion yet", "B/D are not part of this A/C experiment",
                       "test results were not used for model/parameter selection"]}
    write_json(output / "summary.json", summary)
    write_json(output / "plot_data.json", {"models": by_condition, "training_run": str(TRAIN)})
    print(json.dumps({"status": "passed", "selected_C_epoch": train_check["selected_epoch"],
                      "all_subset_comparisons": [r for r in comparisons if r["subset"] == "all"]}, indent=2))


if __name__ == "__main__":
    main()
