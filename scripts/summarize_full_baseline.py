"""Archive a completed, validation-selected full A run and its clean evaluations."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transparent_seg.prepare import sha256_file, write_json


def summarize(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    if [r["epoch"] for r in records] != list(range(1, config["epochs"] + 1)):
        raise ValueError("Full, uninterrupted history is required")
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    selected = max(records, key=lambda r: r["validation"]["all"]["foreground_iou"])
    if (best["epoch"] != selected["epoch"] or last["epoch"] != config["epochs"]
            or last["global_step"] != 18600 or any(r["optimizer_steps_per_epoch"] != 620 for r in records)
            or any(r["validation"]["all"]["images"] != 995 for r in records)):
        raise ValueError("Training budget/selection checks failed")
    evaluations = {}
    for split, count in (("validation", 995), ("test", 4428)):
        folder = run / f"{split}_evaluation"
        metrics = json.loads((folder / "clean/metrics.json").read_text(encoding="utf-8"))
        provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        if metrics["scores"]["all"]["images"] != count or provenance["checkpoint_sha256"] != sha256_file(run / "best.pt"):
            raise ValueError("Evaluation membership/checkpoint mismatch")
        if metrics["scores"]["all"]["boundary_images"] != count:
            raise ValueError("Full boundary evaluation is missing")
        if split == "validation":
            for group in ("all", "easy", "hard"):
                for key in ("tp", "fp", "fn", "tn", "foreground_iou", "region_f1"):
                    if metrics["scores"][group][key] != selected["validation"][group][key]:
                        raise ValueError(f"Selection/final validation mismatch: {group}/{key}")
        shutil.copytree(folder, output / f"{split}_evaluation", dirs_exist_ok=True)
        evaluations[split] = metrics
    for name in ("config.json", "environment.json", "pip_freeze.txt", "history.jsonl"):
        shutil.copy2(run / name, output / name)
    rows = [{"epoch": r["epoch"], "train_loss": r["loss"]["total"],
        "validation_iou": r["validation"]["all"]["foreground_iou"],
        "validation_f1": r["validation"]["all"]["region_f1"], "selected": r["selected"],
        "learning_rate": r["lr"], "train_seconds": r["train_seconds"],
        "validation_seconds": r["validation_seconds"], "peak_cuda_memory_mb": r["peak_cuda_memory_mb"],
        "peak_cuda_reserved_mb": r["peak_cuda_reserved_mb"], "steady_microbatch_seconds": r["steady_microbatch_seconds"]}
        for r in records]
    with (output / "training_curve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"status": "completed", "scope": "member2_full_A_baseline", "epochs": len(records),
        "optimizer_steps": last["global_step"], "selected_epoch": best["epoch"],
        "selection": "full validation pooled foreground IoU; earliest tie",
        "train_seconds": sum(r["train_seconds"] for r in records),
        "validation_seconds": sum(r["validation_seconds"] for r in records),
        "training_loop_seconds": sum(r["seconds"] for r in records),
        "peak_cuda_memory_mb": max(r["peak_cuda_memory_mb"] for r in records),
        "peak_cuda_reserved_mb": max(r["peak_cuda_reserved_mb"] for r in records),
        "first_loss": rows[0]["train_loss"], "last_loss": rows[-1]["train_loss"],
        "checkpoint_sha256": {name: sha256_file(run / name) for name in ("best.pt", "last.pt")},
        "evaluations": evaluations, "member3_human_review": "pending",
        "limitations": ["single seed", "member1 reviewed derivative, not unfiltered official train/val",
                        "B/C/D and robustness/transfer experiments are separate team tasks"]}
    write_json(output / "summary.json", summary)
    print(json.dumps({"epochs": len(records), "selected_epoch": best["epoch"],
        "validation": evaluations["validation"]["scores"]["all"], "test": evaluations["test"]["scores"]["all"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summarize(args.run, args.output)
