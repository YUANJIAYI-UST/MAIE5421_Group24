"""Render training curves in a separate plotting environment."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(history, output):
    records = [json.loads(line) for line in Path(history).read_text(encoding="utf-8").splitlines()]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(epochs, [r["loss"]["total"] for r in records], color="#1764ab", marker=".", linewidth=1.8)
    axes[0].set(title="C training objective", xlabel="Epoch", ylabel="BCE + Dice")
    for group, color in (("all", "#1764ab"), ("easy", "#2b9348"), ("hard", "#e07a15")):
        axes[1].plot(epochs, [r["validation"][group]["foreground_iou"] * 100 for r in records],
                     label=group, color=color, marker=".", linewidth=1.8)
    best = max(records, key=lambda r: r["validation"]["all"]["foreground_iou"])
    axes[1].axvline(best["epoch"], color="#666666", linestyle="--", label="Best so far")
    axes[1].set(title="All 995 validation images each epoch", xlabel="Epoch", ylabel="Pooled foreground IoU (%)")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlim(0.5, max(2, len(records)) + 0.5)
    fig.suptitle(f"C | seed 42 | 4,959 train / 995 validation | {len(records)}/30 epochs", fontsize=12)
    temp = output / "training_curves.tmp.png"
    fig.savefig(temp, dpi=180)
    temp.replace(output / "training_curves.png")
    plt.close(fig)
    with (output / "training_curve.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss", "validation_iou", "selected", "global_optimizer_step"])
        writer.writeheader()
        for r in records:
            writer.writerow({"epoch": r["epoch"], "loss": r["loss"]["total"],
                "validation_iou": r["validation"]["all"]["foreground_iou"], "selected": r["selected"],
                "global_optimizer_step": r["global_optimizer_step"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.history, args.output)
