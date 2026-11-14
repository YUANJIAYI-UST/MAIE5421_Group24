"""Plot an archived full baseline (requires matplotlib, outside the training environment)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--pdf")
    args = parser.parse_args()
    root = Path(args.report)
    rows = [json.loads(line) for line in (root / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].plot(epochs, [row["loss"]["total"] for row in rows], color="#1764ab", linewidth=2)
    axes[0].set(xlabel="Epoch", ylabel="Training loss (BCE + Dice)", title="Training objective")
    for group, color in [("all", "#1764ab"), ("easy", "#2b9348"), ("hard", "#e07a15")]:
        axes[1].plot(epochs, [100 * row["validation"][group]["foreground_iou"] for row in rows],
                     label=group, color=color, linewidth=1.8)
    axes[1].axvline(summary["selected_epoch"], color="#666666", linestyle="--", linewidth=1,
                   label="Selected epoch")
    axes[1].set(xlabel="Epoch", ylabel="Pooled foreground IoU (%)",
                title="Full original-resolution validation")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(1, len(rows))
    fig.suptitle("A baseline | seed 42 | 4,959 training / 995 validation images", fontsize=12)
    fig.savefig(root / "training_curves.png", dpi=180)
    if args.pdf:
        fig.savefig(args.pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()
