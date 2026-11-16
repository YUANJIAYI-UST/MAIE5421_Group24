"""Report-ready A/C robustness curves; plotting runs outside the frozen environment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(root):
    root = Path(root)
    models = json.loads((root / "plot_data.json").read_text(encoding="utf-8"))["models"]
    operations = {"brightness": [0.6, 0.8, 1.2, 1.4],
                  "contrast": [0.5, 0.75, 1.25, 1.5], "gamma": [0.6, 0.8, 1.2, 1.6]}
    metrics = {"foreground_iou": "Foreground IoU (%)", "region_f1": "Region F1 (%)", "boundary_iou": "Boundary IoU (%)"}
    for group in ("all", "easy", "hard"):
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
        for col, (op, values) in enumerate(operations.items()):
            for row, (metric, label) in enumerate(metrics.items()):
                ax = axes[row, col]
                for model, color in (("A", "#1764ab"), ("C", "#d95f02")):
                    x = sorted(values + [1.0])
                    y = [models[model]["clean" if v == 1 else f"{op}_{v:g}"]["scores"][group][metric] * 100 for v in x]
                    ax.plot(x, y, label=model, color=color, marker="o", linewidth=1.7)
                ax.axvline(1.0, color="#888888", linestyle=":", linewidth=1)
                ax.set(xlabel=op.capitalize() + (" exponent" if op == "gamma" else " factor"),
                       ylabel=label)
                ax.grid(alpha=0.2)
                if row == 0:
                    ax.set_title(op.capitalize())
                    ax.legend(frameon=False)
        fig.suptitle(f"A vs C | {group} | Full Trans10K test | dashed line = clean", fontsize=14)
        fig.savefig(root / f"robustness_{group}.png", dpi=180)
        fig.savefig(root / f"robustness_{group}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    main(args.root)
