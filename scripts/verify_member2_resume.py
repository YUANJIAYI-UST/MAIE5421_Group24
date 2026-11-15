"""Real SegFormer resume check on synthetic fixtures; not a research result."""
import argparse
import sys
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transparent_seg.evaluate import load_checkpoint
from transparent_seg.prepare import write_json
from transparent_seg.train import load_config, train


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    config = load_config("configs/smoke.json")
    config.update(epochs=2, device=args.device, amp=args.device.startswith("cuda"),
                  gradient_accumulation_steps=2, deterministic=True)
    train(config, output / "continuous")
    train(config, output / "partial", stop_after_epoch=1)
    train(config, output / "resumed", resume=output / "partial/last.pt")
    full = torch.load(output / "continuous/last.pt", weights_only=True)
    resumed = torch.load(output / "resumed/last.pt", weights_only=True)
    checks = {key: equal(full[key], resumed[key]) for key in full}
    model, _ = load_checkpoint(output / "resumed/best.pt", args.device)
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 64, 64, device=args.device))["logits"]
    checks["inference_load_finite"] = bool(torch.isfinite(logits).all())
    write_json(output / "verification.json", {"synthetic_only": True,
        "device": args.device, "torch": str(torch.__version__), "checks": checks,
        "passed": all(checks.values())})
    if not all(checks.values()):
        raise SystemExit("Resume differs; inspect verification.json")
    print("All checkpoint states match exactly; inference reload passed.")


if __name__ == "__main__":
    main()
