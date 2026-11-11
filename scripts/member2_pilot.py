"""Measure real-data 512px A-baseline pilots at batch 4 and 8; no test-set use."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transparent_seg.prepare import write_json
from transparent_seg.train import load_config, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_member1_final_v1_20261029.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root")
    args = parser.parse_args()
    base = load_config(args.config)
    if base.get("purpose") != "pilot" or not base.get("pretrained"):
        raise ValueError("Resource pilot requires real-data pilot config and ImageNet initialization")
    if not torch.cuda.is_available():
        raise RuntimeError("M01 requires CUDA")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for batch in (4, 8):
        config = {**base, "size": 512, "batch_size": batch, "epochs": 1,
                  "boundary": False, "augmentation": False, "device": "cuda",
                  "gradient_accumulation_steps": 1, "max_steps_per_epoch": 20}
        if args.root:
            config["root"] = args.root
        write_json(output / f"batch{batch}.json", config)
        try:
            train(config, output / f"batch{batch}")
            record = json.loads((output / f"batch{batch}/history.jsonl").read_text().splitlines()[-1])
            rows.append({"batch_size": batch, "status": "measured", **record})
        except torch.cuda.OutOfMemoryError as error:
            rows.append({"batch_size": batch, "status": "out_of_memory", "error": str(error)})
        finally:
            torch.cuda.empty_cache()
            write_json(output / "resources.json", {"gpu": torch.cuda.get_device_name(),
                "torch": str(torch.__version__), "rows": rows,
                "note": "A baseline only; B/D need their own memory check. Step times exclude data loading; train_seconds includes it."})


if __name__ == "__main__":
    main()
