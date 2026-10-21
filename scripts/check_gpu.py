"""512x512 CUDA/AMP engineering check; synthetic tensors, NOT a dataset pilot."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transparent_seg.losses import segmentation_loss
from transparent_seg.model import TransparentSegformer
from transparent_seg.prepare import write_json
from transparent_seg.train import load_config, seed_everything


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: run scripts/setup.ps1 -Device cuda")
    config = load_config("configs/D_combined.json")
    seed_everything(42)
    model = TransparentSegformer(True, True, config["encoder_revision"], ".cache/huggingface").cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5)
    images = torch.randn(2, 3, 512, 512, device="cuda")
    target = torch.zeros(2, 1, 512, 512, device="cuda")
    target[:, :, 96:400, 128:384] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    torch.cuda.reset_peak_memory_stats()
    durations, losses = [], []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(images, return_boundary=True)
            loss = segmentation_loss(output, target, valid)["total"]
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        losses.append(loss.item())
    report = {"purpose": "synthetic CUDA engineering check; not Trans10K performance",
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "batch_size": 2, "size": 512,
              "pretrained": True, "encoder_revision": model.encoder_revision,
              "amp": True, "amp_dtype": "bfloat16", "steps": 4, "finite_forward_backward": True,
              "losses": losses, "seconds_per_step": durations,
              "mean_seconds_excluding_first": sum(durations[1:]) / 3,
              "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
    write_json("reports/gpu_check.json", report)
    print(report)


if __name__ == "__main__":
    main()
