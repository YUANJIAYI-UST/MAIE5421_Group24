"""Render preselected and explicitly ranked failure cases after full evaluation."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from transparent_seg.data import read_manifest, decode_mask, image_tensor, MEAN, STD
from transparent_seg.evaluate import load_checkpoint, predict
from transparent_seg.evaluate_resumable import CONDITIONS
from transparent_seg.prepare import write_json

ROOT = Path("runs/member4_stage4_20261115")


def main():
    result = ROOT / "results"
    selection = json.loads((result / "gallery_selection.json").read_text(encoding="utf-8"))
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    meta, base = read_manifest("manifests/member1_final_20261029/v1/test.json")
    samples = {s["id"]: s for s in meta["samples"]}
    folder = result / "gallery"
    folder.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    models = {name: load_checkpoint(path, "cuda")[0] for name, path in summary["checkpoint_paths"].items()}
    scores = {}
    for model in models:
        scores[model] = {}
        evaluation = Path(f"runs/member4_{model}_full_test_20261115")
        for condition in selection["conditions"]:
            with (evaluation / condition / "per_image.csv").open(encoding="utf-8", newline="") as stream:
                scores[model][condition] = {r["id"]: r for r in csv.DictReader(stream)}
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 17)
    rows = []
    for index, identity in enumerate(selection["all_ids"]):
        sample = samples[identity]
        with Image.open(base / sample["image"]) as im:
            image = im.convert("RGB")
        target, valid = decode_mask(base / sample["mask"], meta["encoding"], meta.get("void_values", []))
        reasons = (["preselected before formal training"] if identity in selection["fixed_ids"] else [])
        reasons += [r["selection_reason"] for r in selection["failures"] if r["id"] == identity]
        canvas = Image.new("RGB", (1280, 62 + 354 * len(selection["conditions"])), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 7), f"{index + 1:02d}  {identity}", fill="black", font=font)
        draw.text((10, 31), "实际增强输入预览 / 人工标签 / A预测叠图 / C预测叠图；选择依据见 gallery_index.json", fill="black", font=font)
        thumb = image.copy()
        thumb.thumbnail((306, 304))
        base_thumb = np.array(thumb, dtype=np.float32)
        for row, condition in enumerate(selection["conditions"]):
            preds = {name: predict(model, image, 512, "cuda", CONDITIONS[condition]) for name, model in models.items()}
            for name, p in preds.items():
                digest = hashlib.sha256(np.packbits(p).tobytes()).hexdigest()
                if digest != scores[name][condition][identity]["prediction_sha256"]:
                    raise ValueError(f"Gallery prediction differs from saved evaluation: {identity}/{name}/{condition}")
            tensor = image_tensor(image, 512, CONDITIONS[condition]).numpy().transpose(1, 2, 0)
            pixels = np.rint(np.clip(tensor * STD + MEAN, 0, 1) * 255).astype(np.uint8)
            input_view = Image.fromarray(pixels).resize(thumb.size, Image.Resampling.LANCZOS)
            label_view = Image.fromarray(target.astype(np.uint8) * 255).resize(thumb.size, Image.Resampling.NEAREST).convert("RGB")
            views = [input_view, label_view]
            for name in ("A", "C"):
                p = Image.fromarray(preds[name].astype(np.uint8) * 255).resize(thumb.size, Image.Resampling.NEAREST)
                rgb = base_thumb.copy()
                foreground = np.array(p) > 0
                rgb[foreground] = 0.5 * rgb[foreground] + 0.5 * np.array([0, 220, 160])
                views.append(Image.fromarray(rgb.astype(np.uint8)))
            labels = [condition, "GT：白色为前景",
                      f"A IoU {float(scores['A'][condition][identity]['foreground_iou']):.4f}",
                      f"C IoU {float(scores['C'][condition][identity]['foreground_iou']):.4f}"]
            for col, (view, label) in enumerate(zip(views, labels)):
                top = 62 + row * 354
                draw.text((col * 320 + 8, top), label, fill="black", font=font)
                canvas.paste(view, (col * 320 + (320 - view.width) // 2, top + 28 + (310 - view.height) // 2))
        filename = f"{index + 1:02d}.png"
        canvas.save(folder / filename)
        rows.append({"id": identity, "file": filename, "selection_reasons": reasons, "prediction_hashes_match_full_evaluation": True})
        print(f"Gallery {index + 1}/{len(selection['all_ids'])}: {identity}", flush=True)
    write_json(folder / "gallery_index.json", rows)


if __name__ == "__main__":
    main()
