"""Image-major FP32 evaluation with reusable GT boundaries and verified chunks."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import csv

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter
import torch
from torch.nn import functional as F

from .data import PERTURBATIONS, decode_mask, image_tensor, read_manifest
from .evaluate import load_checkpoint, predict
from .metrics import SegmentationMetrics, inner_boundary
from .prepare import sha256_file

CONDITIONS = {"clean": None, **{f"{op}_{value:g}": (op, value) for op, value in PERTURBATIONS}}
COUNT_KEYS = ("tp", "fp", "fn", "tn")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


@contextmanager
def exclusive_run(directory):
    """OS lock releases on process death; no unsafe manual stale-lock deletion."""
    stream = (Path(directory) / ".lock").open("a+b")
    try:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another evaluator is using this output directory") from exc
        yield
    finally:
        stream.close()


def erosion_backend(name):
    if name == "scipy":
        return None
    if name != "opencv":
        raise ValueError(f"Unknown boundary backend: {name}")
    import cv2

    def erode(mask, width, border):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * width + 1, 2 * width + 1))
        return cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                         borderValue=border).astype(bool)

    return erode


class PreparedTarget:
    """Reuse target-only work; preserve the original void/border conventions."""
    def __init__(self, target, valid, erode=None):
        self.y, self.v = np.asarray(target, bool), np.asarray(valid, bool)
        if self.y.shape != self.v.shape or self.y.ndim != 2:
            raise ValueError("Expected equal HxW target and validity")
        self.valid_count = int(np.count_nonzero(self.v))
        self.foreground = int(np.count_nonzero(self.y & self.v))
        self.width = max(1, int(round(0.02 * math.hypot(*self.y.shape))))
        self.erode = erode
        self.safe = self.v if self.v.all() else (
            erode(self.v, self.width, 1) if erode else minimum_filter(
                self.v, size=2 * self.width + 1, mode="constant", cval=1))
        self.has_boundary = bool(self.safe.any()) and self.valid_count > 0
        self.boundary = self.inner(self.y & self.v) & self.safe if self.has_boundary else None

    def inner(self, mask):
        if self.erode is None:
            return inner_boundary(mask, self.width)
        return mask & ~self.erode(mask, self.width, 0)

    def score(self, prediction):
        p = np.asarray(prediction, bool)
        if p.shape != self.y.shape:
            raise ValueError("Prediction/target shape mismatch")
        meter = SegmentationMetrics()
        if not self.valid_count:
            meter.skipped = 1
            return meter.compute()
        meter.images = 1
        meter.tp = int(np.count_nonzero(p & self.y & self.v))
        meter.fp = int(np.count_nonzero(p & self.v)) - meter.tp
        meter.fn = self.foreground - meter.tp
        meter.tn = self.valid_count - meter.tp - meter.fp - meter.fn
        if self.has_boundary:
            pb = self.inner(p & self.v) & self.safe
            union = np.count_nonzero(pb | self.boundary)
            meter.boundaries.append(float(np.count_nonzero(pb & self.boundary) / union) if union else 1.0)
        return meter.compute()


@torch.inference_mode()
def predict_resized(model, resized_image, original_shape, size, device, condition=None):
    # The same tensor conversion/model/bilinear-logits/sigmoid threshold as predict().
    tensor = image_tensor(resized_image, size, condition)[None].to(device)
    logits = model(tensor)["logits"]
    logits = F.interpolate(logits.float(), size=original_shape, mode="bilinear", align_corners=False)
    return logits.sigmoid()[0, 0].cpu().numpy() >= 0.5


def code_identity():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"commit": commit, "files": {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in sorted((root / "transparent_seg").glob("*.py"))}}


def merge_score(meter, score):
    for key in COUNT_KEYS:
        setattr(meter, key, getattr(meter, key) + score[key])
    meter.images += score["images"]
    meter.skipped += score["skipped_all_void"]
    if score["boundary_images"]:
        meter.boundaries.append(score["boundary_iou"])


def aggregate(rows, names):
    meters = {name: {g: SegmentationMetrics() for g in ("all", "easy", "hard")} for name in names}
    for row in rows:
        for name in names:
            groups = ["all"] + ([row["difficulty"]] if row["difficulty"] in ("easy", "hard") else [])
            for group in groups:
                merge_score(meters[name][group], row["conditions"][name]["metrics"])
    return {n: {g: m.compute() for g, m in groups.items()} for n, groups in meters.items()}


def read_chunk(path, identity_hash, start, expected_ids, names):
    """Return None for a damaged/incomplete block; never accept partial membership."""
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        body = envelope["body"]
        if envelope["sha256"] != digest(body):
            return None
        if (body["identity_sha256"] != identity_hash or body["start"] != start
                or body["end"] != start + len(expected_ids)
                or [r["id"] for r in body["rows"]] != expected_ids):
            return None
        for row in body["rows"]:
            if set(row["conditions"]) != set(names):
                return None
            for result in row["conditions"].values():
                score = result["metrics"]
                if (set(score) != set(SegmentationMetrics().compute())
                        or any(type(score[k]) is not int or score[k] < 0 for k in
                               (*COUNT_KEYS, "images", "skipped_all_void", "boundary_images"))
                        or score["images"] + score["skipped_all_void"] != 1
                        or score["boundary_images"] > score["images"]
                        or any(v is not None and not math.isfinite(v) for v in score.values())
                        or len(result["prediction_sha256"]) != 64):
                    return None
        return body
    except (OSError, ValueError, KeyError, TypeError, OverflowError, AttributeError, IndexError):
        return None


def archive_exports(output, names):
    """Do not leave stale 'complete' exports visible during a recovery attempt."""
    paths = [output / "result.json", output / "robustness.json"]
    paths += [output / n / f for n in names for f in ("metrics.json", "per_image.csv")]
    archive = output / "previous_exports" / str(time.time_ns())
    for path in paths:
        if path.is_file():
            target = archive / path.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)


def write_results(output, meta, manifest_hash, names, rows, identity_hash):
    if len(rows) != len(meta["samples"]) or [r["id"] for r in rows] != [s["id"] for s in meta["samples"]]:
        raise ValueError("Final image membership is incomplete or repeated")
    scores = aggregate(rows, names)
    results = {}
    for name in names:
        result = {"split": meta["split"], "dataset": meta["dataset"],
            "synthetic": meta.get("synthetic", False),
            "complete_official_split": meta.get("complete_official_split", False),
            "manifest_sha256": manifest_hash, "condition": CONDITIONS[name] or "clean",
            "threshold": 0.5, "resolution": "original", "scores": scores[name],
            "complete": True, "image_count": len(rows), "identity_sha256": identity_hash}
        folder = output / name
        atomic_json(folder / "metrics.json", result)
        temp = folder / "per_image.csv.tmp"
        with temp.open("w", encoding="utf-8", newline="") as stream:
            fields = ["id", *SegmentationMetrics().compute(), "prediction_sha256"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                item = row["conditions"][name]
                writer.writerow({"id": row["id"], **item["metrics"], "prediction_sha256": item["prediction_sha256"]})
        temp.replace(folder / "per_image.csv")
        results[name] = result
    if set(names) == set(CONDITIONS):
        summary = {}
        for group in ("all", "easy", "hard"):
            summary[group] = {}
            for metric in ("foreground_iou", "region_f1", "boundary_iou"):
                values = [scores[n][group][metric] for n in names if n != "clean"]
                mean = float(np.mean(values)) if all(v is not None for v in values) else None
                clean = scores["clean"][group][metric]
                summary[group][metric] = {"clean": clean, "perturbed_mean": mean,
                    "drop": clean - mean if clean is not None and mean is not None else None}
        atomic_json(output / "robustness.json", {"conditions": [results[n] for n in names if n != "clean"],
            "summary": summary, "units": "fraction; multiply drop by 100 for percentage points",
            "complete": True, "identity_sha256": identity_hash})
    return scores


def run(checkpoint, manifest, output, device="cuda", names=None, root=None, resume=False,
        chunk_size=25, stop_after_chunks=None, verify_legacy=False, boundary_backend="scipy"):
    if chunk_size < 1 or (stop_after_chunks is not None and stop_after_chunks < 1):
        raise ValueError("Chunk size and diagnostic stop must be positive")
    names = list(CONDITIONS) if names is None else list(names)
    if not names or len(set(names)) != len(names) or any(n not in CONDITIONS for n in names):
        raise ValueError("Choose unique registered conditions")
    output = Path(output)
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError("Use a new output directory or explicitly --resume")
    if resume and not (output / "identity.json").is_file():
        raise ValueError("Resume requires a previously initialized evaluation")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_run(output):
        return _run(checkpoint, manifest, output, device, names, root, resume,
                    chunk_size, stop_after_chunks, verify_legacy, boundary_backend)


def _run(checkpoint, manifest, output, device, names, root, resume, chunk_size, stop_after_chunks, verify_legacy,
         boundary_backend):
    started = time.perf_counter()
    torch.set_num_threads(4)
    erode = erosion_backend(boundary_backend)
    model, ckpt = load_checkpoint(checkpoint, device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(ckpt["config"].get("deterministic", False))
    size = ckpt["config"]["size"]
    meta, base = read_manifest(manifest, root)
    inventory = []
    for sample in meta["samples"]:
        hashes = {key: sha256_file(base / sample[key]) for key in ("image", "mask")}
        if any(sample.get(key + "_sha256", hashes[key]) != hashes[key] for key in hashes):
            raise ValueError("Source file differs from manifest hash: " + sample["id"])
        inventory.append({"id": sample["id"], **hashes})
    identity = {"schema": 1, "checkpoint_sha256": sha256_file(checkpoint),
        "manifest_sha256": sha256_file(manifest), "source_inventory_sha256": digest(inventory),
        "source_count": len(inventory), "conditions": {n: CONDITIONS[n] for n in names},
        "condition_order": names, "size": size, "chunk_size": chunk_size,
        "precision": "float32", "threshold": 0.5, "original_resolution": True,
        "code": code_identity(), "verify_legacy": verify_legacy,
        "runtime": {"python": platform.python_version(), "device": device,
            "gpu": torch.cuda.get_device_name(torch.device(device)) if device.startswith("cuda") else None,
            "versions": {p: version(p) for p in ("torch", "numpy", "Pillow", "scipy", "transformers")},
            "deterministic": ckpt["config"].get("deterministic", False),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "cpu_threads": 4}}
    identity_hash = digest(identity)
    if erode is not None:
        import cv2
        identity["runtime"]["boundary_backend"] = {"opencv": cv2.__version__}
        identity_hash = digest(identity)
    if resume:
        existing = json.loads((output / "identity.json").read_text(encoding="utf-8"))
        if canonical(existing) != canonical(identity):
            raise ValueError("Resume identity mismatch: checkpoint/data/conditions/code/runtime changed")
    else:
        atomic_json(output / "identity.json", identity)
        atomic_json(output / "source_inventory.json", inventory)
    startup_seconds = time.perf_counter() - started
    if resume:
        archive_exports(output, names)
    atomic_json(output / "progress.json", {"complete": False, "status": "running",
        "completed_images": 0, "total_images": len(meta["samples"]), "conditions": names})
    rows, reused, computed, processing_seconds = [], 0, 0, 0.0
    attempt = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "identity_sha256": identity_hash}
    try:
        for start in range(0, len(inventory), chunk_size):
            samples = meta["samples"][start:start + chunk_size]
            expected_ids = [s["id"] for s in samples]
            path = output / "chunks" / f"{start:06d}.json"
            block = read_chunk(path, identity_hash, start, expected_ids, names) if path.exists() else None
            if block is not None:
                reused += 1
            else:
                if path.exists():
                    invalid = output / "invalid_chunks" / (path.stem + f"_{time.time_ns()}.json")
                    invalid.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(invalid)
                block_started = time.perf_counter()
                block_rows = []
                for offset, sample in enumerate(samples):
                    image_started = time.perf_counter()
                    buffers = {key: (base / sample[key]).read_bytes() for key in ("image", "mask")}
                    if any(hashlib.sha256(buffers[k]).hexdigest() != inventory[start + offset][k] for k in buffers):
                        raise ValueError("Source changed during evaluation: " + sample["id"])
                    with Image.open(io.BytesIO(buffers["image"])) as im:
                        image = im.convert("RGB")
                    target, valid = decode_mask(io.BytesIO(buffers["mask"]), meta["encoding"], meta.get("void_values", []))
                    if target.shape != (image.height, image.width):
                        raise ValueError("Image/mask geometry differs")
                    prepared = PreparedTarget(target, valid, erode)
                    resized = image.resize((size, size), Image.Resampling.BILINEAR)
                    item = {"id": sample["id"], "difficulty": sample.get("difficulty", "unspecified"),
                            "pixels": image.width * image.height, "conditions": {}}
                    for name in names:
                        pred = predict_resized(model, resized, target.shape, size, device, CONDITIONS[name])
                        score = prepared.score(pred)
                        if verify_legacy:
                            reference = predict(model, image, size, device, CONDITIONS[name])
                            if not np.array_equal(pred, reference):
                                raise AssertionError(f"Prediction pixels differ: {sample['id']} {name}")
                            meter = SegmentationMetrics()
                            meter.update(reference, target, valid)
                            if score != meter.compute():
                                raise AssertionError(f"Metrics differ: {sample['id']} {name}")
                        item["conditions"][name] = {"metrics": score,
                            "prediction_sha256": hashlib.sha256(np.packbits(pred).tobytes()).hexdigest()}
                    item["seconds"] = time.perf_counter() - image_started
                    block_rows.append(item)
                duration = time.perf_counter() - block_started
                processing_seconds += duration
                block = {"identity_sha256": identity_hash, "start": start, "end": start + len(samples),
                         "rows": block_rows, "seconds": duration}
                atomic_json(path, {"body": block, "sha256": digest(block)})
                computed += 1
            rows.extend(block["rows"])
            complete = len(rows) == len(meta["samples"])
            atomic_json(output / "progress.json", {"complete": False, "status": "processing",
                "all_chunks_available": complete, "completed_images": len(rows),
                "total_images": len(meta["samples"]), "conditions": names, "reused_chunks": reused,
                "computed_chunks": computed, "elapsed_seconds": time.perf_counter() - started})
            print(f"Evaluation: {len(rows)}/{len(meta['samples'])} images x {len(names)} conditions; "
                  f"computed {computed}, reused {reused} chunks", flush=True)
            if stop_after_chunks is not None and computed >= stop_after_chunks:
                break
        complete = len(rows) == len(meta["samples"])
        scores = write_results(output, meta, sha256_file(manifest), names, rows, identity_hash) if complete else None
        result = {"complete": complete, "image_count": len(rows), "expected_images": len(meta["samples"]),
            "conditions": names, "reused_chunks": reused, "computed_chunks": computed,
            "startup_seconds": startup_seconds, "processing_seconds": processing_seconds,
            "elapsed_seconds": time.perf_counter() - started, "scores": scores,
            "identity_sha256": identity_hash, "verify_legacy": verify_legacy}
        atomic_json(output / "result.json", result)
        atomic_json(output / "progress.json", {"complete": complete,
            "status": "complete" if complete else "diagnostic_pause", "completed_images": len(rows),
            "total_images": len(meta["samples"]), "conditions": names,
            "reused_chunks": reused, "computed_chunks": computed, "elapsed_seconds": result["elapsed_seconds"]})
        attempt.update(status="complete" if complete else "diagnostic_pause", **result)
        return result
    except BaseException as exc:
        attempt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error=f"{type(exc).__name__}: {exc}", completed_images=len(rows))
        raise
    finally:
        atomic_json(output / "attempts" / f"{time.time_ns()}.json", attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root")
    parser.add_argument("--conditions", default="all", help="all or comma-separated registered names, e.g. clean,gamma_0.8")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--stop-after-chunks", type=int, help="Diagnostic pause after complete newly computed chunks")
    parser.add_argument("--verify-legacy", action="store_true", help="Diagnostic exact pixel/metric comparison; not for timing")
    parser.add_argument("--boundary-backend", choices=("scipy", "opencv"), default="scipy")
    args = parser.parse_args()
    names = None if args.conditions == "all" else args.conditions.split(",")
    result = run(args.checkpoint, args.manifest, args.output, args.device, names, args.root,
                 args.resume, args.chunk_size, args.stop_after_chunks, args.verify_legacy, args.boundary_backend)
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
