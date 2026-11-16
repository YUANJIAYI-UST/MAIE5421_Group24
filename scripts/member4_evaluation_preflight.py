"""Real-weight equivalence, isolated timing and deliberate process-death recovery."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from transparent_seg.data import SegmentationDataset
from transparent_seg.evaluate import evaluate_selection, load_checkpoint
import transparent_seg.evaluate_resumable as engine
from transparent_seg.prepare import sha256_file, write_json

ROOT = Path("runs/member4_stage3_20261115")
A = Path("runs/member2_full_A_20261114/best.pt")
TIMING = ROOT / "timing40.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def crash_worker(output):
    # Only this diagnostic subprocess is terminated. The production evaluator is unchanged.
    original = engine.predict_resized
    calls = [0]

    def interrupted(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 25 * 13 + 1:
            partial = output / "chunks/000025.json.tmp"
            with partial.open("wb") as stream:
                stream.write(b'{"incomplete_diagnostic_chunk":')
                stream.flush()
                os.fsync(stream.fileno())
            write_json(output / "injected_process_death.json", {
                "completed_image_conditions": 325, "exit_code": 86,
                "mechanism": "os._exit(86) skips finally/normal cleanup after first complete 25-image block",
                "incomplete_file_sha256": sha256_file(partial)})
            os._exit(86)
        return original(*args, **kwargs)
    engine.predict_resized = interrupted
    engine.run(A, TIMING, output)
    raise RuntimeError("Diagnostic process did not stop at the injection point")


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    pilot = ROOT / "training_preflight/pilot_C_20_steps"
    model, checkpoint = load_checkpoint(pilot / "best.pt", "cuda")
    config = checkpoint["config"]
    dataset = SegmentationDataset(config["val_manifest"], config["size"], cache_dir=config["validation_cache"])
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        validation = evaluate_selection(model, dataset, "cuda", progress_every=200)
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - start
    history = json.loads((pilot / "history.jsonl").read_text().splitlines()[0])
    require(validation["scores"] == history["validation"], "Reloaded full validation differs")
    write_json(output / "isolated_validation_timing.json", {
        "images": len(dataset), "seconds": validation_seconds, "cache_initialization_excluded": True,
        "no_concurrent_test_or_cache_job": True, "reloaded_best_matches_training_record_exactly": True,
        "scores": validation["scores"], "purpose": "pilot reload and per-epoch validation timing only"})
    del model, checkpoint, dataset
    torch.cuda.empty_cache()
    print(f"Isolated full validation: {validation_seconds:.3f}s, exact reload match", flush=True)
    exact = engine.run(A, ROOT / "equivalence8.json", output / "equivalence8", verify_legacy=True)
    require(exact["complete"] and exact["image_count"] == 8, "Equivalence coverage incomplete")
    print("Real-image exact equivalence passed: 8 images x 13 conditions", flush=True)
    continuous = engine.run(A, TIMING, output / "continuous40")
    require(continuous["complete"] and continuous["image_count"] == 40, "Timing coverage incomplete")
    torch.cuda.empty_cache()
    start = time.perf_counter()
    process = subprocess.run([sys.executable, "-X", "utf8", "-B", str(Path(__file__)),
                              "--crash-worker", "--output", str(output / "resumed40")])
    interrupted_wall = time.perf_counter() - start
    require(process.returncode == 86, f"Unexpected worker exit {process.returncode}")
    require((output / "resumed40/chunks/000025.json.tmp").exists(), "Expected unfinished file")
    partial = json.loads((output / "resumed40/progress.json").read_text())
    require(partial["completed_images"] == 25 and not partial["complete"], "Interruption state incorrect")
    completed_chunk_hash = sha256_file(output / "resumed40/chunks/000000.json")
    resumed = engine.run(A, TIMING, output / "resumed40", resume=True)
    require(resumed["complete"] and resumed["reused_chunks"] == 1 and resumed["computed_chunks"] == 1,
            "Expected first block reuse and remaining block recomputation")
    require(sha256_file(output / "resumed40/chunks/000000.json") == completed_chunk_hash,
            "Complete block should not be rewritten")
    require(not (output / "resumed40/chunks/000025.json.tmp").exists(), "Unfinished file not replaced")
    require(continuous["scores"] == resumed["scores"], "Recovered aggregate scores differ")
    for name in engine.CONDITIONS:
        require((output / "continuous40" / name / "per_image.csv").read_bytes()
                == (output / "resumed40" / name / "per_image.csv").read_bytes(),
                "Recovered per-image scores/prediction hashes differ: " + name)
    report = {"status": "passed", "formal_evaluation_started": False,
        "equivalence_images": 8, "equivalence_conditions": 13,
        "every_prediction_pixel_and_metric_exact": True,
        "timing_images": 40, "timing_conditions": 13, "continuous": continuous,
        "hard_process_exit_code": process.returncode, "interrupted_worker_wall_seconds": interrupted_wall,
        "resume": resumed, "recovered_per_image_csv_bytes_exact_all_13": True,
        "complete_chunk_reused_without_rewrite": True, "incomplete_chunk_replaced": True,
        "isolated_full_validation_seconds": validation_seconds,
        "validation_reloaded_exact": True, "code_identity": engine.code_identity(),
        "script_sha256": sha256_file(__file__),
        "limits": ["40 validation images, 20 easy/20 hard, selected before timing.",
                   "Timing uses A; C has the same inference architecture but masks/thermal conditions may affect throughput.",
                   "No test subset or formal A/C score is produced here."]}
    write_json(output / "evaluation_preflight_report.json", report)
    print(json.dumps({"status": "passed", "validation_seconds": validation_seconds,
        "continuous_40_images_13_conditions_seconds": continuous["processing_seconds"],
        "continuous_total_seconds": continuous["elapsed_seconds"],
        "resume_total_seconds": resumed["elapsed_seconds"], "hard_crash_recovery_exact": True}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crash-worker", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if args.crash_worker:
        crash_worker(output)
    else:
        try:
            main(output)
        except Exception as exc:
            if output.exists() and not (output / "evaluation_preflight_report.json").exists():
                write_json(output / "evaluation_preflight_failure.json", {"status": "failed", "error": str(exc)})
            raise
