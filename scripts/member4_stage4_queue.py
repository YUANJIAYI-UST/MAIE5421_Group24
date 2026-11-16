"""Run the authorized frozen C training and full A/C experiments sequentially."""
import argparse
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transparent_seg.prepare import sha256_file
from transparent_seg.protocol import validate_frozen_protocol, SHARED_INITIAL_SHA256
from transparent_seg.evaluate_resumable import atomic_json, exclusive_run

ROOT = Path("runs/member4_stage4_20261115")
TRAIN = Path("runs/member4_full_C_seed42_20261115")
A = Path("runs/member2_full_A_20261114")
A_TEST = Path("runs/member4_A_full_test_20261115")
C_TEST = Path("runs/member4_C_full_test_20261115")
C_VALIDATION = Path("runs/member4_C_full_validation_20261115")
CONFIG = Path("configs/member4_full_C_20261115.json")
MANIFESTS = Path("manifests/member1_final_20261029/v1")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen_checks():
    pins = read("runs/member4_stage3_20261115/acceptance/source_sha256.json")
    for path, digest in pins.items():
        require(sha256_file(path) == digest, "Accepted stage3 source/config changed: " + path)
    require(sha256_file(A / "best.pt") == "3a785feb578ecbd2199684c53130737c872cd6c2f5803f221297f1e9dac0b25a",
            "Published A checkpoint changed")
    require(sha256_file(MANIFESTS / "test.json") == "ecf3ad309423aa0247183cfc1e315921a7666c47d10700f381199a7dcfb41012",
            "Formal test manifest changed")
    config = read(CONFIG)
    validate_frozen_protocol(config, read(config["train_manifest"]), read(config["val_manifest"]))
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Required CUDA/BF16 unavailable")
    require("5090" in torch.cuda.get_device_name(0), "Expected local 5090")
    return pins


def history():
    return [json.loads(line) for line in (TRAIN / "history.jsonl").read_text(encoding="utf-8").splitlines()]


def verify_training():
    records = history()
    require([r["epoch"] for r in records] == list(range(1, 31)), "Missing/repeated training epoch")
    require(all(r["global_optimizer_step"] == r["epoch"] * 620 and r["optimizer_steps_per_epoch"] == 620
                and r["effective_batch_size"] == 8 and r["validation"]["all"]["images"] == 995
                for r in records), "Training/validation budget mismatch")
    selected = max(records, key=lambda r: r["validation"]["all"]["foreground_iou"])
    best = torch.load(TRAIN / "best.pt", map_location="cpu", weights_only=True)
    last = torch.load(TRAIN / "last.pt", map_location="cpu", weights_only=True)
    require(last["epoch"] == 30 and last["global_step"] == 18600, "Final budget incomplete")
    require(best["epoch"] == selected["epoch"]
            and best["validation_foreground_iou"] == selected["validation"]["all"]["foreground_iou"],
            "Best model was not the earliest maximum validation IoU")
    for checkpoint in (best, last):
        require(checkpoint["config"] == read(CONFIG)
                and checkpoint["initial_shared_state_sha256"] == SHARED_INITIAL_SHA256,
                "Checkpoint configuration/initialization differs")
    previous = -1.0
    for row in records:
        score = row["validation"]["all"]["foreground_iou"]
        require(row["selected"] == (score > previous), "Selection rule differs")
        previous = max(previous, score)
    result = {"passed": True, "epochs": 30, "optimizer_steps": 18600, "validation_images_per_epoch": 995,
              "selected_epoch": best["epoch"], "selected_validation": selected["validation"],
              "initial_shared_state_sha256": SHARED_INITIAL_SHA256,
              "best_sha256": sha256_file(TRAIN / "best.pt"), "last_sha256": sha256_file(TRAIN / "last.pt")}
    atomic_json(ROOT / "training_verification.json", result)
    return result


def verify_evaluation(folder, count, checkpoint, names):
    result, identity = read(folder / "result.json"), read(folder / "identity.json")
    require(result["complete"] and result["image_count"] == count and len(result["conditions"]) == names,
            "Evaluation is incomplete")
    require(identity["checkpoint_sha256"] == sha256_file(checkpoint), "Wrong evaluated checkpoint")
    for name in result["conditions"]:
        scores = result["scores"][name]["all"]
        require(scores["images"] == scores["boundary_images"] == count and scores["skipped_all_void"] == 0,
                "Incomplete region or boundary metrics")
    return result


def command(stage, arguments, state, plot_python=None):
    frozen_checks()
    log = ROOT / "logs" / (stage + f"_{time.time_ns()}.log")
    log.parent.mkdir(exist_ok=True)
    started = time.perf_counter()
    with log.open("x", encoding="utf-8", buffering=1) as stream:
        process = subprocess.Popen([sys.executable, "-X", "utf8", "-u", "-B", *arguments],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        state.update(stage=stage, status="running", child_pid=process.pid, updated_at=now())
        state["jobs"].append({"stage": stage, "arguments": arguments, "log": str(log),
                              "started_at": now(), "pid": process.pid})
        atomic_json(ROOT / "state.json", state)
        try:
            for line in process.stdout:
                stream.write(line)
                if line.startswith('{"epoch":'):
                    record = json.loads(line)
                    state["last_epoch"] = record["epoch"]
                    state["last_validation_iou"] = record["validation"]["all"]["foreground_iou"]
                    state["updated_at"] = now()
                    atomic_json(ROOT / "state.json", state)
                    print(f"C epoch {record['epoch']}/30; validation IoU "
                          f"{record['validation']['all']['foreground_iou']:.6f}; "
                          f"{record['seconds']:.1f}s", flush=True)
                    if plot_python:
                        plot = subprocess.run([plot_python, "scripts/plot_member4_progress.py",
                            "--history", str(TRAIN / "history.jsonl"), "--output", str(ROOT / "figures")],
                            stdout=stream, stderr=stream)
                        if plot.returncode:
                            print("Training continues; curve renderer requires inspection.", flush=True)
                else:
                    print(line.rstrip(), flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    state["jobs"][-1].update(exit_code=code, seconds=time.perf_counter() - started, ended_at=now())
    atomic_json(ROOT / "state.json", state)
    require(code == 0, f"{stage} failed with exit code {code}; settings unchanged; see {log}")


def main(resume, plot_python):
    pins = frozen_checks()
    if not resume:
        require(not ROOT.exists(), "Use --resume for an existing queue")
        require(not any(p.exists() for p in (TRAIN, A_TEST, C_TEST, C_VALIDATION)), "Formal output already exists")
        ROOT.mkdir(parents=True)
        atomic_json(ROOT / "authorization.json", {
            "accepted_by": "user", "recorded_at": now(), "stage": 4,
            "instruction": "阶段3通过，请用本机5090启动阶段4的正式训练与全量A/C评估，按冻结配置连续执行并保存日志和检查点，完成后给我看结果，等我验收后再进入阶段5。",
            "stage5_authorized": False, "stage3_report_sha256": sha256_file("runs/member4_stage3_20261115/acceptance/stage3_report.json")})
        atomic_json(ROOT / "source_sha256.json", {**pins, "scripts/member4_stage4_queue.py": sha256_file(__file__)})
        test = read(MANIFESTS / "test.json")
        selected = []
        for group in ("easy", "hard"):
            samples = [s for s in test["samples"] if s["difficulty"] == group]
            samples.sort(key=lambda s: hashlib.sha256(("member4-stage4/42/" + s["id"]).encode()).hexdigest())
            selected.extend(samples[:6])
        atomic_json(ROOT / "qualitative_selection.json", {
            "fixed_rule": "lowest SHA256(member4-stage4/42/<id>), six easy and six hard",
            "fixed_ids": [s["id"] for s in selected],
            "conditions": ["clean", "brightness_0.6", "contrast_0.5", "gamma_1.6"],
            "failure_rules": ["three lowest C clean foreground IoU, ties by id",
                              "three smallest C-minus-A foreground IoU under gamma_1.6, ties by id"],
            "note": "Selection rules fixed before any formal C/test results; failures are diagnostic examples, not additional tuning."})
        state = {"status": "starting", "stage": "training_C", "started_at": now(), "jobs": [],
                 "gpu": torch.cuda.get_device_name(0), "queue_pid": os.getpid(), "stage5_started": False}
    else:
        state = read(ROOT / "state.json")
        state.update(queue_pid=os.getpid(), resumed_at=now())
    with exclusive_run(ROOT):
        awake = False
        try:
            if os.name == "nt":
                awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            state["automatic_sleep_inhibited_for_this_process"] = awake
            atomic_json(ROOT / "state.json", state)
            if not (ROOT / "training_verification.json").exists():
                if TRAIN.exists():
                    # Existing training is only skipped when its full budget verifies.
                    # Partial training must be recovered into a new directory by the existing trainer.
                    verify_training()
                else:
                    command("training_C", ["-m", "transparent_seg.train", "--config", str(CONFIG),
                                           "--output", str(TRAIN)], state, plot_python)
                    verify_training()
            else:
                verify_training()
            for stage, folder, model, manifest, conditions, count in (
                ("validation_C", C_VALIDATION, TRAIN / "best.pt", MANIFESTS / "validation.json", "clean", 995),
                ("test_A", A_TEST, A / "best.pt", MANIFESTS / "test.json", "all", 4428),
                ("test_C", C_TEST, TRAIN / "best.pt", MANIFESTS / "test.json", "all", 4428),
            ):
                result_path = folder / "result.json"
                if not result_path.exists() or not read(result_path).get("complete"):
                    args = ["-m", "transparent_seg.evaluate_resumable", "--checkpoint", str(model),
                            "--manifest", str(manifest), "--conditions", conditions, "--output", str(folder)]
                    if folder.exists():
                        args.append("--resume")
                    command(stage, args, state)
                result = verify_evaluation(folder, count, model, 1 if conditions == "clean" else 13)
                if stage == "validation_C":
                    selected = read(ROOT / "training_verification.json")["selected_validation"]
                    for group in ("all", "easy", "hard"):
                        for key in ("tp", "fp", "fn", "tn", "foreground_iou", "region_f1"):
                            require(result["scores"]["clean"][group][key] == selected[group][key],
                                    "Reloaded full C validation differs from selected epoch")
                    atomic_json(ROOT / "validation_verification.json", {"passed": True, "images": 995,
                                "selection_region_counts_exact": True, "boundary_metrics_complete": True})
            frozen_checks()
            state.update(status="experiments_complete", stage="awaiting_report_assembly",
                         ended_at=now(), child_pid=None, stage5_started=False)
            atomic_json(ROOT / "state.json", state)
            print("Frozen stage4 experiments complete. Report assembly and user acceptance remain; stage5 not started.", flush=True)
        except BaseException as exc:
            state.update(status="stopped_requires_inspection", error=f"{type(exc).__name__}: {exc}", updated_at=now())
            atomic_json(ROOT / "state.json", state)
            raise
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--plot-python")
    args = parser.parse_args()
    if args.check_only:
        frozen_checks()
        print("Frozen sources, configuration, formal manifests, A checkpoint and local 5090 checks passed.")
    else:
        main(args.resume, args.plot_python)
