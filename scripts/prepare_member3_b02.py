"""Create hash-pinned configs/protocols for the Member 3 B02 validation grid."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs/member2_full_A_20261114.json"
TRAIN = ROOT / "manifests/member1_final_20261029/v1/train.json"
VALIDATION = ROOT / "manifests/member1_final_20261029/v1/validation.json"
OUTPUT = ROOT / "configs/member3_b02"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    # Use the user's Apple M5 GPU through MPS while preserving BF16 and the
    # project's pinned Python package versions. Device differences from A are
    # recorded in each run's environment.json.
    base["device"] = "cuda"
    base["expected_versions"]["torch"] = "2.7.1+cu128"
    train = json.loads(TRAIN.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))
    data = {
        "train": {"count": len(train["samples"]), "sha256": sha256(TRAIN)},
        "validation": {"count": len(validation["samples"]), "sha256": sha256(VALIDATION)},
    }
    expected = {
        "train": "7a3094e0c1aeacf0edd7335f196af79eacd4b551cd9f78cce2cf7c903e910ca2",
        "validation": "861c24027bfddcfba031e253a6518343010e31d296fb66c010f8d88ee9a90497",
    }
    if {split: item["sha256"] for split, item in data.items()} != expected:
        raise ValueError("V1 manifest hashes differ from the preregistered B02 inputs")
    if (data["train"]["count"], data["validation"]["count"]) != (4959, 995):
        raise ValueError("V1 manifest sample counts differ from the preregistered B02 inputs")

    cells = ((1, 0.1), (1, 0.3), (2, 0.1), (2, 0.3))
    generated = []
    for radius, weight in cells:
        weight_id = str(weight).replace(".", "p")
        run_id = f"B02_r{radius}_lam{weight_id}_seed42"
        config = dict(base)
        config.update({
            "boundary": True,
            "boundary_radius": radius,
            "boundary_weight": weight,
            "ablation": "B",
            "b02_grid_id": "member3_B02_boundary_grid_v1",
        })
        protocol_path = OUTPUT / "protocols" / f"{run_id}.json"
        config_path = OUTPUT / f"{run_id}.json"
        config["frozen_protocol"] = protocol_path.relative_to(ROOT).as_posix()
        protocol_config = dict(config)
        protocol_config.pop("frozen_protocol")
        protocol_config.pop("frozen_protocol_sha256")
        protocol = {
            "schema_version": 1,
            "scope": "member3_B02_boundary_grid",
            "grid_id": "member3_B02_boundary_grid_v1",
            "cell": {"boundary_radius": radius, "boundary_weight": weight},
            "config": protocol_config,
            "data_protocol_id": "member1_final_20261029_v1",
            "data": data,
            "selection": {
                "metric": "validation all.foreground_iou",
                "tie_breaker": "validation all.boundary_iou",
                "final_tie": "r=1, lambda=0.1",
                "test_used_for_selection": False,
            },
            "comparability": "Use the same seed, initialization revision, software, budget, batch and accumulation in every cell and in A/D comparisons.",
        }
        write_json(protocol_path, protocol)
        config["frozen_protocol_sha256"] = sha256(protocol_path)
        write_json(config_path, config)
        generated.append({"run_id": run_id, "config": config_path.relative_to(ROOT).as_posix(),
                          "protocol": protocol_path.relative_to(ROOT).as_posix(),
                          "protocol_sha256": config["frozen_protocol_sha256"]})
    print(json.dumps({"grid_id": "member3_B02_boundary_grid_v1", "runs": generated}, indent=2))


if __name__ == "__main__":
    main()
