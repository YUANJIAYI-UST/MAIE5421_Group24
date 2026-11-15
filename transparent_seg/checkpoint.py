"""Epoch-boundary checkpoints, readable with torch.load(weights_only=True)."""
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng(generator):
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    npstate = state["numpy"]
    np.random.set_state((npstate[0], np.asarray(npstate[1], dtype=np.uint32), *npstate[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device topology")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def validate_resume(checkpoint, config, manifests):
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("Checkpoint lacks resumable optimizer/scheduler/RNG state")
    # Do not silently stretch the original learning-rate schedule or change data.
    if checkpoint["config"] != config:
        raise ValueError("Resume config must exactly match the original frozen run config")
    if checkpoint["manifest_sha256"] != manifests:
        raise ValueError("Resume manifest hashes changed")
    if checkpoint["torch_version"] != str(torch.__version__):
        raise ValueError("Resume requires the original PyTorch version")
