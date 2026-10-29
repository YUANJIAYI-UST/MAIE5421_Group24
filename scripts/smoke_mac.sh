#!/usr/bin/env bash
# macOS counterpart of smoke.ps1. Synthetic data verifies the pipeline only.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
RUN_DIR="${1:-runs/smoke_mac_$(TZ=Asia/Hong_Kong date +%Y%m%d_%H%M%S)}"

if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo "Missing .venv/bin/python. See docs/08_MAC_QUICKSTART.txt." >&2
    exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "Choose a new run directory; existing results are preserved: $RUN_DIR" >&2
    exit 1
fi
mkdir -p "$RUN_DIR"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"

run_step() {
    "$@" 2>&1 | tee -a "$RUN_DIR/console.log"
}

run_step "$PROJECT_PYTHON" -m pip check
run_step "$PROJECT_PYTHON" -m pytest -q --junitxml="$RUN_DIR/pytest.xml"
if [[ ! -f manifests/smoke/train.json ]]; then
    run_step "$PROJECT_PYTHON" scripts/make_smoke_data.py
fi
run_step "$PROJECT_PYTHON" -m transparent_seg.train \
    --config configs/smoke.json --output "$RUN_DIR/training" --device cpu
run_step "$PROJECT_PYTHON" -m transparent_seg.evaluate \
    --checkpoint "$RUN_DIR/training/best.pt" \
    --manifest manifests/smoke/test.json \
    --output "$RUN_DIR/evaluation" --robustness --device cpu
run_step "$PROJECT_PYTHON" -m transparent_seg.predict \
    --checkpoint "$RUN_DIR/training/best.pt" \
    --image data/smoke/test/easy/images/000.jpg \
    --output "$RUN_DIR/prediction" --device cpu

run_step echo "Passed. Results: $RUN_DIR"
run_step echo "Synthetic smoke only; these scores are not Trans10K benchmark results."
