#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MAIE5421_Group24
export HF_ENDPOINT=https://hf-mirror.com
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/root/autodl-tmp/maie5421-venv/bin/python
mkdir -p runs/member3_B02_cuda
for run_id in B02_r1_lam0p1_seed42 B02_r1_lam0p3_seed42 B02_r2_lam0p1_seed42 B02_r2_lam0p3_seed42; do
  output="runs/member3_B02_cuda/${run_id}"
  if [[ -e "$output" ]]; then
    echo "Refusing existing output: $output" >&2
    exit 2
  fi
  echo "Starting $run_id on CUDA RTX 4090"
  "$PY" -u -m transparent_seg.train --config "configs/member3_b02/${run_id}.json" --output "$output" --device cuda 2>&1 | tee -a runs/member3_B02_cuda/training_console.log
done
