#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MAIE5421_Group24
PY=/root/autodl-tmp/maie5421-venv/bin/python
for run_id in B02_r1_lam0p1_seed42 B02_r1_lam0p3_seed42 B02_r2_lam0p1_seed42 B02_r2_lam0p3_seed42; do
  "$PY" -u -m transparent_seg.evaluate \
    --checkpoint "runs/member3_B02_cuda/${run_id}/best.pt" \
    --manifest manifests/member1_final_20261029/v1/validation.json \
    --output "runs/member3_B02_cuda/${run_id}/validation_evaluation" \
    --device cuda --progress-every 200 2>&1 | tee -a runs/member3_B02_cuda/validation_evaluation_console.log
done
