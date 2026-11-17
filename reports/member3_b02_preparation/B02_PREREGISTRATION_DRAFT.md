# B02 validation-grid preregistration — draft for group confirmation

**Owner:** YIN Qinyi (Member 3)
**Status:** Draft written before B02 runs; do not start the grid until the group confirms the execution device/slot and accepts this frozen protocol.

## Question

Choose the boundary supervision radius `r` and loss weight `lambda` for the B/D ablations using validation data only.

## Data and release

- Protocol: `member1_final_20261029_v1`
- Train manifest: `manifests/member1_final_20261029/v1/train.json`
  - SHA256: `7a3094e0c1aeacf0edd7335f196af79eacd4b551cd9f78cce2cf7c903e910ca2`
  - Effective samples: 4,959
- Validation manifest: `manifests/member1_final_20261029/v1/validation.json`
  - SHA256: `861c24027bfddcfba031e253a6518343010e31d296fb66c010f8d88ee9a90497`
  - Effective samples: 995
- `test.json` and `test.zip` are excluded from parameter selection and will not be read for this grid.

## Grid

Run these four cells in the listed order, one run at a time:

| Order | `r` | `lambda` | Run ID |
|---:|---:|---:|---|
| 1 | 1 | 0.1 | `B02_r1_lam0p1_seed42` |
| 2 | 1 | 0.3 | `B02_r1_lam0p3_seed42` |
| 3 | 2 | 0.1 | `B02_r2_lam0p1_seed42` |
| 4 | 2 | 0.3 | `B02_r2_lam0p3_seed42` |

## Frozen training controls

Use the member 2 full-A execution protocol unless the group explicitly freezes a replacement before any B02 run:

- Seed 42; ImageNet MiT-B0 revision `80983a413c30d36a39c20203974ae7807835e2b4`
- Input size 512; physical batch 8; gradient accumulation 1
- 30 epochs; 620 optimizer steps per epoch; 18,600 total optimizer steps
- AdamW, learning rate `6e-5`, weight decay `0.01`
- BF16 autocast; gradient clipping at 1.0
- 930 warmup optimizer steps; polynomial decay power 0.9
- Full 995-image validation at every epoch; choose each checkpoint by validation pooled foreground IoU
- Initialize each grid cell from the same shared initial state as A; do not initialize B runs from A's trained `best.pt`
- Fresh run directory for every cell; preserve config, environment, data/cache hashes, history, checkpoint hashes, per-image validation results and visual evidence

## Selection rule

1. Rank the four runs by best-checkpoint validation `all.foreground_iou` (higher is better).
2. If tied at the reported full precision, use validation `all.boundary_iou` (higher is better).
3. If both are still tied, choose the least added supervision: `r=1, lambda=0.1`.
4. Record the selected pair and freeze it for both B and D. Do not inspect test results to choose or revise it.

## Results table (fill after runs)

| `r` | `lambda` | Best epoch | Val foreground IoU | Val Boundary IoU | Optimizer steps | Runtime | Peak GPU memory | Initial shared hash | Run/checkpoint SHA256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 0.1 | pending | pending | pending | pending | pending | pending | pending | pending |
| 1 | 0.3 | pending | pending | pending | pending | pending | pending | pending | pending |
| 2 | 0.1 | pending | pending | pending | pending | pending | pending | pending | pending |
| 2 | 0.3 | pending | pending | pending | pending | pending | pending | pending | pending |

## Before execution

- Confirm with WANG Zenan who will run the grid and reserve the RTX 5070 Ti slot; this Mac's current PyTorch environment reports MPS/CUDA unavailable.
- Verify the merged main configuration and current data release hashes in the actual repository checkout.
- Confirm B/D use batch 8 and accumulation 1, or freeze a different common setting before any run. Do not change controls between cells.
- Keep PR #2's complete A baseline as the reference; its test result is not part of this selection rule.
