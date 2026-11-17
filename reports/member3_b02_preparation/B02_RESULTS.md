# B02 boundary grid results

**Owner:** YIN Qinyi (Member 3)
**Dataset release:** `member1_final_20261029_v1`
**Selection split:** official validation only; test data was not used.

## Frozen selection rule

Rank the four B02 cells by full-validation pooled foreground IoU (higher is better). A full-precision tie would be broken by Boundary IoU, then by the least added supervision (`r=1, lambda=0.1`). The four runs used seed 42, 512 input, batch 8, accumulation 1, 30 epochs / 18,600 optimizer steps, BF16, AdamW (`lr=6e-5`, `weight_decay=0.01`) and the pinned MiT-B0 revision `80983a413c30d36a39c20203974ae7807835e2b4`.

## Grid results

| `r` | `lambda` | Epochs | Best epoch | Best validation foreground IoU | Selected |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.1 | 30/30 | 30 | 0.8829718933 | |
| 1 | 0.3 | 30/30 | 29 | 0.8843500323 | |
| 2 | 0.1 | 30/30 | 30 | **0.8848859646** | Yes |
| 2 | 0.3 | 30/30 | 29 | 0.8843860765 | |

The selected pair is **`r=2, lambda=0.1`**. Its margin over the next cell is about 0.000500 IoU (0.05 percentage points); this is the outcome of the preregistered rule, not evidence of a large or statistically tested improvement.

## Selected checkpoint: full-resolution validation

Checkpoint: `runs/member3_B02_cuda/B02_r2_lam0p1_seed42/best.pt` (epoch 30). The final clean evaluation read all 995 official validation images at original resolution:

| Subset | Images | Foreground IoU | Region F1 | Boundary IoU |
|---|---:|---:|---:|---:|
| All | 995 | 0.8848859646 | 0.9389278515 | 0.6950515597 |
| Easy | 785 | 0.9247043231 | 0.9608793538 | 0.7365298216 |
| Hard | 210 | 0.7824811570 | 0.8779685035 | 0.5400018663 |

These are validation results, not test results. The validation manifest SHA256 is `861c24027bfddcfba031e253a6518343010e31d296fb66c010f8d88ee9a90497`; the training manifest SHA256 is `7a3094e0c1aeacf0edd7335f196af79eacd4b551cd9f78cce2cf7c903e910ca2`.

## Artifacts and provenance

Each of the four run folders contains `best.pt`, `last.pt`, `config.json`, all 30 `history.jsonl` epoch records, `environment.json`, and `pip_freeze.txt`. The selected checkpoint SHA256 is `4f2051ea15517828c65ac2a7493fb999cda3207f02efa49fe9aad5346489a9ae`.

The AutoDL source upload did not contain a `.git` directory, so the captured `environment.json` correctly records `git_commit: null`. After training, the exact source snapshot was committed as `64d437ff6d5bb70ee1a36240b37a07fa4bda8004`. All 31 non-AppleDouble source files from the original bundle were verified byte-for-byte against that commit. This is a post-run provenance commit, not a runtime-captured commit. See `source_provenance.json`.

The final GitHub delivery should keep code/configs/reports in this branch and put the model bundle on a GitHub Release asset. The archive SHA256 should be published beside the asset (for example, in the release notes and `.sha256` file), because a ZIP cannot contain its own final archive hash.

## Remaining project work

B02's four-cell selection and selected-model Boundary IoU are complete. The member plan's B03 also calls for training D with the selected pair and then analyzing B/D against A/C. The current frozen-protocol guard only authorizes the four B02 B cells; D needs its own pinned protocol/config and review before it can be labeled a formal research run. Do not use test data to revisit the selected pair.
