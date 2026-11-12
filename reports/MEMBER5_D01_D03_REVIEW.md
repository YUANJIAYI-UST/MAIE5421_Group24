# Member 5 — D01–D03 Independent Review

Reviewer: ZHANG Xinyu
Status: Automated verification passed after the LF fix.
Human spot-check feedback recorded;Pair 141 has a separately documented follow-up review.

## Completed checks

- Verified all 240 files listed in the delivery package's
  CHECKSUMS.json: no missing files or hash mismatches.
- Reproduced the final manifests from the frozen inputs.
- After the LF newline fix described below, all 22 generated files
  were byte-identical to the delivered release.
- Structural verification passed:
  - 28 pinned inputs unchanged.
  - 66 duplicate components checked.
  - No confirmed same-source components spanning retained splits.
  - 46 unique exclusions: 41 training and 5 validation images.
  - Frozen class selection and official test subsets unchanged.
  - Holdout filtering and category counts verified against the
    packaged metadata.
- Delivery package data tests: 29 passed.
- Repository data and E01 metric tests: 37 passed.

## Issue found and fix

The original JSON writer did not explicitly set newline handling.
On Windows, generated JSON files used CRLF instead of LF, causing
byte comparisons and dependent file hashes to differ.

Added newline="\n" to write_json() in transparent_seg/prepare.py.
Regenerated the manifests in a separate directory and reran
structural verification successfully.

The unmodified delivery did not pass byte-identical reproduction
on Windows; reproduction passed after this fix.

## Real-data verification and full tests

- V1 verification passed: all 20,856 original image and mask files
  matched the recorded hashes.
- Verified 20,866 distinct original and effective files in total,
  including the 10 retained corrected masks.
- Official test membership and binary labels were unchanged.
- Final V1 counts: train 4,959; validation 995; test 4,428.
- Real-data loading and a CPU forward/backward diagnostic passed.
  This was not a formal training experiment.
- V2 import and alignment audits reported no issues for 5,000
  training, 1,000 validation, and 4,428 test images.
- D03 verification against actual category masks passed:
  train 3,230; validation 646; test 4,428.
- Held-out classes 5 and 8 had zero annotated pixels in D03
  training and validation.
- Test held-out (1,493) and seen (2,935) subsets were disjoint
  and together covered the official test set.
- V1 binary fields and current mask hashes were preserved.
- All 11 D03 release files were reproduced byte-for-byte.
- After installing the pinned requirements, the complete repository
  test suite passed: 69 tests.
- The V1 verification recorded Python 3.12.0 and torch 2.7.1+cpu.
  Earlier targeted checks used torch 2.15.0.

## Human spot-check feedback

ZHANG Xinyu reported manually viewing some candidate images in
runs/trans10k_member1_20261028/review_gallery and finding no
obvious issues in the inspected samples.

The human_review fields in automated verification outputs are
inherited delivery metadata, not evidence of member 5 sign-off.

Traceable follow-up review: ZHANG Xinyu inspected pair 141
(near_within_contact_11.jpg, displayed row 061):
test/easy/images/6755.jpg and test/easy/images/8481.jpg.

Member 5 classified this pair as C: different images/scenes.
The owner's original classification was D: uncertain.
Both images remain in the official test set; this clarification
does not require changes to the data manifests.

## Evidence

- member5_structural_verification_lf.json
- member5_data_tests.xml
- member5_v1_verification.json
- member5_d03_verification.json
- member5_full_tests.xml

V2 alignment audit inventories and issue reports remain in the
local delivery directory:
- runs/member5_v2_train_val_audit/
- runs/member5_v2_test_audit/

## Scope and limitations

Automated reproducibility and real-data checks passed after the
documented LF fix. Human review evidence is limited to the
unindexed spot-check feedback above.

Equivalence between the pinned third-party V2 mirror and the
original author's archive remains unverified.

Same-scene photographs and within-split duplicates may remain
under the delivery policy; scene-disjoint evaluation is not claimed.
The owner recorded pair 141 as D (uncertain); member 5 subsequently
classified it as C (different images/scenes). The original owner
record is preserved, with the follow-up documented above.

Formal GPU training and model-performance experiments are outside
this data-delivery review.
