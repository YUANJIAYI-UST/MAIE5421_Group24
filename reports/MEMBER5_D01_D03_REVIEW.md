# Member 5 — D01–D03 Independent Review

Reviewer: ZHANG Xinyu
Status: Partial — source image and mask verification pending.

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

## Evidence

- member5_structural_verification_lf.json
- member5_data_tests.xml

## Remaining checks

- Verify actual V1 source images, masks, and repaired copies.
- Verify D03 filtering against actual V2 category masks.
- Review the human decision records and supporting image evidence.
- Align the verification environment with the pinned requirements
  before full software validation. Current checks used
  torch 2.15.0; requirements.txt specifies torch 2.7.1.
