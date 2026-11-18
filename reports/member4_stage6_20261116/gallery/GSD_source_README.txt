---
license: cc-by-nc-4.0
task_categories:
  - image-segmentation
tags:
  - glass-surface-detection
  - segmentation
  - reflection
  - computer-vision
pretty_name: Glass Surface Detection (GSD)
size_categories:
  - 1K<n<10K
---

# Glass Surface Detection (GSD) Dataset

Dataset from the CVPR 2021 paper:

> **Rich Context Aggregation with Reflection Prior for Glass Surface Detection**  
> Jiaying Lin, Zebang He, Rynson W.H. Lau  
> *Proceedings of CVPR 2021*  
> Project page: https://jiaying.link/cvpr2021-gsd/

## Dataset Summary

GSD is a large-scale benchmark for glass surface detection in the wild. The dataset contains three splits:

- **train** — 2,710 images with full annotations (mask, reflection, edge). This is the split used to train GlassNet.
- **extra** — 579 additional images with mask and edge annotations but no reflections. Not used to train GlassNet.
- **test** — 813 images with mask annotations for evaluation.

## Dataset Structure

| Split | Images | Masks | Reflections | Edges |
|-------|-------:|------:|------------:|------:|
| train | 2,710  | 2,710 | 2,710       | 2,710 |
| extra |   579  |   579 | —           |   579 |
| test  |   813  |   813 | —           | —     |

### Fields

- **image_id** — original filename stem (e.g. `glass_0001`), unique within each split
- **image** — RGB photograph containing glass surfaces
- **mask** — binary segmentation mask (white = glass)
- **reflections** — RGB reflection image paired with the scene (train only; `None` otherwise)
- **edge** — edge annotation map (train and extra only; `None` for test)

## Usage

```python
from datasets import load_dataset

ds = load_dataset("garrying/GSD")
sample = ds["train"][0]
sample["image_id"]     # original filename stem, e.g. "glass_0001"
sample["image"]        # PIL Image
sample["mask"]         # PIL Image (binary mask)
sample["reflections"]  # PIL Image (None for extra/test)
sample["edge"]         # PIL Image (None for test)
```

## Converting Back to Raw Files

A helper script [`parquet_to_raw.py`](parquet_to_raw.py) is included in this repository to convert the dataset back to a folder of raw image files:

```bash
# download the script
huggingface-cli download garrying/GSD parquet_to_raw.py --repo-type dataset --local-dir .

# convert all splits to raw PNG files
python parquet_to_raw.py --repo garrying/GSD --out GSD
```

Output layout:
```
GSD/
  train/  image/  mask/  reflections/  edge/  metadata.jsonl
  extra/  image/  mask/  edge/          metadata.jsonl
  test/   image/  mask/  metadata.jsonl
```

## Pretrained Model

A pretrained **GlassNet** checkpoint (`GSD.pth`) is available in the companion model repository:

👉 [garrying/GSD-GlassNet](https://huggingface.co/garrying/GSD-GlassNet)

### Inference

```bash
# download the checkpoint
huggingface-cli download garrying/GSD-GlassNet GSD.pth --local-dir .

# run inference
python infer.py
```

## Model Architecture

**GlassNet** uses a ResNeXt-101 backbone with:

- **DenseContrastModule** — multi-scale dilated convolutions (rates 1/2/4/8) with pairwise feature subtraction to capture cross-context contrast
- **SELayer** — grouped squeeze-and-excitation for context-aware channel reweighting
- **RefNet** — a lightweight U-Net-style decoder that jointly predicts the binary glass mask and reconstructs the reflection image as auxiliary output
- **CRF post-processing** — dense CRF refinement of predicted masks at inference time

## Citation

```bibtex
@inproceedings{GSD:2021,
    title     = {Rich Context Aggregation with Reflection Prior for Glass Surface Detection},
    author    = {Lin, Jiaying and He, Zebang and Lau, Rynson W.H.},
    booktitle = {Proc. CVPR},
    year      = {2021}
}
```

## Contact

jiayinlin5-c@my.cityu.edu.hk
