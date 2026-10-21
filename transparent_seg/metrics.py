"""Dataset-pooled foreground metrics and image-mean inner Boundary IoU.

Boundary morphology follows Cheng et al. (CVPR 2021), including image borders.
Empty/empty mask union scores 1; all-void images are excluded explicitly.
"""
import math
import numpy as np
from scipy.ndimage import minimum_filter


def inner_boundary(mask, width):
    mask = np.asarray(mask, dtype=bool)
    eroded = minimum_filter(mask, size=2 * width + 1, mode="constant", cval=0)
    return mask & ~eroded


def boundary_iou(prediction, target, valid=None):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError("Boundary IoU expects equal HxW masks")
    valid = np.ones_like(target) if valid is None else np.asarray(valid, bool)
    width = max(1, int(round(0.02 * math.hypot(*target.shape))))
    safe = minimum_filter(valid, size=2 * width + 1, mode="constant", cval=1)
    if not safe.any():
        return None
    p, y = inner_boundary(prediction & valid, width) & safe, inner_boundary(target & valid, width) & safe
    union = np.count_nonzero(p | y)
    return float(np.count_nonzero(p & y) / union) if union else 1.0


class SegmentationMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.boundaries = []
        self.images = self.skipped = 0

    def update(self, prediction, target, valid=None):
        p, y = np.asarray(prediction, bool), np.asarray(target, bool)
        v = np.ones_like(y) if valid is None else np.asarray(valid, bool)
        if p.shape != y.shape or v.shape != y.shape or y.ndim != 2:
            raise ValueError("Expected equal HxW masks")
        if not v.any():
            self.skipped += 1
            return
        self.images += 1
        self.tp += int(np.count_nonzero(p & y & v))
        self.fp += int(np.count_nonzero(p & ~y & v))
        self.fn += int(np.count_nonzero(~p & y & v))
        self.tn += int(np.count_nonzero(~p & ~y & v))
        score = boundary_iou(p, y, v)
        if score is not None:
            self.boundaries.append(score)

    def compute(self):
        union, denom = self.tp + self.fp + self.fn, 2 * self.tp + self.fp + self.fn
        return {
            "images": self.images, "skipped_all_void": self.skipped,
            "foreground_iou": (self.tp / union if union else 1.0) if self.images else None,
            "region_f1": (2 * self.tp / denom if denom else 1.0) if self.images else None,
            "boundary_iou": float(np.mean(self.boundaries)) if self.boundaries else None,
            "boundary_images": len(self.boundaries),
            "background_fpr": self.fp / (self.fp + self.tn) if self.fp + self.tn else None,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
        }
