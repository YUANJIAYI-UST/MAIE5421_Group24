"""Region BCE + per-image soft Dice + class-balanced boundary BCE."""
import torch
from torch.nn import functional as F


def boundary_targets(target, valid, radius=1):
    if radius not in (1, 2):
        raise ValueError("Proposal boundary radius must be 1 or 2")
    k = 2 * radius + 1
    # Zero outside the image for mask morphology. Outside-image validity is true;
    # only documented void pixels invalidate neighboring boundary supervision.
    dilated = F.max_pool2d(F.pad(target, (radius,) * 4, value=0), k, stride=1)
    eroded = -F.max_pool2d(-F.pad(target, (radius,) * 4, value=0), k, stride=1)
    safe = -F.max_pool2d(-F.pad(valid.float(), (radius,) * 4, value=1), k, stride=1)
    return (dilated - eroded).clamp(0, 1), safe.bool()


def masked_mean(x, valid):
    return (x * valid).sum() / valid.sum().clamp_min(1)


def balanced_bce(logits, target, valid):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    positive, negative = valid & (target > 0.5), valid & (target <= 0.5)
    if positive.any() and negative.any():
        return 0.5 * (loss[positive].mean() + loss[negative].mean())
    return masked_mean(loss, valid)


def segmentation_loss(outputs, target, valid, boundary_weight=0.1, radius=1):
    logits = F.interpolate(outputs["logits"], target.shape[-2:], mode="bilinear",
                           align_corners=False).float()
    bce = masked_mean(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), valid)
    p, y = logits.sigmoid() * valid, target * valid
    axes = (1, 2, 3)
    per_image = 1 - (2 * (p * y).sum(axes) + 1e-6) / (p.sum(axes) + y.sum(axes) + 1e-6)
    has_valid = valid.sum(axes) > 0
    dice = masked_mean(per_image, has_valid)
    boundary = logits.sum() * 0
    if "boundary_logits" in outputs:
        edges, edge_valid = boundary_targets(target, valid, radius)
        edge_logits = F.interpolate(outputs["boundary_logits"], target.shape[-2:],
                                    mode="bilinear", align_corners=False).float()
        boundary = balanced_bce(edge_logits, edges, edge_valid)
    return {"total": bce + dice + boundary_weight * boundary,
            "bce": bce, "dice": dice, "boundary": boundary}
