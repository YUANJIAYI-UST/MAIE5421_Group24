import numpy as np
import pytest
import torch
from PIL import Image
from scipy.ndimage import binary_erosion

from transparent_seg.data import decode_mask, perturb, PERTURBATIONS
from transparent_seg.losses import boundary_targets, segmentation_loss, balanced_bce
from transparent_seg.metrics import SegmentationMetrics, inner_boundary, boundary_iou
from transparent_seg.model import TransparentSegformer
from transparent_seg.train import load_config, seed_everything, shared_parameter_hash

torch.set_num_threads(2)


def test_rgb_mask_mean_not_luminance(tmp_path):
    arr = np.array([[[0, 0, 0], [255, 0, 0], [85, 85, 85], [255, 255, 255]]], dtype=np.uint8)
    path = tmp_path / "mask.png"
    Image.fromarray(arr).save(path)
    y, valid = decode_mask(path)
    assert y.tolist() == [[0, 1, 1, 1]]
    assert valid.all()  # White 255 remains foreground!


def test_unknown_mask_fails(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.full((2, 2, 3), 128, np.uint8)).save(path)
    with pytest.raises(ValueError, match="Unexpected"):
        decode_mask(path)
    y, valid = decode_mask(path, void_values=[128])
    assert not valid.any() and not y.any()
    with pytest.raises(ValueError, match="collide"):
        decode_mask(path, void_values=[255])


def test_binary_mask_is_explicit(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.array([[0, 255]], np.uint8)).save(path)
    assert decode_mask(path, "binary_0255")[0].tolist() == [[0, 1]]


@pytest.mark.parametrize("radius", [1, 2])
def test_boundary_void_neighborhood(radius):
    y = torch.zeros(1, 1, 12, 12)
    y[:, :, 3:9, 3:9] = 1
    valid = torch.ones_like(y, dtype=torch.bool)
    valid[:, :, 3, 3] = False
    edge, safe = boundary_targets(y, valid, radius)
    assert edge[0, 0, 3, 3] == 1
    assert not safe[0, 0, 3-radius:4+radius, 3-radius:4+radius].any()
    assert safe[0, 0, 10, 10]


def test_no_edge_degenerate_bce_learns():
    x = torch.zeros(1, 1, 4, 4, requires_grad=True)
    loss = balanced_bce(x, torch.zeros_like(x), torch.ones_like(x, dtype=torch.bool))
    loss.backward()
    assert loss.item() > 0 and x.grad.abs().sum() > 0


def test_all_void_zero_loss_and_grad():
    x = torch.zeros(1, 1, 8, 8, requires_grad=True)
    losses = segmentation_loss({"logits": x, "boundary_logits": x}, torch.ones_like(x),
                                torch.zeros_like(x, dtype=torch.bool))
    losses["total"].backward()
    assert losses["total"].item() == 0 and x.grad.abs().sum().item() == 0


@pytest.mark.parametrize("shape", [(20, 30), (80, 100), (300, 400)])
def test_inner_boundary_matches_iterated_official_morphology(shape):
    rng = np.random.default_rng(42)
    mask = rng.random(shape) > 0.1
    width = max(1, round(0.02 * np.hypot(*shape)))
    reference = mask & ~binary_erosion(mask, structure=np.ones((3, 3)), iterations=width, border_value=0)
    np.testing.assert_array_equal(inner_boundary(mask, width), reference)
    assert inner_boundary(np.ones(shape, bool), width)[0].all()


def test_metrics_perfect_empty_and_shifted():
    y = np.zeros((30, 40), bool)
    y[5:20, 5:30] = True
    assert boundary_iou(y, y) == 1
    assert boundary_iou(y, np.roll(y, 3, axis=1)) < 1
    meter = SegmentationMetrics()
    meter.update(y, y)
    result = meter.compute()
    assert result["foreground_iou"] == result["region_f1"] == result["boundary_iou"] == 1
    empty = SegmentationMetrics()
    empty.update(np.zeros_like(y), np.zeros_like(y))
    assert empty.compute()["foreground_iou"] == 1


def test_iou_is_dataset_pooled():
    meter = SegmentationMetrics()
    meter.update(np.ones((1, 1)), np.ones((1, 1)))
    meter.update(np.zeros((10, 10)), np.ones((10, 10)))
    assert meter.compute()["foreground_iou"] == pytest.approx(1 / 101)
    assert meter.compute()["region_f1"] == pytest.approx(2 / 102)


def test_invalid_pixels_do_not_change_region_scores():
    meter = SegmentationMetrics()
    meter.update(np.array([[1, 1]]), np.array([[1, 0]]), np.array([[1, 0]]))
    assert meter.compute()["foreground_iou"] == 1


@pytest.mark.parametrize("op,value", PERTURBATIONS)
def test_photometric_range_and_shape(op, value):
    x = np.random.default_rng(0).random((17, 23, 3)).astype(np.float32)
    out = perturb(x, op, value)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    np.testing.assert_allclose(perturb(x, op, 1), x, atol=1e-7)


def test_shared_initialization_and_auxiliary_gradients():
    seed_everything(42)
    baseline = TransparentSegformer(boundary=False, pretrained=False)
    baseline_hash = shared_parameter_hash(baseline)
    seed_everything(42)
    combined = TransparentSegformer(boundary=True, pretrained=False)
    assert shared_parameter_hash(combined) == baseline_hash
    combined.train()
    x = torch.randn(2, 3, 64, 64)
    y = torch.zeros(2, 1, 64, 64)
    y[:, :, 10:40, 10:40] = 1
    outputs = combined(x, return_boundary=True)
    segmentation_loss(outputs, y, torch.ones_like(y, dtype=torch.bool))["total"].backward()
    assert combined.boundary[-1].weight.grad.abs().sum() > 0
    assert next(combined.encoder.parameters()).grad.abs().sum() > 0
    combined.eval()
    assert "boundary_logits" not in combined(x)


def test_ablation_configs_match_budget():
    configs = [load_config(f"configs/{name}.json") for name in
               ("A_baseline", "B_boundary", "C_augmentation", "D_combined")]
    assert [(c["boundary"], c["augmentation"]) for c in configs] == [(False, False), (True, False), (False, True), (True, True)]
    for key in ("seed", "size", "batch_size", "epochs", "lr", "encoder_revision", "boundary_radius", "boundary_weight"):
        assert len({c[key] for c in configs}) == 1
