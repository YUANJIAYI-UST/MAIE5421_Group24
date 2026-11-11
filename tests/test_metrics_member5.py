"""E01: hand-calculated metric examples, independent of model/data downloads."""
import numpy as np
import pytest

from transparent_seg.metrics import SegmentationMetrics, boundary_iou, inner_boundary


def test_hand_counted_confusion_matrix():
    # TP=2, FP=1, FN=1, TN=2.
    meter = SegmentationMetrics()
    meter.update([[1, 1, 1], [0, 0, 0]], [[1, 1, 0], [1, 0, 0]])
    r = meter.compute()
    assert [r[k] for k in ('tp', 'fp', 'fn', 'tn')] == [2, 1, 1, 2]
    assert r['foreground_iou'] == pytest.approx(1 / 2)
    assert r['region_f1'] == pytest.approx(2 / 3)
    assert r['background_fpr'] == pytest.approx(1 / 3)


def test_region_pooled_but_boundary_image_mean():
    # Both shapes have boundary width 1. A 1x1 mask has 1 boundary pixel;
    # a full 3x3 mask has 8. One perfect image and one total miss.
    meter = SegmentationMetrics()
    meter.update(np.ones((1, 1)), np.ones((1, 1)))
    meter.update(np.zeros((3, 3)), np.ones((3, 3)))
    r = meter.compute()
    assert r['foreground_iou'] == pytest.approx(1 / 10)
    assert r['region_f1'] == pytest.approx(2 / 11)
    assert r['boundary_iou'] == pytest.approx(1 / 2)  # not pooled 1/9
    assert r['background_fpr'] is None


def test_inner_boundary_includes_image_border():
    expected = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], bool)
    np.testing.assert_array_equal(inner_boundary(np.ones((3, 3), bool), 1), expected)


def test_hand_counted_shifted_boundary():
    # Each 2x2 square is entirely boundary; overlap=2, union=6.
    y = np.zeros((5, 5), bool)
    p = np.zeros_like(y)
    y[1:3, 1:3] = True
    p[1:3, 2:4] = True
    assert boundary_iou(p, y) == pytest.approx(1 / 3)


def test_empty_and_false_positive_conventions():
    meter = SegmentationMetrics()
    meter.update(np.zeros((3, 3)), np.zeros((3, 3)))
    r = meter.compute()
    assert r['foreground_iou'] == r['region_f1'] == r['boundary_iou'] == 1
    assert r['background_fpr'] == 0
    p = np.zeros((3, 3))
    p[1, 1] = 1
    meter = SegmentationMetrics()
    meter.update(p, np.zeros_like(p))
    r = meter.compute()
    assert r['foreground_iou'] == r['region_f1'] == r['boundary_iou'] == 0
    assert r['background_fpr'] == pytest.approx(1 / 9)


def test_all_void_is_excluded_not_perfect():
    meter = SegmentationMetrics()
    meter.update(np.ones((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)))
    r = meter.compute()
    assert r['images'] == r['boundary_images'] == 0
    assert r['skipped_all_void'] == 1
    for key in ('foreground_iou', 'region_f1', 'boundary_iou', 'background_fpr'):
        assert r[key] is None


def test_void_changes_do_not_affect_scores():
    y = np.zeros((9, 9), bool)
    y[2:7, 2:7] = True
    valid = np.ones_like(y)
    valid[4, 4] = False
    p = y.copy()
    p[4, 4] = False
    a, b = SegmentationMetrics(), SegmentationMetrics()
    a.update(y, y, valid)
    b.update(p, y, valid)
    assert a.compute() == b.compute()


def test_valid_region_without_safe_boundary_pixels():
    valid = np.zeros((5, 5), bool)
    valid[2, 2] = True
    meter = SegmentationMetrics()
    meter.update(np.zeros_like(valid), np.zeros_like(valid), valid)
    r = meter.compute()
    assert r['images'] == 1
    assert r['foreground_iou'] == 1
    assert r['boundary_images'] == 0
    assert r['boundary_iou'] is None
