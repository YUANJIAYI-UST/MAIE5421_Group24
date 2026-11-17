import numpy as np
import pytest

from transparent_seg.metrics import CategoryRecall


def test_category_recall_pools_pixels_across_images():
    metric = CategoryRecall()

    # V2 类别标签：5 是玻璃门，8 是另一类透明物体，0 是背景。
    # 模型预测：True 表示预测为透明物体。
    metric.update(
        prediction=np.array([[True, False, True], [False, True, True]]),
        category_ids=np.array([[5, 5, 5], [8, 8, 0]]),
    )
    metric.update(
        prediction=np.array([[True, False]]),
        category_ids=np.array([[5, 0]]),
    )

    result = metric.compute()
    assert result[5]["recall"] == pytest.approx(3 / 4)
    assert result[8]["recall"] == pytest.approx(1 / 2)


def test_predict_in_v2_frame():
    import torch
    from transparent_seg.evaluate import predict_in_v2_frame

    logits = torch.tensor([[[[2., -2., -2.],
                             [-2., -2., -2.]]]])

    for orientation, shape, expected in [
        ("none",          (2, 3), [[1, 0, 0], [0, 0, 0]]),
        ("rotate_90_cw",  (3, 2), [[0, 0], [0, 0], [1, 0]]),
        ("rotate_90_ccw", (3, 2), [[0, 1], [0, 0], [0, 0]]),
    ]:
        actual = predict_in_v2_frame(logits, shape, orientation)
        np.testing.assert_array_equal(actual, expected)

    # 中间像素插值后的 logit 为 -1，所以预测为背景。
    scores = torch.tensor([[[[2., -4.]]]])
    actual = predict_in_v2_frame(scores, (1, 3), "none")
    np.testing.assert_array_equal(actual, [[1, 0, 0]])