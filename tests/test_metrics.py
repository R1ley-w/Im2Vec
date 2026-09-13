import numpy as np
import pytest

from im2vec.metrics import (
    foreground_mask,
    iou,
    l1,
    render_svg,
    rgb_over_white,
    rmse,
    ssim,
)

SQUARE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
    '<rect x="64" y="64" width="128" height="128" fill="#ff0000"/></svg>'
)
EMPTY_SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256"></svg>'


def test_render_svg_returns_rgba_at_canvas_size():
    img = render_svg(SQUARE_SVG)
    assert img.mode == "RGBA"
    assert img.size == (256, 256)


def test_rgb_over_white_is_unit_range_rgb():
    arr = rgb_over_white(render_svg(SQUARE_SVG))
    assert arr.shape == (256, 256, 3)
    assert arr.min() >= 0.0 and arr.max() <= 1.0
    # transparent corner composites to white; the square centre stays red
    assert np.allclose(arr[0, 0], [1.0, 1.0, 1.0], atol=1e-3)
    assert arr[128, 128, 0] > 0.9 and arr[128, 128, 1] < 0.1


def test_identical_images_score_perfectly():
    a = rgb_over_white(render_svg(SQUARE_SVG))
    assert l1(a, a) == pytest.approx(0.0, abs=1e-6)
    assert rmse(a, a) == pytest.approx(0.0, abs=1e-6)
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-4)


def test_metrics_degrade_on_different_images():
    a = rgb_over_white(render_svg(SQUARE_SVG))
    b = rgb_over_white(render_svg(EMPTY_SVG))
    assert l1(a, b) > 0.0
    assert rmse(a, b) > 0.0
    assert ssim(a, b) < 1.0


def test_rmse_is_scale_correct():
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.full((4, 4, 3), 0.5, dtype=np.float32)
    assert rmse(a, b) == pytest.approx(0.5)
    assert l1(a, b) == pytest.approx(0.5)


def test_rmse_penalises_outliers_more_than_l1():
    a = np.zeros((10, 10, 3), dtype=np.float32)
    b = a.copy()
    b[0, 0, :] = 1.0  # one bright outlier pixel
    assert rmse(a, b) > l1(a, b)


def test_iou_of_identical_masks_is_one():
    m = foreground_mask(render_svg(SQUARE_SVG))
    assert iou(m, m) == pytest.approx(1.0)


def test_iou_of_disjoint_masks_is_zero():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[:4] = True
    b[4:] = True
    assert iou(a, b) == 0.0


def test_iou_of_two_empty_masks_is_zero_not_nan():
    empty = np.zeros((8, 8), dtype=bool)
    assert iou(empty, empty) == 0.0


def test_iou_half_overlap():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((4, 4), dtype=bool)
    a[:, :2] = True
    b[:, 1:3] = True
    # intersection 4 px, union 12 px
    assert iou(a, b) == pytest.approx(4 / 12)


def test_foreground_mask_marks_only_opaque_pixels():
    m = foreground_mask(render_svg(SQUARE_SVG))
    assert m.dtype == bool
    assert not m[0, 0]
    assert m[128, 128]


def test_ssim_is_symmetric():
    a = rgb_over_white(render_svg(SQUARE_SVG))
    b = rgb_over_white(render_svg(EMPTY_SVG))
    assert ssim(a, b) == pytest.approx(ssim(b, a), abs=1e-6)


def test_metrics_do_not_require_torch():
    import sys

    assert "torch" not in sys.modules
