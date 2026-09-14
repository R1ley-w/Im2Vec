import pytest

from im2vec.frontier import (
    SWEEP,
    FrontierPoint,
    headroom,
    pareto_front,
    sweep_raster,
    tokens_at,
)
from im2vec.metrics import render_svg, rgb_over_white
from im2vec.pairs import synthesize
from im2vec.tracer import JPEG_PARAMS

from test_pairs import BUSY_SVG


def P(tokens, rmse, label=""):
    return FrontierPoint(label, tokens, rmse)


def test_pareto_front_keeps_only_non_dominated_points_sorted_by_tokens():
    points = [P(100, 10), P(50, 20), P(80, 12), P(90, 15), P(40, 30), P(60, 20)]
    front = pareto_front(points)
    assert [(p.tokens, p.rmse) for p in front] == [(40, 30), (50, 20), (80, 12), (100, 10)]


def test_pareto_front_breaks_exact_ties_once():
    front = pareto_front([P(50, 20, "a"), P(50, 20, "b")])
    assert len(front) == 1


def test_tokens_at_is_cheapest_point_meeting_the_fidelity():
    front = pareto_front([P(40, 30), P(50, 20), P(80, 12), P(100, 10)])
    assert tokens_at(front, 20) == 50
    assert tokens_at(front, 12.5) == 80
    assert tokens_at(front, 10) == 100
    assert tokens_at(front, 9.9) is None


def test_headroom_against_a_reference_that_beats_the_curve():
    front = pareto_front([P(40, 30), P(50, 20), P(80, 12), P(100, 10)])
    # Reference: 60 tokens at RMSE 12. The frontier needs 80 there.
    assert headroom(front, ref_tokens=60, ref_rmse=12) == pytest.approx(0.25)


def test_headroom_is_negative_when_the_reference_sits_outside_the_curve():
    front = pareto_front([P(40, 30), P(50, 20), P(80, 12)])
    assert headroom(front, ref_tokens=100, ref_rmse=12) < 0


def test_headroom_is_undefined_when_the_curve_never_reaches_that_fidelity():
    front = pareto_front([P(40, 30), P(50, 20)])
    assert headroom(front, ref_tokens=10, ref_rmse=5) is None


def test_sweep_includes_the_defaults_and_the_shipped_jpeg_branch():
    labels = {label: params for label, params in SWEEP}
    assert len(labels) == len(SWEEP)
    as_items = [sorted(p.items()) for p in labels.values()]
    defaults = {"filter_speckle": 4, "color_precision": 6, "layer_difference": 16}
    assert sorted(defaults.items()) in as_items
    shipped = {**defaults, **JPEG_PARAMS}
    assert sorted(shipped.items()) in as_items


def test_sweep_raster_reproduces_the_messy_trace_at_defaults():
    row = synthesize(BUSY_SVG, quality=20)
    reference = rgb_over_white(render_svg(BUSY_SVG))
    configs = [("defaults", {"filter_speckle": 4, "color_precision": 6, "layer_difference": 16}),
               ("sp16", {"filter_speckle": 16, "color_precision": 6, "layer_difference": 16})]
    points = sweep_raster(row["jpeg"], reference, configs)
    assert [p.label for p in points] == ["defaults", "sp16"]
    assert points[0].tokens == row["messy_tokens"]
    assert points[0].rmse == pytest.approx(row["messy_rmse"], abs=1e-3)
    assert points[1].tokens < points[0].tokens
