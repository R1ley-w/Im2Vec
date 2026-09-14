import numpy as np
import pytest

from im2vec.excess import (
    classify_excess,
    paint_layers,
    parse_trace,
    path_masks,
)
from im2vec.metrics import render_svg, rgb_over_white

HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">'
BG = '<path d="M0 0 L256 0 L256 256 L0 256 Z" fill="#FFFFFF"/>'


def rect(x, y, w, h, fill):
    return f'<path d="M{x} {y} L{x + w} {y} L{x + w} {y + h} L{x} {y + h} Z" fill="{fill}"/>'


def svg(*paths):
    return HEAD + "".join(paths) + "</svg>"


def reference(*paths):
    return rgb_over_white(render_svg(svg(*paths)))


def test_parse_trace_keeps_paint_order_and_fills():
    paths = parse_trace(svg(BG, rect(10, 10, 20, 20, "#FF0000"), rect(0, 0, 5, 5, "#00ff80")))
    assert [p.fill for p in paths] == [(255, 255, 255), (255, 0, 0), (0, 255, 128)]
    assert paths[1].d.startswith("M10 10")


def test_parse_trace_handles_vtracer_transform_attribute():
    trace = HEAD + '<path d="M0 0 L10 0 L10 10 L0 10 Z" fill="#123456" transform="translate(50,60)"/></svg>'
    (path,) = parse_trace(trace)
    masks = path_masks([path])
    ys, xs = np.nonzero(masks[0])
    assert xs.min() == 50 and ys.min() == 60


def test_paint_layers_reports_top_and_second_path():
    paths = parse_trace(svg(BG, rect(0, 0, 100, 100, "#FF0000"), rect(50, 50, 100, 100, "#0000FF")))
    top, second = paint_layers(path_masks(paths))
    assert top[75, 75] == 2 and second[75, 75] == 1
    assert top[25, 25] == 1 and second[25, 25] == 0
    assert top[200, 200] == 0 and second[200, 200] == -1


def test_speckle_over_a_uniform_region_is_droppable():
    ref = reference(rect(40, 40, 160, 160, "#FF0000"))
    clean = svg(BG, rect(40, 40, 160, 160, "#FF0000"))
    messy = svg(
        BG,
        rect(40, 40, 160, 160, "#FF0000"),
        rect(100, 100, 3, 3, "#3040F0"),
        rect(150, 70, 4, 2, "#20E020"),
    )
    report = classify_excess(messy, clean, ref)
    assert report.excess == 2
    assert report.dropped == 2 and report.merged == 0
    assert report.removable_fraction == pytest.approx(1.0)
    assert report.repaired_paths == 2
    assert report.repaired_rmse <= report.messy_rmse + 1e-6


def test_same_colour_fragment_is_mergeable_not_droppable():
    ref = reference(rect(40, 40, 160, 160, "#FF0000"))
    clean = svg(BG, rect(40, 40, 160, 160, "#FF0000"))
    # The square traced as two abutting pieces; dropping either exposes white.
    messy = svg(BG, rect(40, 40, 80, 160, "#FF0000"), rect(120, 40, 80, 160, "#FE0101"))
    report = classify_excess(messy, clean, ref)
    assert report.excess == 1
    assert report.merged == 1 and report.dropped == 0
    assert report.removable_fraction == pytest.approx(1.0)


def test_essential_fragment_is_counted_as_distorted():
    # Left half red, right half orange: a real two-colour feature.
    ref = reference(rect(40, 40, 160, 160, "#FF0000"), rect(120, 40, 80, 160, "#FF9900"))
    clean = svg(BG, rect(40, 40, 160, 160, "#FF0000"), rect(120, 40, 80, 160, "#FF9900"))
    # Messy splits the orange half into two pieces with a red seam between:
    # neither piece can go without exposing red, and no orange neighbour
    # touches either, so the extra path cannot be dropped or merged.
    messy = svg(
        BG,
        rect(40, 40, 160, 160, "#FF0000"),
        rect(120, 40, 80, 76, "#FF9900"),
        rect(120, 124, 80, 76, "#FF9900"),
    )
    report = classify_excess(messy, clean, ref)
    assert report.excess == 1
    assert report.dropped == 0 and report.merged == 0
    assert report.distorted == 1
    assert report.removable_fraction == 0.0


def test_fully_occluded_path_is_droppable():
    ref = reference(rect(40, 40, 160, 160, "#FF0000"))
    clean = svg(BG, rect(40, 40, 160, 160, "#FF0000"))
    messy = svg(BG, rect(60, 60, 20, 20, "#00FF00"), rect(40, 40, 160, 160, "#FF0000"))
    report = classify_excess(messy, clean, ref)
    assert report.dropped == 1


def test_no_excess_gives_no_fraction():
    ref = reference(rect(40, 40, 160, 160, "#FF0000"))
    trace = svg(BG, rect(40, 40, 160, 160, "#FF0000"))
    report = classify_excess(trace, trace, ref)
    assert report.excess == 0
    assert report.removable_fraction is None


def test_removals_never_exceed_what_fidelity_allows():
    # A speckle that *fixes* a real feature must stay: the reference has a
    # small blue dot, the clean trace missed it, the messy trace caught it.
    ref = reference(rect(40, 40, 160, 160, "#FF0000"), rect(100, 100, 6, 6, "#0000FF"))
    clean = svg(BG, rect(40, 40, 160, 160, "#FF0000"))
    messy = svg(BG, rect(40, 40, 160, 160, "#FF0000"), rect(100, 100, 6, 6, "#0000FF"))
    report = classify_excess(messy, clean, ref)
    assert report.dropped == 0 and report.merged == 0


def test_repaired_trace_is_a_valid_svg_with_the_reported_path_count():
    ref = reference(rect(40, 40, 160, 160, "#FF0000"))
    clean = svg(BG, rect(40, 40, 160, 160, "#FF0000"))
    messy = svg(BG, rect(40, 40, 160, 160, "#FF0000"), rect(100, 100, 3, 3, "#3040F0"))
    report = classify_excess(messy, clean, ref)
    assert len(parse_trace(report.dropped_only_svg)) == 2
    render_svg(report.repaired_svg)  # must render
