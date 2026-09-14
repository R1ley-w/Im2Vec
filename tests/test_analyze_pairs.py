import json

import pytest

from im2vec.data.analyze_pairs import (
    aggregate_curve,
    gate2,
    messiness_curve,
    per_image_headroom,
    run,
    summarise,
)
from im2vec.data.build_pairs import build_pairs

from test_pairs import BUSY_SVG


def _excess(key, q, **kw):
    base = dict(source_key=key, jpeg_quality=q, dataset="d", clean_tokens=60, clean_rmse=12.0,
                dropped_only_tokens=70, dropped_only_rmse=12.0, excess=10, dropped=5, merged=2,
                removable_fraction=0.7, messy_rmse=14.0, repaired_rmse=13.0)
    base.update(kw)
    return base


def _sweep(key, q, points):
    return [dict(source_key=key, jpeg_quality=q, dataset="d", label=l, tokens=t, rmse=r)
            for l, t, r in points]


CURVE = [("sp4_cp6_ld16", 100, 10.0), ("sp16_cp5_ld16", 80, 12.0), ("sp48_cp3_ld64", 40, 30.0)]


def test_gate2_pools_removals_capped_at_each_images_excess():
    excess = [
        _excess("a", 15, excess=10, dropped=6, merged=6, removable_fraction=1.0),  # capped at 10
        _excess("b", 15, excess=30, dropped=3, merged=0, removable_fraction=0.1),
        _excess("c", 15, excess=0, dropped=0, merged=0, removable_fraction=None),
    ]
    g = gate2(excess)
    assert g["images_with_excess"] == 2
    assert g["pooled_removable_fraction"] == pytest.approx(13 / 40)
    assert g["share_images_removable_above_threshold"] == pytest.approx(0.5)


def test_per_image_headroom_separates_unreachable_images():
    sweep = _sweep("a", 15, CURVE) + _sweep("b", 15, CURVE)
    excess = [
        _excess("a", 15, clean_tokens=60, clean_rmse=12.0),  # frontier needs 80 -> 25%
        _excess("b", 15, clean_tokens=60, clean_rmse=5.0),  # no setting reaches 5
    ]
    h = per_image_headroom(sweep, excess)
    assert h["clean"]["reachable"] == 1 and h["clean"]["unreachable"] == 1
    assert h["clean"]["headroom"]["median"] == pytest.approx(0.25)


def test_aggregate_curve_compares_clean_corner_with_named_settings():
    sweep = _sweep("a", 15, CURVE)
    curve = aggregate_curve(sweep, [_excess("a", 15, clean_tokens=60, clean_rmse=12.0)])
    assert curve["clean_corner"]["headroom"] == pytest.approx(0.25)
    assert curve["vs_shipped_jpeg_branch"]["clean_saves"] == pytest.approx(0.25)
    assert curve["vs_defaults"]["clean_saves"] == pytest.approx(0.4)
    assert [p["label"] for p in curve["front"]] == ["sp48_cp3_ld64", "sp16_cp5_ld16", "sp4_cp6_ld16"]


def test_run_and_summarise_end_to_end(tmp_path):
    root = tmp_path / "svg-emoji-hf"
    path = root / "valid" / "svg" / "a.svg"
    path.parent.mkdir(parents=True)
    path.write_text(BUSY_SVG)
    build_pairs([root], tmp_path / "pairs", ladder_size=1, workers=2)

    out = tmp_path / "analysis"
    result = run(tmp_path / "pairs", out, split="valid", workers=2)
    assert result == {"pairs": 5, "analysed": 5, "errors": 0}

    summary = summarise(out, tmp_path / "pairs")
    assert summary["pairs"] == 5
    assert set(summary["messiness_curve"]["all"]) == {"15", "30", "50", "75", "95"}
    assert set(summary["by_quality"]) == {"15", "30", "50", "75", "95"}
    json.loads((out / "summary.json").read_text())


def test_messiness_curve_groups_by_dataset_and_quality():
    row = dict(dataset="svg-emoji", clean_tokens=100, messy_tokens=400, clean_paths=5,
               messy_paths=50, clean_rmse=10.0, messy_rmse=12.0, jpeg_rmse=6.0)
    rows = [dict(row, jpeg_quality=15), dict(row, jpeg_quality=95, messy_tokens=150)]
    curve = messiness_curve(rows)
    assert curve["all"]["15"]["token_ratio"]["median"] == pytest.approx(4.0)
    assert curve["svg-emoji"]["95"]["token_ratio"]["median"] == pytest.approx(1.5)
    assert curve["all"]["15"]["path_ratio"]["median"] == pytest.approx(10.0)
