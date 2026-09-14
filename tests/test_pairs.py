import io
from collections import Counter

import pytest
from PIL import Image

from im2vec.pairs import (
    LADDER,
    QUALITY_RANGE,
    RenderError,
    Source,
    assign_qualities,
    discover_sources,
    synthesize,
)
from im2vec.tracer import DEFAULT_PARAMS, trace_svg

# Antialiased curves in several colours. A flat-colour rectangle is a trap:
# JPEG reproduces it almost perfectly, so it never exercises speckle.
BUSY_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256">'
    '<circle cx="90" cy="100" r="70" fill="#e53935"/>'
    '<circle cx="160" cy="130" r="60" fill="#1e88e5" fill-opacity="0.8"/>'
    '<circle cx="120" cy="180" r="45" fill="#fdd835"/>'
    '<path d="M20 230 Q128 20 236 230" stroke="#2e7d32" stroke-width="9" fill="none"/>'
    '<rect x="30" y="30" width="40" height="12" fill="#6a1b9a" transform="rotate(20 50 36)"/>'
    "</svg>"
)


def _write(path, text="<svg/>"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# -- discovery ---------------------------------------------------------------


def test_discover_sources_handles_flat_and_sharded_layouts(tmp_path):
    _write(tmp_path / "svg-emoji-hf/train/svg/a.svg")
    _write(tmp_path / "svg-emoji-hf/valid/svg/b.svg")
    _write(tmp_path / "svg-stack-hf/train/svg/shard00/c.svg")
    _write(tmp_path / "svg-stack-hf/train/svg/shard03/d.svg")
    _write(tmp_path / "svg-stack-hf/test/svg/e.svg")
    # PNGs sitting alongside must not be picked up as sources.
    _write(tmp_path / "svg-emoji-hf/train/png/a.png")

    sources = discover_sources(
        [tmp_path / "svg-emoji-hf", tmp_path / "svg-stack-hf"]
    )

    got = {(s.dataset, s.split, s.source_id) for s in sources}
    assert got == {
        ("svg-emoji", "train", "a"),
        ("svg-emoji", "valid", "b"),
        ("svg-stack", "train", "c"),
        ("svg-stack", "train", "d"),
        ("svg-stack", "test", "e"),
    }


def test_discover_sources_is_sorted_and_records_size(tmp_path):
    _write(tmp_path / "svg-emoji-hf/train/svg/z.svg", "x" * 10)
    _write(tmp_path / "svg-emoji-hf/train/svg/a.svg", "x" * 3)
    sources = discover_sources([tmp_path / "svg-emoji-hf"])
    assert [s.source_id for s in sources] == ["a", "z"]
    assert [s.size_bytes for s in sources] == [3, 10]


# -- quality assignment -----------------------------------------------------


def _fake_sources(n_emoji=400, n_stack=1600):
    out = []
    for i in range(n_emoji):
        out.append(Source("svg-emoji", ("train", "valid", "test")[i % 3], f"e{i}", None, 100 + i))
    for i in range(n_stack):
        out.append(Source("svg-stack", ("train", "valid", "test")[i % 3], f"s{i}", None, 50 + 7 * i))
    return out


def test_ladder_subset_has_exact_size_and_full_ladder():
    plan = assign_qualities(_fake_sources(), ladder_size=500, seed=0)
    ladder = [p for p in plan if p.subset == "ladder"]
    assert len({p.source.key for p in ladder}) == 500
    per_source = Counter(p.source.key for p in ladder)
    assert set(per_source.values()) == {len(LADDER)}
    assert {p.quality for p in ladder} == set(LADDER)


def test_rest_get_one_random_quality_in_range():
    sources = _fake_sources()
    plan = assign_qualities(sources, ladder_size=500, seed=0)
    rest = [p for p in plan if p.subset == "random"]
    assert len(rest) == len(sources) - 500
    assert len({p.source.key for p in rest}) == len(rest)
    lo, hi = QUALITY_RANGE
    assert all(lo <= p.quality <= hi for p in rest)
    # Uses the whole range rather than collapsing onto a few values.
    assert len({p.quality for p in rest}) > 50


def test_every_source_is_planned_exactly_once_per_subset():
    sources = _fake_sources()
    plan = assign_qualities(sources, ladder_size=500, seed=0)
    subsets = {}
    for p in plan:
        subsets.setdefault(p.source.key, set()).add(p.subset)
    assert len(subsets) == len(sources)
    assert all(len(v) == 1 for v in subsets.values())


def test_ladder_is_stratified_by_dataset_and_split():
    sources = _fake_sources()
    plan = assign_qualities(sources, ladder_size=500, seed=0)
    ladder = {p.source for p in plan if p.subset == "ladder"}
    by_stratum = Counter((s.dataset, s.split) for s in ladder)
    totals = Counter((s.dataset, s.split) for s in sources)
    for stratum, total in totals.items():
        expected = 500 * total / len(sources)
        assert abs(by_stratum[stratum] - expected) <= 5, stratum


def test_ladder_spans_source_complexity():
    sources = _fake_sources()
    plan = assign_qualities(sources, ladder_size=500, seed=0)
    stack = sorted(
        (s for s in sources if s.dataset == "svg-stack"), key=lambda s: s.size_bytes
    )
    smallest = {s.key for s in stack[: len(stack) // 4]}
    largest = {s.key for s in stack[-len(stack) // 4 :]}
    ladder = {p.source.key for p in plan if p.subset == "ladder"}
    assert len(ladder & smallest) > 50
    assert len(ladder & largest) > 50


def test_assignment_is_deterministic_and_seeded():
    sources = _fake_sources()
    a = assign_qualities(sources, ladder_size=500, seed=0)
    b = assign_qualities(list(reversed(sources)), ladder_size=500, seed=0)
    c = assign_qualities(sources, ladder_size=500, seed=1)
    key = lambda plan: sorted((p.source.key, p.subset, p.quality) for p in plan)
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_ladder_larger_than_population_takes_everything():
    sources = _fake_sources(10, 10)
    plan = assign_qualities(sources, ladder_size=500, seed=0)
    assert {p.subset for p in plan} == {"ladder"}
    assert len(plan) == 20 * len(LADDER)


# -- synthesis --------------------------------------------------------------


def test_synthesize_produces_decodable_rasters_and_traces():
    row = synthesize(BUSY_SVG, quality=30)
    assert Image.open(io.BytesIO(row["clean_png"])).format == "PNG"
    jpeg = Image.open(io.BytesIO(row["jpeg"]))
    assert jpeg.format == "JPEG" and jpeg.size == (256, 256)
    assert row["clean_trace"].startswith("<svg")
    assert row["messy_trace"].startswith("<svg")
    assert row["jpeg_quality"] == 30


def test_synthesize_traces_both_inputs_with_identical_default_params():
    # The format branch would give the JPEG speckle filtering, hiding exactly
    # the tracer noise the pairs exist to capture.
    row = synthesize(BUSY_SVG, quality=30)
    assert row["messy_trace"] == trace_svg(row["jpeg"], params=DEFAULT_PARAMS)
    assert row["clean_trace"] == trace_svg(row["clean_png"], params=DEFAULT_PARAMS)


def test_low_quality_jpeg_yields_a_messy_trace():
    row = synthesize(BUSY_SVG, quality=15)
    assert row["messy_paths"] > row["clean_paths"]
    assert row["messy_tokens"] > row["clean_tokens"]


def test_synthesize_reports_fidelity_against_source_raster():
    row = synthesize(BUSY_SVG, quality=15)
    for key in ("clean_rmse", "messy_rmse", "jpeg_rmse"):
        assert 0.0 <= row[key] < 255.0
    assert row["jpeg_rmse"] > 0.0
    assert 0.0 < row["source_fg_fraction"] <= 1.0


def test_higher_quality_damages_the_raster_less():
    low = synthesize(BUSY_SVG, quality=15)
    high = synthesize(BUSY_SVG, quality=95)
    assert high["jpeg_rmse"] < low["jpeg_rmse"]
    assert high["messy_tokens"] < low["messy_tokens"]


@pytest.mark.parametrize("bad", ["", "not xml at all", "<svg><path d='M0 0 L"])
def test_unrenderable_source_raises_render_error(bad):
    with pytest.raises(RenderError):
        synthesize(bad, quality=50)
