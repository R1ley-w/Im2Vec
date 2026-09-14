import json

import pyarrow.parquet as pq
import pytest

from im2vec.data.build_pairs import build_pairs, export_loose, read_pairs, spread_order
from im2vec.pairs import LADDER

from test_pairs import BUSY_SVG


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "svg-emoji-hf"
    for split, name, text in [
        ("train", "a", BUSY_SVG),
        ("train", "b", BUSY_SVG.replace("#e53935", "#00897b")),
        ("valid", "c", BUSY_SVG.replace("#1e88e5", "#f4511e")),
        ("valid", "broken", "<svg><path d='M0 0 L"),
    ]:
        path = root / split / "svg" / f"{name}.svg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _build(tree, out, **kw):
    kw.setdefault("ladder_size", 1)
    kw.setdefault("workers", 2)
    return build_pairs([tree], out, **kw)


def test_build_writes_one_row_per_planned_pair(tree, tmp_path):
    out = tmp_path / "pairs"
    summary = _build(tree, out)

    table = read_pairs(out)
    # 4 sources: 1 on the ladder, 3 random; the broken one fails.
    ladder_rows = len(LADDER)
    assert summary["planned"] == ladder_rows + 3
    assert table.num_rows + summary["failed"] == summary["planned"]
    assert summary["failed"] in (1, len(LADDER))  # broken may be the ladder pick
    keys = table.column("source_key").to_pylist()
    assert not any(k.endswith("/broken") for k in keys)
    assert set(table.column("split").to_pylist()) <= {"train", "valid"}


def test_rows_carry_provenance_and_source(tree, tmp_path):
    out = tmp_path / "pairs"
    _build(tree, out)
    row = read_pairs(out).slice(0, 1).to_pylist()[0]
    assert row["dataset"] == "svg-emoji"
    assert row["subset"] in ("ladder", "random")
    assert row["source_svg"].startswith("<svg")
    assert isinstance(row["jpeg"], bytes)
    assert row["source_key"] == f"{row['dataset']}/{row['split']}/{row['source_id']}"


def test_failures_are_logged_with_a_reason(tree, tmp_path):
    out = tmp_path / "pairs"
    _build(tree, out)
    lines = (out / "failures.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert records and all(r["source_key"].endswith("/broken") for r in records)
    assert all(r["reason"] for r in records)


def test_rebuild_resumes_without_duplicates(tree, tmp_path):
    out = tmp_path / "pairs"
    first = _build(tree, out)
    second = _build(tree, out)
    assert second["skipped"] == first["planned"]
    assert second["written"] == 0 and second["failed"] == 0
    table = read_pairs(out)
    pairs = list(zip(table.column("source_key").to_pylist(), table.column("jpeg_quality").to_pylist(), strict=True))
    assert len(pairs) == len(set(pairs))


def test_partial_run_then_resume_completes(tree, tmp_path):
    out = tmp_path / "pairs"
    partial = _build(tree, out, limit=2)
    assert partial["written"] + partial["failed"] == 2
    rest = _build(tree, out)
    assert rest["skipped"] == 2
    total = read_pairs(out).num_rows + rest["failed"] + partial["failed"]
    assert total == rest["planned"]


def test_shards_are_split_by_dataset_and_split(tree, tmp_path):
    out = tmp_path / "pairs"
    _build(tree, out, shard_rows=2)
    parts = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.parquet"))
    assert all(p.startswith(("svg-emoji/train/", "svg-emoji/valid/")) for p in parts)
    assert all(pq.ParquetFile(out / p).metadata.num_rows <= 2 for p in parts)


def test_metadata_records_tracer_configuration(tree, tmp_path):
    out = tmp_path / "pairs"
    _build(tree, out, seed=7)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["seed"] == 7
    assert meta["ladder"] == list(LADDER)
    assert "vtracer_version" in meta and "trace_params" in meta


def test_export_loose_writes_eyeballable_files(tree, tmp_path):
    out = tmp_path / "pairs"
    _build(tree, out)
    loose = tmp_path / "loose"
    written = export_loose(out, loose, n=3)
    assert len(written) == 3
    for d in written:
        names = {p.name for p in d.iterdir()}
        assert {"source.svg", "clean.png", "compressed.jpg", "clean_trace.svg", "messy_trace.svg", "info.json"} <= names


def test_resume_with_a_different_plan_is_refused(tree, tmp_path):
    out = tmp_path / "pairs"
    _build(tree, out, seed=0)
    with pytest.raises(ValueError, match="seed"):
        _build(tree, out, seed=1)


def test_resume_after_the_source_set_changed_is_refused(tree, tmp_path):
    out = tmp_path / "pairs"
    _build(tree, out)
    (tree / "train" / "svg" / "late.svg").write_text(BUSY_SVG)
    with pytest.raises(ValueError, match="n_sources"):
        _build(tree, out)


def test_spread_order_is_a_permutation_whose_prefixes_span_the_range():
    values = list(range(15, 96))
    order = spread_order(values)
    assert sorted(order) == values
    first = sorted(order[:9])
    assert first[0] == 15 and first[-1] == 95
    gaps = [b - a for a, b in zip(first, first[1:], strict=False)]
    assert max(gaps) <= 20
