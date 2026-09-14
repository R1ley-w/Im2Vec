"""Build the Phase 2 trace-pair dataset as sharded parquet (issue #2).

Output layout::

    <out>/metadata.json                     tracer config, seed, ladder
    <out>/failures.jsonl                    one record per pair that failed
    <out>/<dataset>/<split>/part-NNNNN.parquet

Parquet rather than loose files: ~230k loose files would hit both Hugging
Face limits that bit the source datasets (10k files per directory, 128
commits per hour). :func:`export_loose` writes a small sample as plain files
for eyeballing.

Builds are resumable: pairs already present in the output (or already logged
as failures) are skipped, so an interrupted run is finished by re-running the
same command.

Usage::

    python -m im2vec.data.build_pairs --out data/pairs
    python -m im2vec.data.build_pairs --out data/pairs --loose data/pairs-loose --loose-n 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import multiprocessing
import time
from collections import defaultdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..pairs import (
    LADDER,
    QUALITY_RANGE,
    TRACE_PARAMS,
    PlannedPair,
    assign_qualities,
    discover_sources,
    synthesize,
)

DEFAULT_DATA_DIRS = (Path("data/svg-emoji-hf"), Path("data/svg-stack-hf"))

SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("split", pa.string()),
        ("source_id", pa.string()),
        ("source_key", pa.string()),
        ("subset", pa.string()),
        ("jpeg_quality", pa.int16()),
        ("source_svg", pa.string()),
        ("source_fg_fraction", pa.float32()),
        ("clean_png", pa.binary()),
        ("jpeg", pa.binary()),
        ("jpeg_rmse", pa.float32()),
        ("clean_trace", pa.string()),
        ("messy_trace", pa.string()),
        ("clean_paths", pa.int32()),
        ("messy_paths", pa.int32()),
        ("clean_tokens", pa.int32()),
        ("messy_tokens", pa.int32()),
        ("clean_rmse", pa.float32()),
        ("messy_rmse", pa.float32()),
        ("clean_ssim", pa.float32()),
        ("messy_ssim", pa.float32()),
    ]
)

_KEY_COLUMNS = ["source_key", "jpeg_quality"]


def _read_svg(path: Path) -> str:
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _work(pair: PlannedPair) -> Tuple[bool, Dict[str, object]]:
    """Synthesise one pair in a worker; never raises."""
    s = pair.source
    base = {
        "dataset": s.dataset,
        "split": s.split,
        "source_id": s.source_id,
        "source_key": s.key,
        "subset": pair.subset,
        "jpeg_quality": pair.quality,
    }
    try:
        svg = _read_svg(s.path)
        row = synthesize(svg, pair.quality)
    except Exception as exc:  # noqa: BLE001 - one bad source must not stop the build
        return False, {**base, "reason": f"{type(exc).__name__}: {exc}"[:500]}
    return True, {**base, "source_svg": svg, **row}


def _parts(out: Path) -> List[Path]:
    return sorted(p for p in out.glob("*/*/part-*.parquet"))


def read_pairs(
    out: Path,
    columns: Optional[Sequence[str]] = None,
    filter: Optional[pc.Expression] = None,
) -> pa.Table:
    """Read every shard under ``out`` into one chunked table.

    ``filter`` is applied per shard. The full dataset's string columns exceed
    Arrow's 2 GB-per-array limit, so operations that merge chunks (``take``,
    ``combine_chunks``) fail on an unfiltered full read; filter first.
    """
    parts = _parts(Path(out))
    empty = SCHEMA.empty_table()
    if not parts:
        return empty if columns is None else empty.select(columns)
    return pa.concat_tables(pq.read_table(p, columns=columns, filters=filter) for p in parts)


def _done_keys(out: Path) -> Set[Tuple[str, int]]:
    done: Set[Tuple[str, int]] = set()
    table = read_pairs(out, columns=_KEY_COLUMNS)
    done.update(zip(table.column(0).to_pylist(), table.column(1).to_pylist()))
    failures = out / "failures.jsonl"
    if failures.exists():
        for line in failures.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["source_key"], rec["jpeg_quality"]))
    return done


class _ShardWriter:
    """Buffers rows per (dataset, split) and flushes fixed-size shards atomically."""

    def __init__(self, out: Path, shard_rows: int):
        self.out = out
        self.shard_rows = shard_rows
        self.buffers: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    def add(self, row: dict) -> None:
        group = (row["dataset"], row["split"])
        self.buffers[group].append(row)
        if len(self.buffers[group]) >= self.shard_rows:
            self._flush(group)

    def close(self) -> None:
        for group in list(self.buffers):
            self._flush(group)

    def _flush(self, group: Tuple[str, str]) -> None:
        rows = self.buffers.pop(group, [])
        if not rows:
            return
        directory = self.out / group[0] / group[1]
        directory.mkdir(parents=True, exist_ok=True)
        existing = [int(p.stem.split("-")[1]) for p in directory.glob("part-*.parquet")]
        index = max(existing, default=-1) + 1
        final = directory / f"part-{index:05d}.parquet"
        tmp = final.with_suffix(".parquet.tmp")
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, final)


#: Metadata that fixes which pairs are planned; a resume must not change it.
_PLAN_FIELDS = ("seed", "ladder", "ladder_size", "quality_range", "trace_params", "n_sources")


def _write_metadata(
    out: Path, seed: int, ladder_size: int, roots: Sequence[Path], n_sources: int
) -> None:
    meta = {
        "seed": seed,
        "ladder": list(LADDER),
        "ladder_size": ladder_size,
        "quality_range": list(QUALITY_RANGE),
        "trace_params": TRACE_PARAMS or "vtracer defaults",
        "vtracer_version": importlib_metadata.version("vtracer"),
        "cairosvg_version": importlib_metadata.version("cairosvg"),
        "pillow_version": importlib_metadata.version("pillow"),
        "sources": [str(r) for r in roots],
        # Ladder strata depend on the whole source set, so a partial download
        # plans different pairs than a complete one.
        "n_sources": n_sources,
        "fidelity_reference": "source SVG rendered at 256x256 and composited over white",
    }
    path = out / "metadata.json"
    if path.exists() and _parts(out):
        previous = json.loads(path.read_text())
        changed = [f for f in _PLAN_FIELDS if previous.get(f) != json.loads(json.dumps(meta[f]))]
        if changed:
            raise ValueError(
                f"{out} was built with different {', '.join(changed)}; "
                "resuming would mix two plans in one dataset"
            )
    path.write_text(json.dumps(meta, indent=2) + "\n")


def build_pairs(
    roots: Iterable[Path],
    out: Path,
    ladder_size: int = 5000,
    seed: int = 0,
    workers: Optional[int] = None,
    shard_rows: int = 2000,
    limit: Optional[int] = None,
    log_every: int = 0,
) -> Dict[str, int]:
    """Synthesise every planned pair not already in ``out``.

    Returns counts: ``planned``, ``skipped`` (already done), ``written``,
    ``failed``. ``limit`` caps how many new pairs this call attempts.
    """
    roots = [Path(r) for r in roots]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(roots)
    _write_metadata(out, seed, ladder_size, roots, len(sources))

    plan = assign_qualities(sources, ladder_size=ladder_size, seed=seed)
    done = _done_keys(out)
    todo = [p for p in plan if (p.source.key, p.quality) not in done]
    skipped = len(plan) - len(todo)
    if limit is not None:
        todo = todo[:limit]

    summary = {"planned": len(plan), "skipped": skipped, "written": 0, "failed": 0}
    if not todo:
        return summary

    writer = _ShardWriter(out, shard_rows)
    started = time.monotonic()
    workers = workers or os.cpu_count() or 1
    context = multiprocessing.get_context("forkserver")
    with context.Pool(workers) as pool, open(out / "failures.jsonl", "a") as failures:
        try:
            for i, (ok, record) in enumerate(pool.imap_unordered(_work, todo, chunksize=4), 1):
                if ok:
                    writer.add(record)
                    summary["written"] += 1
                else:
                    failures.write(json.dumps(record) + "\n")
                    failures.flush()
                    summary["failed"] += 1
                if log_every and i % log_every == 0:
                    rate = i / (time.monotonic() - started)
                    eta = (len(todo) - i) / rate
                    print(
                        f"{i}/{len(todo)} pairs  {rate:.1f}/s  eta {eta / 60:.1f} min  "
                        f"failed {summary['failed']}",
                        file=sys.stderr,
                        flush=True,
                    )
        finally:
            # Flush whatever finished, so an interrupted build resumes from here.
            writer.close()
    return summary


def spread_order(values: Sequence[int]) -> List[int]:
    """Reorder sorted ``values`` so every prefix is spread across the range.

    Ends first, then repeated midpoints (a binary subdivision), so taking the
    first ``k`` gives roughly evenly spaced values for any ``k``.
    """
    if len(values) <= 2:
        return list(values)
    order = [0, len(values) - 1]
    seen = set(order)
    intervals = [(0, len(values) - 1)]
    while intervals:
        next_intervals = []
        for lo, hi in intervals:
            mid = (lo + hi) // 2
            if mid not in seen:
                order.append(mid)
                seen.add(mid)
            if mid - lo > 1:
                next_intervals.append((lo, mid))
            if hi - mid > 1:
                next_intervals.append((mid, hi))
        intervals = next_intervals
    return [values[i] for i in order]


def export_loose(out: Path, dest: Path, n: int = 50, seed: int = 0) -> List[Path]:
    """Write ``n`` pairs as plain files for eyeballing, spread across qualities.

    Works shard by shard: the full dataset's string columns exceed Arrow's
    2 GB-per-array limit, so it must never be materialised as one table.
    """
    index = []  # (shard, row in shard, quality)
    for part in _parts(Path(out)):
        qualities = pq.read_table(part, columns=["jpeg_quality"]).column(0).to_pylist()
        index.extend((part, i, q) for i, q in enumerate(qualities))

    rng = random.Random(seed)
    by_quality: Dict[int, List[int]] = defaultdict(list)
    for i, (_, _, quality) in enumerate(index):
        by_quality[quality].append(i)
    for bucket in by_quality.values():
        rng.shuffle(bucket)
    # Visit qualities in spread order so even a sample smaller than the
    # number of distinct qualities spans the whole compression range.
    chosen: List[int] = []
    order = spread_order(sorted(by_quality))
    while len(chosen) < min(n, len(index)):
        for q in order:
            if by_quality[q] and len(chosen) < n:
                chosen.append(by_quality[q].pop())

    by_part: Dict[Path, List[int]] = defaultdict(list)
    for i in chosen:
        part, row, _ = index[i]
        by_part[part].append(row)

    written: List[Path] = []
    for part, rows in by_part.items():
        for row in pq.read_table(part).take(rows).to_pylist():
            name = f"{row['dataset']}_{row['split']}_{row['source_id']}_q{row['jpeg_quality']}"
            d = Path(dest) / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "source.svg").write_text(row.pop("source_svg"))
            (d / "clean.png").write_bytes(row.pop("clean_png"))
            (d / "compressed.jpg").write_bytes(row.pop("jpeg"))
            (d / "clean_trace.svg").write_text(row.pop("clean_trace"))
            (d / "messy_trace.svg").write_text(row.pop("messy_trace"))
            (d / "info.json").write_text(json.dumps(row, indent=2) + "\n")
            written.append(d)
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase 2 (messy, clean) trace pairs")
    p.add_argument("--data-dir", type=Path, action="append", default=None,
                   help="HF-layout source root; repeatable (default: svg-emoji-hf + svg-stack-hf)")
    p.add_argument("--out", type=Path, default=Path("data/pairs"))
    p.add_argument("--ladder-size", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--shard-rows", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None, help="attempt at most this many new pairs")
    p.add_argument("--loose", type=Path, default=None, help="also export a loose sample here")
    p.add_argument("--loose-n", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    roots = args.data_dir or list(DEFAULT_DATA_DIRS)
    summary = build_pairs(
        roots, args.out, ladder_size=args.ladder_size, seed=args.seed,
        workers=args.workers, shard_rows=args.shard_rows, limit=args.limit,
        log_every=1000,
    )
    print(json.dumps(summary))
    if args.loose:
        written = export_loose(args.out, args.loose, n=args.loose_n, seed=args.seed)
        print(f"exported {len(written)} loose pairs to {args.loose}")


if __name__ == "__main__":
    main()
