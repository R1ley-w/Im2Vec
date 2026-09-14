"""Run the two Phase 2 measurements that gate Phase 3 (#2, #3).

For each ladder pair in the chosen split (``valid`` by default; ``test`` is
left untouched for evaluating any Phase 3 model):

1. **Frontier sweep** — trace the JPEG under every setting in
   :data:`im2vec.frontier.SWEEP` (gate 1).
2. **Excess classification** — drop/merge/distorted counts for the messy
   trace (gate 2), plus the oracle-cleaned traces it produces.

Per-pair results go to ``<out>/sweep.parquet`` and ``<out>/excess.parquet``;
:func:`summarise` turns them into ``<out>/summary.json``.

Usage::

    python -m im2vec.data.analyze_pairs --pairs data/pairs --out data/analysis
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..excess import classify_excess
from ..frontier import SWEEP, VTRACER_DEFAULTS, FrontierPoint, headroom, pareto_front, sweep_raster
from ..metrics import render_svg, rgb_over_white
from ..pairs import LADDER, count_tokens
from ..tracer import JPEG_PARAMS
from .build_pairs import read_pairs

GATE1_THRESHOLD = 0.25
GATE2_THRESHOLD = 0.60

_DEFAULT_LABEL = next(l for l, p in SWEEP if p == VTRACER_DEFAULTS)
_SHIPPED_LABEL = next(l for l, p in SWEEP if p == {**VTRACER_DEFAULTS, **JPEG_PARAMS})

_ROW_COLUMNS = [
    "source_key", "dataset", "split", "jpeg_quality", "source_svg", "jpeg",
    "clean_trace", "messy_trace", "clean_tokens", "clean_rmse", "messy_tokens",
    "messy_rmse", "clean_paths", "messy_paths",
]


def _analyse(row: dict) -> Dict[str, object]:
    base = {k: row[k] for k in ("source_key", "dataset", "split", "jpeg_quality")}
    try:
        reference = rgb_over_white(render_svg(row["source_svg"]))
        points = sweep_raster(row["jpeg"], reference)
        report = classify_excess(row["messy_trace"], row["clean_trace"], reference)
    except Exception as exc:  # noqa: BLE001 - record and move on
        return {**base, "error": f"{type(exc).__name__}: {exc}"[:500]}
    return {
        **base,
        "error": None,
        "sweep": [
            {"label": p.label, "tokens": p.tokens, "rmse": p.rmse, "paths": p.paths}
            for p in points
        ],
        "excess": {
            "clean_tokens": row["clean_tokens"],
            "clean_rmse": row["clean_rmse"],
            "clean_paths": row["clean_paths"],
            "messy_tokens": row["messy_tokens"],
            "messy_paths": report.messy_paths,
            "excess": report.excess,
            "dropped": report.dropped,
            "merged": report.merged,
            "distorted": report.distorted,
            "removable_fraction": report.removable_fraction,
            "rounds": report.rounds,
            "messy_rmse": report.messy_rmse,
            "repaired_paths": report.repaired_paths,
            "repaired_rmse": report.repaired_rmse,
            "dropped_only_paths": report.dropped_only_paths,
            "dropped_only_rmse": report.dropped_only_rmse,
            "dropped_only_tokens": count_tokens(report.dropped_only_svg),
        },
    }


def select_rows(pairs: Path, split: str, subset: str = "ladder", limit: Optional[int] = None) -> List[dict]:
    wanted = (pc.field("split") == split) & (pc.field("subset") == subset)
    table = read_pairs(pairs, columns=_ROW_COLUMNS, where=wanted)
    if limit is not None:
        # Keep whole ladders together so every source has all qualities.
        keys = sorted(set(table.column("source_key").to_pylist()))[: max(1, limit // len(LADDER))]
        table = table.filter(pc.is_in(table["source_key"], pa.array(keys)))
    return table.to_pylist()


def run(pairs: Path, out: Path, split: str = "valid", workers: Optional[int] = None,
        limit: Optional[int] = None) -> Dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    rows = select_rows(pairs, split, limit=limit)
    sweep_rows: List[dict] = []
    excess_rows: List[dict] = []
    errors = 0
    started = time.monotonic()
    context = multiprocessing.get_context("forkserver")
    with context.Pool(workers or os.cpu_count() or 1) as pool:
        for i, result in enumerate(pool.imap_unordered(_analyse, rows, chunksize=1), 1):
            base = {k: result[k] for k in ("source_key", "dataset", "split", "jpeg_quality")}
            if result["error"]:
                errors += 1
                print(f"[error] {base['source_key']} q{base['jpeg_quality']}: {result['error']}",
                      file=sys.stderr)
                continue
            sweep_rows.extend({**base, **p} for p in result["sweep"])
            excess_rows.append({**base, **result["excess"]})
            if i % 100 == 0:
                rate = i / (time.monotonic() - started)
                print(f"{i}/{len(rows)}  {rate:.1f}/s  eta {(len(rows) - i) / rate / 60:.1f} min",
                      file=sys.stderr, flush=True)
    pq.write_table(pa.Table.from_pylist(sweep_rows), out / "sweep.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pylist(excess_rows), out / "excess.parquet", compression="zstd")
    return {"pairs": len(rows), "analysed": len(excess_rows), "errors": errors}


# -- summaries ---------------------------------------------------------------


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(arr):
        return {"n": 0}
    q = np.quantile(arr, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {"n": int(len(arr)), "mean": float(arr.mean()), "p10": q[0], "p25": q[1],
            "median": q[2], "p75": q[3], "p90": q[4]}


def _image_fronts(sweep: List[dict]) -> Dict[tuple, List[FrontierPoint]]:
    """Each (source, quality) pair's own frontier."""
    points: Dict[tuple, List[FrontierPoint]] = {}
    for r in sweep:
        points.setdefault((r["source_key"], r["jpeg_quality"]), []).append(
            FrontierPoint(r["label"], r["tokens"], r["rmse"]))
    return {k: pareto_front(v) for k, v in points.items()}


def _setting_points(sweep: List[dict], stat: str = "median") -> List[FrontierPoint]:
    """One point per setting: (stat of tokens, stat of RMSE) over all images."""
    f = np.median if stat == "median" else np.mean
    by_label: Dict[str, Dict[str, List[float]]] = {}
    for r in sweep:
        d = by_label.setdefault(r["label"], {"tokens": [], "rmse": []})
        d["tokens"].append(r["tokens"])
        d["rmse"].append(r["rmse"])
    return [FrontierPoint(l, float(f(d["tokens"])), float(f(d["rmse"]))) for l, d in by_label.items()]


def aggregate_curve(sweep: List[dict], excess: List[dict], stat: str = "median") -> Dict[str, object]:
    """Dataset-level frontier: one setting applied to every image, as in Phase 1.

    Each setting becomes one point, (stat of tokens, stat of RMSE) over the
    images. The clean trace and the oracle cleanups become corner points
    summarised the same way, with headroom measured against the curve.
    """
    f = np.median if stat == "median" else np.mean
    points = _setting_points(sweep, stat)
    front = pareto_front(points)

    def corner(tokens_key: str, rmse_key: str) -> Dict[str, object]:
        t = float(f([e[tokens_key] for e in excess]))
        r = float(f([e[rmse_key] for e in excess]))
        return {"tokens": t, "rmse": r, "headroom": headroom(front, t, r)}

    by = {p.label: p for p in points}
    clean = corner("clean_tokens", "clean_rmse")

    def against(label: str) -> Dict[str, object]:
        p = by[label]
        return {"label": label, "tokens": p.tokens, "rmse": p.rmse,
                "clean_saves": 1 - clean["tokens"] / p.tokens,
                "clean_fidelity_no_worse": clean["rmse"] <= p.rmse}

    best_fidelity = min(front, key=lambda p: p.rmse)
    return {
        "stat": stat,
        "front": [{"label": p.label, "tokens": p.tokens, "rmse": p.rmse} for p in front],
        "clean_corner": clean,
        "dropped_only_corner": corner("dropped_only_tokens", "dropped_only_rmse"),
        "vs_defaults": against(_DEFAULT_LABEL),
        "vs_shipped_jpeg_branch": against(_SHIPPED_LABEL),
        "vs_best_fidelity_frontier_point": against(best_fidelity.label),
    }


#: RMSE slack (0-255 scale) granted to the frontier when comparing it with a
#: reference trace whose fidelity no swept setting reaches.
SLACKS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def slack_curve(
    sweep: List[dict], excess: List[dict], tokens_key: str = "clean_tokens",
    rmse_key: str = "clean_rmse", slacks: Sequence[float] = SLACKS,
) -> Dict[str, object]:
    """Headroom when the frontier may be up to ``slack`` RMSE worse than the reference.

    The reference (e.g. the clean trace) often has better fidelity than any
    setting reaches, so strict equal-or-better headroom is undefined. This
    shows how the answer depends on the slack allowed: per image (share of
    images with a frontier point within the slack, and their headroom), and
    on the median aggregate curve.
    """
    fronts = _image_fronts(sweep)
    aggregate = pareto_front(_setting_points(sweep, "median"))
    ref_tokens = float(np.median([e[tokens_key] for e in excess]))
    ref_rmse = float(np.median([e[rmse_key] for e in excess]))

    out = {}
    for slack in slacks:
        values = []
        for e in excess:
            h = headroom(fronts[(e["source_key"], e["jpeg_quality"])], e[tokens_key], e[rmse_key] + slack)
            if h is not None:
                values.append(h)
        out[str(slack)] = {
            "aggregate_median_headroom": headroom(aggregate, ref_tokens, ref_rmse + slack),
            "per_image_reachable_share": len(values) / len(excess) if excess else None,
            "per_image_headroom": _quantiles(values),
            "per_image_share_above_threshold": float(np.mean([h > GATE1_THRESHOLD for h in values])) if values else None,
        }
    return out


def per_image_headroom(sweep: List[dict], excess: List[dict]) -> Dict[str, object]:
    """Headroom per (source, quality) against that image's own frontier."""
    fronts = _image_fronts(sweep)
    out = {}
    for ref, (tk, rk) in {"clean": ("clean_tokens", "clean_rmse"),
                          "dropped_only": ("dropped_only_tokens", "dropped_only_rmse")}.items():
        values, unreachable = [], 0
        for e in excess:
            h = headroom(fronts[(e["source_key"], e["jpeg_quality"])], e[tk], e[rk])
            if h is None:
                unreachable += 1
            else:
                values.append(h)
        out[ref] = {
            "reachable": len(values),
            "unreachable": unreachable,
            "headroom": _quantiles(values),
            "share_above_threshold": float(np.mean([h > GATE1_THRESHOLD for h in values])) if values else None,
        }
    return out


def gate2(excess: List[dict]) -> Dict[str, object]:
    with_excess = [e for e in excess if e["excess"] > 0]
    removed = sum(min(e["dropped"] + e["merged"], e["excess"]) for e in with_excess)
    total = sum(e["excess"] for e in with_excess)
    return {
        "images": len(excess),
        "images_with_excess": len(with_excess),
        "pooled_removable_fraction": removed / total if total else None,
        "per_image_removable_fraction": _quantiles([e["removable_fraction"] for e in with_excess]),
        "dropped_share": sum(e["dropped"] for e in with_excess) / total if total else None,
        "merged_share": sum(e["merged"] for e in with_excess) / total if total else None,
        "repaired_minus_messy_rmse": _quantiles([e["repaired_rmse"] - e["messy_rmse"] for e in excess]),
        "dropped_only_minus_messy_rmse": _quantiles([e["dropped_only_rmse"] - e["messy_rmse"] for e in excess]),
        "share_images_removable_above_threshold": float(np.mean(
            [e["removable_fraction"] > GATE2_THRESHOLD for e in with_excess])) if with_excess else None,
    }


_CURVE_COLUMNS = ["dataset", "jpeg_quality", "clean_tokens", "messy_tokens", "clean_paths",
                  "messy_paths", "clean_rmse", "messy_rmse", "jpeg_rmse"]


def messiness_curve(rows: List[dict]) -> Dict[str, object]:
    """Compression -> messiness: how the messy trace degrades with JPEG quality.

    Built from ladder pairs, where every source appears at every quality, so
    each quality's statistics describe the same images.
    """
    def stats(group: List[dict]) -> Dict[str, object]:
        return {
            "pairs": len(group),
            "token_ratio": _quantiles([r["messy_tokens"] / r["clean_tokens"] for r in group]),
            "path_ratio": _quantiles([r["messy_paths"] / max(r["clean_paths"], 1) for r in group]),
            "messy_tokens": _quantiles([r["messy_tokens"] for r in group]),
            "jpeg_rmse": _quantiles([r["jpeg_rmse"] for r in group]),
            "messy_minus_clean_rmse": _quantiles([r["messy_rmse"] - r["clean_rmse"] for r in group]),
        }

    curve: Dict[str, object] = {}
    for dataset in ["all"] + sorted({r["dataset"] for r in rows}):
        subset = rows if dataset == "all" else [r for r in rows if r["dataset"] == dataset]
        curve[dataset] = {
            str(q): stats([r for r in subset if r["jpeg_quality"] == q])
            for q in sorted({r["jpeg_quality"] for r in subset})
        }
    return curve


def summarise(out: Path, pairs: Optional[Path] = None) -> Dict[str, object]:
    sweep = pq.read_table(out / "sweep.parquet").to_pylist()
    excess = pq.read_table(out / "excess.parquet").to_pylist()
    qualities = sorted({e["jpeg_quality"] for e in excess})

    def section(sw, ex):
        return {
            "gate1_aggregate_median": aggregate_curve(sw, ex, "median"),
            "gate1_aggregate_mean": aggregate_curve(sw, ex, "mean"),
            "gate1_per_image": per_image_headroom(sw, ex),
            "gate1_slack_clean": slack_curve(sw, ex),
            "gate1_slack_dropped_only": slack_curve(sw, ex, "dropped_only_tokens", "dropped_only_rmse"),
            "gate2": gate2(ex),
        }

    summary = {
        "thresholds": {"gate1": GATE1_THRESHOLD, "gate2": GATE2_THRESHOLD},
        "sources": len({e["source_key"] for e in excess}),
        "pairs": len(excess),
        "datasets": sorted({e["dataset"] for e in excess}),
        "all": section(sweep, excess),
        "by_quality": {
            str(q): section([s for s in sweep if s["jpeg_quality"] == q],
                            [e for e in excess if e["jpeg_quality"] == q])
            for q in qualities
        },
        "by_dataset": {
            d: section([s for s in sweep if s["dataset"] == d], [e for e in excess if e["dataset"] == d])
            for d in sorted({e["dataset"] for e in excess})
        },
    }
    if pairs is not None:
        ladder = read_pairs(pairs, columns=_CURVE_COLUMNS, where=pc.field("subset") == "ladder")
        summary["messiness_curve"] = messiness_curve(ladder.to_pylist())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 gate measurements")
    p.add_argument("--pairs", type=Path, default=Path("data/pairs"))
    p.add_argument("--out", type=Path, default=Path("data/analysis"))
    p.add_argument("--split", default="valid")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="cap pairs (whole ladders kept)")
    p.add_argument("--summary-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.summary_only:
        print(json.dumps(run(args.pairs, args.out, args.split, args.workers, args.limit)))
    summarise(args.out, args.pairs)
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
