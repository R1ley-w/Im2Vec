"""Synthesise (messy, clean) trace pairs from source SVGs — Phase 2, issue #2.

Each source SVG is rendered to a raster, JPEG-compressed at a logged quality,
and both rasters are traced::

    source SVG --render--> source raster --PNG--> clean trace
                                  \\--JPEG q--> messy trace

The target of any cleanup is the tracer's *own* clean trace, not the
hand-authored source SVG, so the pairs describe "undo tracer noise" rather
than "guess the author's paths" (see ``CONTEXT.md``).

Both traces use vtracer's defaults. The tracer's format branch would hand
the JPEG speckle filtering, which removes exactly the excess paths these pairs
exist to capture — so the branch is bypassed deliberately.

Fidelity is always measured against the source raster (the render of the
source SVG composited over white), never against the JPEG.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .metrics import SIZE, foreground_mask, render_svg, rgb_over_white, rmse, ssim
from .raster import flatten_to_rgb
from .tokenizer import SVGTokenizer
from .tracer import DEFAULT_PARAMS, trace_svg

#: JPEG qualities every ladder source is compressed at: a compression ->
#: messiness curve per source.
LADDER: Tuple[int, ...] = (15, 30, 50, 75, 95)

#: Inclusive range the single random quality of non-ladder sources is drawn from.
QUALITY_RANGE: Tuple[int, int] = (15, 95)

#: Tracer parameters used for *both* traces of a pair.
TRACE_PARAMS = DEFAULT_PARAMS

#: Number of source-size bands per (dataset, split) when stratifying the ladder.
SIZE_BANDS = 4

_PATH_TAG = re.compile(r"<path\b")
_TOKENIZER = SVGTokenizer()


class RenderError(ValueError):
    """The source SVG could not be rasterised."""


@dataclass(frozen=True)
class Source:
    """One source SVG on disk."""

    dataset: str
    split: str
    source_id: str
    path: Optional[Path]
    size_bytes: int

    @property
    def key(self) -> str:
        return f"{self.dataset}/{self.split}/{self.source_id}"


@dataclass(frozen=True)
class PlannedPair:
    """A (source, JPEG quality) combination to synthesise."""

    source: Source
    quality: int
    subset: str  # "ladder" or "random"


# -- discovery ---------------------------------------------------------------


def discover_sources(roots: Iterable[Path]) -> List[Source]:
    """Find every source SVG under HF-layout dataset roots.

    Expects ``<root>/<split>/svg/*.svg``, optionally sharded one level deeper
    (``<root>/<split>/svg/shardNN/*.svg``). The dataset name is the root's
    directory name with any ``-hf`` suffix dropped, so ``data/svg-emoji-hf``
    yields ``svg-emoji``. Splits are inherited from the directory layout.
    """
    sources: List[Source] = []
    for root in roots:
        root = Path(root)
        dataset = root.name[:-3] if root.name.endswith("-hf") else root.name
        for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            svg_dir = split_dir / "svg"
            if not svg_dir.is_dir():
                continue
            for path in svg_dir.rglob("*.svg"):
                sources.append(
                    Source(dataset, split_dir.name, path.stem, path, path.stat().st_size)
                )
    sources.sort(key=lambda s: s.key)
    return sources


# -- quality assignment -----------------------------------------------------


def _unit(seed: int, purpose: str, key: str) -> float:
    """Deterministic pseudo-random number in ``[0, 1)`` for ``key``."""
    digest = hashlib.sha256(f"{seed}:{purpose}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _strata(sources: Sequence[Source]) -> Dict[Tuple[str, str, int], List[Source]]:
    """Group by (dataset, split, size band); bands are per (dataset, split)."""
    by_group: Dict[Tuple[str, str], List[Source]] = defaultdict(list)
    for s in sources:
        by_group[(s.dataset, s.split)].append(s)

    strata: Dict[Tuple[str, str, int], List[Source]] = {}
    for (dataset, split), group in by_group.items():
        group = sorted(group, key=lambda s: (s.size_bytes, s.key))
        for i, s in enumerate(group):
            band = i * SIZE_BANDS // len(group)
            strata.setdefault((dataset, split, band), []).append(s)
    return strata


def _allocate(sizes: Dict[Tuple, int], total: int) -> Dict[Tuple, int]:
    """Proportional allocation by largest remainder; sums exactly to ``total``."""
    population = sum(sizes.values())
    quotas = {k: total * n / population for k, n in sizes.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    leftover = total - sum(counts.values())
    by_remainder = sorted(quotas, key=lambda k: (-(quotas[k] - counts[k]), k))
    for k in by_remainder[:leftover]:
        counts[k] += 1
    return counts


def assign_qualities(
    sources: Sequence[Source], ladder_size: int = 5000, seed: int = 0
) -> List[PlannedPair]:
    """Plan every pair to synthesise.

    A stratified subset of ``ladder_size`` sources gets every quality in
    :data:`LADDER`; every other source gets one quality drawn uniformly from
    :data:`QUALITY_RANGE`. Strata are (dataset, split, source-size band), so
    the ladder covers both datasets, all splits and the full range of source
    complexity in proportion. Selection depends only on source keys and
    ``seed``, never on input order.
    """
    if ladder_size >= len(sources):
        ladder_keys = {s.key for s in sources}
    else:
        strata = _strata(sources)
        counts = _allocate({k: len(v) for k, v in strata.items()}, ladder_size)
        ladder_keys = set()
        for stratum, members in strata.items():
            ranked = sorted(members, key=lambda s: _unit(seed, "ladder", s.key))
            ladder_keys.update(s.key for s in ranked[: counts[stratum]])

    lo, hi = QUALITY_RANGE
    plan: List[PlannedPair] = []
    for s in sorted(sources, key=lambda s: s.key):
        if s.key in ladder_keys:
            plan.extend(PlannedPair(s, q, "ladder") for q in LADDER)
        else:
            q = lo + int(_unit(seed, "quality", s.key) * (hi - lo + 1))
            plan.append(PlannedPair(s, q, "random"))
    return plan


# -- synthesis --------------------------------------------------------------


def count_paths(svg: str) -> int:
    """Number of ``<path>`` elements in a trace (vtracer emits nothing else)."""
    return len(_PATH_TAG.findall(svg))


def count_tokens(svg: str) -> int:
    """Compactness in ``SVGTokenizer`` tokens, the measure used since Phase 1."""
    return len(_TOKENIZER.encode_svg(svg))


def _encode(image: Image.Image, fmt: str, **kwargs) -> bytes:
    buf = io.BytesIO()
    image.save(buf, fmt, **kwargs)
    return buf.getvalue()


def _fidelity(reference: np.ndarray, trace: str, size: int) -> Tuple[float, float]:
    """(RMSE on the 0-255 scale, SSIM) of a trace against the source raster."""
    rendered = rgb_over_white(render_svg(trace, size), size)
    return rmse(reference, rendered) * 255.0, ssim(reference, rendered)


def synthesize(svg_text: str, quality: int, size: int = SIZE) -> Dict[str, object]:
    """Build one (messy, clean) pair from a source SVG.

    :raises RenderError: if the source SVG cannot be rasterised.
    """
    try:
        source = render_svg(svg_text, size)
    except Exception as exc:  # noqa: BLE001 - cairosvg raises many types
        raise RenderError(f"could not render source SVG: {exc}") from exc

    reference = rgb_over_white(source, size)
    raster = flatten_to_rgb(source)
    clean_png = _encode(raster, "PNG")
    jpeg = _encode(raster, "JPEG", quality=quality)

    clean_trace = trace_svg(clean_png, params=TRACE_PARAMS)
    messy_trace = trace_svg(jpeg, params=TRACE_PARAMS)

    decoded = np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.float32) / 255.0
    clean_rmse, clean_ssim = _fidelity(reference, clean_trace, size)
    messy_rmse, messy_ssim = _fidelity(reference, messy_trace, size)

    return {
        "jpeg_quality": quality,
        "source_fg_fraction": float(foreground_mask(source, size).mean()),
        "clean_png": clean_png,
        "jpeg": jpeg,
        "jpeg_rmse": rmse(reference, decoded) * 255.0,
        "clean_trace": clean_trace,
        "messy_trace": messy_trace,
        "clean_paths": count_paths(clean_trace),
        "messy_paths": count_paths(messy_trace),
        "clean_tokens": count_tokens(clean_trace),
        "messy_tokens": count_tokens(messy_trace),
        "clean_rmse": clean_rmse,
        "messy_rmse": messy_rmse,
        "clean_ssim": clean_ssim,
        "messy_ssim": messy_ssim,
    }
