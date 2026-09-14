"""The classical frontier over tracer settings — Phase 3 gate 1 (#3).

    The classical Pareto frontier leaves >25% token reduction available at
    equal-or-better RMSE

The **frontier** (``CONTEXT.md``) is the set of traces for which no better
fidelity is available without losing compactness. Here it is traced
empirically: the compressed raster is traced under every setting in
:data:`SWEEP`, and the non-dominated (tokens, RMSE) points are kept.

**Headroom** against a reference trace — e.g. the clean trace, or an oracle
cleanup — is the fraction of tokens the reference saves over the cheapest
frontier trace of equal-or-better fidelity::

    headroom = 1 - ref_tokens / tokens_at(front, ref_rmse)

It is undefined when no swept setting reaches the reference's fidelity: then
there is no frontier point to compare at that RMSE, and extrapolating one
would be invented data.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import SIZE, render_svg, rgb_over_white, rmse
from .pairs import count_paths, count_tokens
from .tracer import trace_svg

#: vtracer's own defaults, spelled out so sweep entries are comparable.
VTRACER_DEFAULTS = {"filter_speckle": 4, "color_precision": 6, "layer_difference": 16}

_SPECKLE = (4, 8, 16, 24, 32, 48)
_COLOR_PRECISION = (6, 5, 4, 3)
_LAYER_DIFFERENCE = (16, 32, 64)


def _label(params: Dict[str, int]) -> str:
    return f"sp{params['filter_speckle']}_cp{params['color_precision']}_ld{params['layer_difference']}"


#: Every tracer setting the frontier is built from: the compactness-relevant
#: knobs (speckle filter, colour quantisation, layer separation). Includes
#: vtracer's defaults and the tracer's shipped JPEG branch.
SWEEP: List[Tuple[str, Dict[str, int]]] = [
    (_label(p), p)
    for p in (
        {"filter_speckle": s, "color_precision": c, "layer_difference": l}
        for s, c, l in itertools.product(_SPECKLE, _COLOR_PRECISION, _LAYER_DIFFERENCE)
    )
]


@dataclass(frozen=True)
class FrontierPoint:
    label: str
    tokens: float  # an int per trace; a median or mean on aggregate curves
    rmse: float
    paths: int = 0


def sweep_raster(
    raster: bytes,
    reference: np.ndarray,
    configs: Sequence[Tuple[str, Dict[str, int]]] = SWEEP,
    size: int = SIZE,
) -> List[FrontierPoint]:
    """Trace ``raster`` under each config; fidelity against ``reference``."""
    points = []
    for label, params in configs:
        svg = trace_svg(raster, params=params)
        rendered = rgb_over_white(render_svg(svg, size), size)
        points.append(
            FrontierPoint(label, count_tokens(svg), rmse(reference, rendered) * 255.0, count_paths(svg))
        )
    return points


def pareto_front(points: Sequence[FrontierPoint]) -> List[FrontierPoint]:
    """Non-dominated points (fewer tokens, lower RMSE), sorted by tokens."""
    front: List[FrontierPoint] = []
    best_rmse = float("inf")
    for p in sorted(points, key=lambda p: (p.tokens, p.rmse)):
        if p.rmse < best_rmse:
            front.append(p)
            best_rmse = p.rmse
    return front


def tokens_at(front: Sequence[FrontierPoint], max_rmse: float) -> Optional[float]:
    """Fewest tokens on the frontier with RMSE at or below ``max_rmse``."""
    meeting = [p.tokens for p in front if p.rmse <= max_rmse]
    return min(meeting) if meeting else None


def headroom(
    front: Sequence[FrontierPoint], ref_tokens: float, ref_rmse: float
) -> Optional[float]:
    """Token reduction a reference trace shows over the frontier at its fidelity."""
    frontier_tokens = tokens_at(front, ref_rmse)
    if frontier_tokens is None:
        return None
    return 1.0 - ref_tokens / frontier_tokens
