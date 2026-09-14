"""Classify a messy trace's excess paths: droppable, mergeable, or distorted.

This is the second of the two measurements that gate Phase 3 (#3):

    >60% of excess paths are droppable/mergeable rather than geometrically
    distorted

If most excess paths can simply be removed, cleanup is a per-path
classification problem. If they cannot, it is a generation problem.

Definitions, all judged against the **source raster** (never the clean trace,
which has its own errors):

* **excess** — ``messy paths - clean paths``: how many more paths the messy
  trace spends than the tracer needs on the uncompressed image.
* **droppable** — deleting the path does not worsen fidelity. In vtracer's
  stacked output, the pixels it covered then show whatever lies beneath.
* **mergeable** — the path can be unioned into an adjacent or underlying
  path without worsening fidelity. For fidelity, a union is equivalent to
  repainting the path in its neighbour's colour, so that is how merges are
  evaluated. A merge removes one path.
* **distorted** — excess not accounted for by drops and merges: paths that
  are needed to reach the trace's fidelity and cannot be removed as a whole.

Costs are computed exactly on an aliasing-free paint map: each path is
rasterised on its own with ``shape-rendering="crispEdges"``, so every pixel
has a well-defined topmost path and the path beneath it. Removals are
applied greedily in rounds (cheapest first, with no two interacting
removals in one round). The repaired trace is then re-rendered normally
(antialiased) and its RMSE reported, so the classification is checked
against a real render rather than taken on trust.

The first path (vtracer's canvas-sized base layer) is never removed.

Greedy rounds are conservative: a merged path and its merge target are
frozen for later rounds, so chains of merges are under-counted, never
over-counted.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, find_objects

from .metrics import SIZE, render_svg, rgb_over_white, rmse

_SVG_OPEN = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_PATH_ELEMENT = re.compile(r"<path\b[^>]*/>", re.IGNORECASE)
_D_ATTR = re.compile(r'\sd="([^"]*)"')
_FILL_ATTR = re.compile(r'\sfill="#([0-9a-fA-F]{6})"')

#: Fidelity slack, in summed squared error on the [0, 1] scale, allowed per
#: removal. Zero means a removal may not make the paint map any worse.
DEFAULT_TOLERANCE = 0.0

_EPS = 1e-9
_WHITE = np.ones(3)
_NEIGHBOURHOOD = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class TracePath:
    """One ``<path>`` of a trace, in paint order."""

    element: str
    d: str
    fill: Tuple[int, int, int]


@dataclass
class ExcessReport:
    messy_paths: int
    clean_paths: int
    excess: int
    dropped: int
    merged: int
    distorted: int
    removable_fraction: Optional[float]
    repaired_paths: int
    messy_rmse: float
    repaired_rmse: float
    dropped_only_paths: int
    dropped_only_rmse: float
    repaired_svg: str
    dropped_only_svg: str
    rounds: int


def parse_trace(svg: str) -> List[TracePath]:
    """Paths of a vtracer trace in paint order (later paths paint over earlier)."""
    out = []
    for element in _PATH_ELEMENT.findall(svg):
        d = _D_ATTR.search(element)
        fill = _FILL_ATTR.search(element)
        if not d or not fill:
            continue
        hexcode = fill.group(1)
        rgb = tuple(int(hexcode[i : i + 2], 16) for i in (0, 2, 4))
        out.append(TracePath(element, d.group(1), rgb))
    return out


def _svg_open(svg: Optional[str]) -> str:
    match = _SVG_OPEN.search(svg) if svg else None
    if match:
        return match.group(0)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
        f'viewBox="0 0 {SIZE} {SIZE}">'
    )


def path_masks(
    paths: Sequence[TracePath], size: int = SIZE, svg_open: Optional[str] = None
) -> np.ndarray:
    """Aliasing-free coverage mask per path, shape ``(N, size, size)``."""
    import cairosvg

    head = _svg_open(svg_open)
    head = head[:-1] + ' shape-rendering="crispEdges">'
    masks = np.zeros((len(paths), size, size), dtype=bool)
    for i, path in enumerate(paths):
        png = cairosvg.svg2png(
            bytestring=(head + path.element + "</svg>").encode(),
            output_width=size,
            output_height=size,
        )
        alpha = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3]
        masks[i] = alpha > 127
    return masks


def paint_layers(
    masks: np.ndarray, alive: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Index of the topmost and second-topmost path at each pixel (-1 = none)."""
    n, h, w = masks.shape
    top = np.full((h, w), -1, dtype=np.int32)
    second = np.full((h, w), -1, dtype=np.int32)
    for i in range(n):
        if alive is not None and not alive[i]:
            continue
        m = masks[i]
        second[m] = top[m]
        top[m] = i
    return top, second


def _render_rmse(svg: str, reference: np.ndarray, size: int) -> float:
    return rmse(reference, rgb_over_white(render_svg(svg, size), size)) * 255.0


def _assemble(svg_open: str, paths: Sequence[TracePath], keep: np.ndarray, colours: np.ndarray) -> str:
    parts = [svg_open]
    for i, path in enumerate(paths):
        if not keep[i]:
            continue
        rgb = tuple(int(round(c * 255)) for c in colours[i])
        if rgb != path.fill:
            element = _FILL_ATTR.sub(' fill="#%02X%02X%02X"' % rgb, path.element, count=1)
        else:
            element = path.element
        parts.append(element)
    parts.append("</svg>")
    return "\n".join(parts)


def classify_excess(
    messy_svg: str,
    clean_svg: str,
    reference: np.ndarray,
    tolerance: float = DEFAULT_TOLERANCE,
    size: int = SIZE,
    max_rounds: int = 50,
) -> ExcessReport:
    """Count how much of a messy trace's excess could be dropped or merged.

    :param reference: the source raster as RGB floats in ``[0, 1]``.
    """
    paths = parse_trace(messy_svg)
    clean_paths = len(parse_trace(clean_svg))
    n = len(paths)
    head = _svg_open(messy_svg)

    masks = path_masks(paths, size, head)
    colours = np.array([p.fill for p in paths], dtype=np.float64).reshape(n, 3) / 255.0
    ref = reference.astype(np.float64)
    ref_sq = (ref**2).sum(axis=-1)

    alive = np.ones(n, dtype=bool)  # still drawn (merged paths stay drawn)
    counted = np.ones(n, dtype=bool)  # still counts as a path
    frozen = np.zeros(n, dtype=bool)
    # vtracer's stacked mode always opens with a canvas-sized base layer, in
    # clean and messy traces alike. Dropping it over a white canvas is often
    # free, but it is not excess, so it is never a removal candidate.
    frozen[:1] = True
    dropped = merged = rounds = 0

    for rounds in range(1, max_rounds + 1):
        top, second = paint_layers(masks, alive)
        slices = find_objects(top + 1, max_label=n)
        candidates = []  # (cost, index, target or -1 for drop, revealed ids)
        for p in range(n):
            if not counted[p] or frozen[p]:
                continue
            box = slices[p]
            if box is None:  # fully occluded: contributes nothing
                candidates.append((0.0, p, -1, ()))
                continue
            box = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
            crop_top, crop_second = top[box], second[box]
            visible = crop_top == p
            r = ref[box][visible]
            k = len(r)
            r_sum, r_sq = r.sum(axis=0), ref_sq[box][visible].sum()

            def cost_of(colour: np.ndarray) -> float:
                return k * float(colour @ colour) - 2 * float(colour @ r_sum) + r_sq

            current = cost_of(colours[p])

            below = crop_second[visible]
            revealed = tuple(int(i) for i in np.unique(below) if i >= 0)
            below_colours = np.where((below >= 0)[:, None], colours[np.maximum(below, 0)], _WHITE)
            drop_cost = float(((below_colours - r) ** 2).sum()) - current
            if drop_cost <= tolerance + _EPS:
                candidates.append((drop_cost, p, -1, revealed))
                continue

            ring = binary_dilation(visible, _NEIGHBOURHOOD) & ~visible
            neighbours = set(int(i) for i in np.unique(crop_top[ring]) if i >= 0)
            neighbours.update(revealed)
            neighbours.discard(p)
            best = None
            for q in neighbours:
                merge_cost = cost_of(colours[q]) - current
                if merge_cost <= tolerance + _EPS and (best is None or merge_cost < best[0]):
                    best = (merge_cost, q)
            if best is not None:
                candidates.append((best[0], p, best[1], ()))

        removed: Set[int] = set()
        protected: Set[int] = set()
        accepted = 0
        for cost, p, target, revealed in sorted(candidates, key=lambda c: (c[0], c[1])):
            if p in removed or p in protected:
                continue
            if target < 0:
                if any(i in removed for i in revealed):
                    continue
                alive[p] = counted[p] = False
                removed.add(p)
                protected.update(revealed)
                dropped += 1
            else:
                if target in removed or not alive[target]:
                    continue
                colours[p] = colours[target]
                counted[p] = False
                frozen[p] = frozen[target] = True
                removed.add(p)
                protected.add(target)
                merged += 1
            accepted += 1
        if not accepted:
            break

    excess = max(n - clean_paths, 0)
    removed_total = dropped + merged
    fraction = min(removed_total, excess) / excess if excess else None

    repaired_svg = _assemble(head, paths, alive, colours)
    original = np.array([p.fill for p in paths], dtype=np.float64).reshape(n, 3) / 255.0
    dropped_only_svg = _assemble(head, paths, alive, original)

    return ExcessReport(
        messy_paths=n,
        clean_paths=clean_paths,
        excess=excess,
        dropped=dropped,
        merged=merged,
        distorted=excess - min(removed_total, excess),
        removable_fraction=fraction,
        repaired_paths=n - removed_total,
        messy_rmse=_render_rmse(messy_svg, reference, size),
        repaired_rmse=_render_rmse(repaired_svg, reference, size),
        dropped_only_paths=int(alive.sum()),
        dropped_only_rmse=_render_rmse(dropped_only_svg, reference, size),
        repaired_svg=repaired_svg,
        dropped_only_svg=dropped_only_svg,
        rounds=rounds,
    )
