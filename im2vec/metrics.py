"""Fidelity metrics for comparing a traced SVG against its source raster.

All metrics are torch-free (numpy + scipy only) so the tracing app and the
Phase 2 analysis run in a plain Python environment with no deep-learning
dependency.

Two independent axes describe a trace (see ``CONTEXT.md``):

* **fidelity** — how closely the re-rendered SVG matches the source pixels,
  measured here by :func:`rmse`, :func:`l1`, :func:`ssim` and :func:`iou`.
* **compactness** — how few paths/tokens the SVG uses, measured elsewhere via
  ``SVGTokenizer``.

Tuning a tracer trades one against the other, so both must be reported
together; a metric quoted alone says nothing about trace quality.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

SIZE = 256

_SSIM_WINDOW = 11
_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def render_svg(svg: str, size: int = SIZE) -> Image.Image:
    """Rasterise an SVG string to an RGBA image at ``size x size``."""
    import cairosvg

    png = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"), output_width=size, output_height=size
    )
    return Image.open(io.BytesIO(png)).convert("RGBA")


def rgb_over_white(img: Image.Image, size: int = SIZE) -> np.ndarray:
    """Composite an RGBA image over white; returns RGB floats in ``[0, 1]``."""
    rgba = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3:4]
    return rgb * alpha + (1.0 - alpha)


def foreground_mask(img: Image.Image, size: int = SIZE) -> np.ndarray:
    """Boolean mask of non-transparent pixels."""
    alpha = np.asarray(img.resize((size, size)), dtype=np.float32)[..., 3]
    return alpha > 0.5


def l1(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error (0 = identical)."""
    return float(np.abs(a - b).mean())


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Root mean squared error (0 = identical).

    Preferred over :func:`l1` for comparing tracer configurations: squaring
    penalises the few badly-wrong pixels that compression artefacts produce,
    which is exactly the failure mode being measured.
    """
    return float(np.sqrt(((a - b) ** 2).mean()))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks; 0 when both are empty."""
    union = np.logical_or(a, b).sum()
    if not union:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity (uniform 11x11 window) on RGB images in ``[0,1]``.

    Zero-padded at the edges to match the reference implementation this
    replaces (``F.conv2d(..., padding=5)``).
    """
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)

    def box(x: np.ndarray) -> np.ndarray:
        # uniform_filter averages over the window; mode="constant" zero-pads,
        # matching a conv2d with a 1/N kernel and zero padding.
        return uniform_filter(x, size=(_SSIM_WINDOW, _SSIM_WINDOW, 1), mode="constant")

    mu_a, mu_b = box(a), box(b)
    mu_a_sq, mu_b_sq, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sigma_a = box(a * a) - mu_a_sq
    sigma_b = box(b * b) - mu_b_sq
    sigma_ab = box(a * b) - mu_ab

    numerator = (2 * mu_ab + _C1) * (2 * sigma_ab + _C2)
    denominator = (mu_a_sq + mu_b_sq + _C1) * (sigma_a + sigma_b + _C2)
    return float((numerator / denominator).mean())
