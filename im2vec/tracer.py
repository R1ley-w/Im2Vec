"""Classical raster-to-SVG tracing via vtracer.

This is the project's primary conversion path. It replaced a generative
Transformer that pooled an image into a single vector and decoded SVG tokens
from it — see ``docs/adr/0001-classical-tracing-over-generative-model.md``.

Tracing trades **fidelity** against **compactness** (``CONTEXT.md``), and the
best trade depends on whether the input was lossily compressed: a JPEG's
artefacts are faithfully reproduced by the tracer as thousands of tiny extra
paths, so aggressive speckle filtering pays for itself. On an uncompressed
raster the same filtering only destroys detail. Measured over 30 logos at
256x256 (RMSE against the source raster, tokens via ``SVGTokenizer``):

===================  =====================  =====================
config               clean PNG in           JPEG (q15-40) in
===================  =====================  =====================
vtracer defaults     12.67 RMSE / 1912 tok  14.63 RMSE / 14906 tok
filter_speckle=16,   16.93 RMSE / 1375 tok  17.43 RMSE /  2494 tok
color_precision=5
===================  =====================  =====================

Hence the format branch below: tuning costs 4.3 RMSE to save 537 tokens on a
PNG (a bad trade) but 2.8 RMSE to save 12412 tokens on a JPEG (a good one).
"""

from __future__ import annotations

import io
import re
from typing import Any, Dict, Optional

import vtracer
from PIL import Image, UnidentifiedImageError

from .raster import flatten_to_rgb

#: vtracer's own defaults — best fidelity, used for losslessly-encoded input.
DEFAULT_PARAMS: Dict[str, Any] = {}

#: Tuned for lossily-compressed input, where artefacts fragment the trace.
JPEG_PARAMS: Dict[str, Any] = {"filter_speckle": 16, "color_precision": 5}

#: Pillow format names that indicate lossy compression.
_LOSSY_FORMATS = frozenset({"JPEG", "JPEG2000", "MPO"})

_XML_PROLOG = re.compile(r"^\s*(?:<\?xml[^>]*\?>|<!--.*?-->)\s*", re.DOTALL)
_SVG_OPEN = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_DIMENSIONS = re.compile(
    r'width="(?P<w>[0-9.]+)"\s+height="(?P<h>[0-9.]+)"', re.IGNORECASE
)


def _clean_svg(svg: str) -> str:
    """Strip vtracer's XML prolog and give the root a ``viewBox``.

    vtracer emits ``<?xml ...?>`` plus a generator comment, which makes the
    output invalid to inline into an HTML document — and the apps embed it
    directly for preview. It also sets ``width``/``height`` with no
    ``viewBox``, so the result does not scale; adding one is what makes the
    output usable as a vector rather than a fixed-size image.
    """
    while True:
        stripped = _XML_PROLOG.sub("", svg, count=1)
        if stripped == svg:
            break
        svg = stripped

    match = _SVG_OPEN.search(svg)
    if match and "viewbox" not in match.group(0).lower():
        dims = _DIMENSIONS.search(match.group(0))
        if dims:
            tag = match.group(0)
            box = f' viewBox="0 0 {dims.group("w")} {dims.group("h")}"'
            svg = svg[: match.start()] + tag[:-1] + box + ">" + svg[match.end() :]

    return svg.strip()


def params_for_format(image_format: Optional[str]) -> Dict[str, Any]:
    """Pick tracing parameters from Pillow's detected format.

    Format is a *proxy* for compression, not a measurement of it: a PNG
    re-encoded from a JPEG gets the lossless branch and traces badly.

    Phase 2 measured how close the JPEG branch is to the best available
    settings (``docs/phase2/results.md``): the best of 72 settings, chosen
    per image with the source raster in hand, saves only ~3% of tokens at
    median, and no single setting beats it at q15-q30.
    """
    if image_format and image_format.upper() in _LOSSY_FORMATS:
        return JPEG_PARAMS
    return DEFAULT_PARAMS


def trace_svg(
    image_bytes: bytes,
    *,
    params: Optional[Dict[str, Any]] = None,
) -> str:
    """Trace raster bytes to an SVG string.

    :param image_bytes: the uploaded file's raw bytes, any Pillow-readable format.
    :param params: vtracer overrides; when omitted they are chosen from the
        detected image format by :func:`params_for_format`.
    :raises ValueError: if the bytes are empty or not a decodable image.
    """
    if not image_bytes:
        raise ValueError("No image data provided")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc

    if params is None:
        params = params_for_format(image.format)

    # Composite away any alpha before tracing: vtracer sees RGB, and a naive
    # alpha drop would blacken transparent backgrounds.
    flat = flatten_to_rgb(image)

    buf = io.BytesIO()
    flat.save(buf, "PNG")
    svg = vtracer.convert_raw_image_to_svg(
        buf.getvalue(), img_format="png", **params
    )
    return _clean_svg(svg)
