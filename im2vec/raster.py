"""Raster input normalisation shared by the tracer and the web apps."""

from __future__ import annotations

from typing import Tuple

from PIL import Image


def flatten_to_rgb(
    image: Image.Image, background: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """Composite a (possibly transparent) image onto ``background`` and drop alpha.

    ``Image.convert("RGB")`` does **not** alpha-composite — it just discards
    the alpha channel, exposing whatever RGB values sit underneath fully
    transparent pixels. Renderers such as cairosvg store ``(0, 0, 0)`` there,
    so a naive ``.convert("RGB")`` on a transparent-background PNG silently
    turns the entire background black. Always flatten through this function
    instead of calling ``.convert("RGB")`` directly on raster input.
    """
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        image = image.convert("RGBA")
        canvas = Image.new("RGB", image.size, background)
        canvas.paste(image, mask=image.split()[-1])
        return canvas
    return image.convert("RGB")
