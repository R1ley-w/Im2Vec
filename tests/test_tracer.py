import io

import numpy as np
import pytest
from PIL import Image

from im2vec.tokenizer import SVGTokenizer
from im2vec.tracer import (
    DEFAULT_PARAMS,
    JPEG_PARAMS,
    params_for_format,
    trace_svg,
)


def _square(mode: str = "RGB", bg=(255, 255, 255)) -> Image.Image:
    img = Image.new(mode, (128, 128), bg)
    inner = Image.new(mode, (64, 64), (255, 0, 0) if mode == "RGB" else (255, 0, 0, 255))
    img.paste(inner, (32, 32))
    return img


def _busy(size: int = 128, block: int = 16) -> Image.Image:
    """Multi-colour blocks: busy enough that JPEG actually leaves artefacts."""
    rng = np.random.default_rng(0)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, block):
        for x in range(0, size, block):
            arr[y : y + block, x : x + block] = rng.integers(0, 255, 3)
    return Image.fromarray(arr)


def _encode(img: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def test_traces_png_to_valid_svg():
    svg = trace_svg(_encode(_square(), "PNG"))
    assert svg.lstrip().startswith("<svg")
    assert "<path" in svg
    assert svg.rstrip().endswith("</svg>")


def test_traced_svg_is_parseable_by_the_tokenizer():
    svg = trace_svg(_encode(_square(), "PNG"))
    tokens = SVGTokenizer().encode_svg(svg)
    assert len(tokens) > 2  # more than just SOS/EOS


def test_traced_shape_keeps_its_colour():
    svg = trace_svg(_encode(_square(), "PNG"))
    assert "#ff0000" in svg.lower()


def test_root_carries_a_viewbox_so_the_output_scales():
    svg = trace_svg(_encode(_square(), "PNG"))
    root = svg[: svg.index(">") + 1]
    assert "viewBox=" in root


def test_output_has_no_xml_prolog_so_it_can_be_inlined():
    svg = trace_svg(_encode(_square(), "PNG"))
    assert "<?xml" not in svg
    assert "<!--" not in svg


def test_transparent_png_does_not_trace_to_black():
    """Regression: .convert('RGB') on transparent input used to blacken it."""
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (64, 64), (255, 0, 0, 255)), (32, 32))
    svg = trace_svg(_encode(img, "PNG"))
    assert "#000000" not in svg.lower()


def test_jpeg_input_selects_tuned_params():
    assert params_for_format("JPEG") == JPEG_PARAMS
    assert params_for_format("jpeg") == JPEG_PARAMS


def test_png_and_webp_select_defaults():
    assert params_for_format("PNG") == DEFAULT_PARAMS
    assert params_for_format("WEBP") == DEFAULT_PARAMS


def test_unknown_format_falls_back_to_defaults():
    assert params_for_format(None) == DEFAULT_PARAMS
    assert params_for_format("TIFF") == DEFAULT_PARAMS


def test_jpeg_branch_yields_a_more_compact_trace_than_defaults():
    """The whole point of the format branch: JPEG artefacts fragment the trace."""
    noisy = _encode(_busy(), "JPEG", quality=15)
    tok = SVGTokenizer()
    branched = len(tok.encode_svg(trace_svg(noisy)))
    untuned = len(tok.encode_svg(trace_svg(noisy, params=DEFAULT_PARAMS)))
    assert branched < untuned


def test_explicit_params_override_the_format_branch():
    noisy = _encode(_busy(), "JPEG", quality=15)
    tok = SVGTokenizer()
    forced = len(tok.encode_svg(trace_svg(noisy, params=DEFAULT_PARAMS)))
    auto = len(tok.encode_svg(trace_svg(noisy)))
    assert forced != auto


def test_rejects_input_that_is_not_an_image():
    with pytest.raises(ValueError):
        trace_svg(b"this is not an image")


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        trace_svg(b"")


def test_tracer_does_not_require_torch():
    import sys

    assert "torch" not in sys.modules
