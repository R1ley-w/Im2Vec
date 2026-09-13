"""Tests for the Gradio app (im2vec.gradio_app)."""

import pytest
from PIL import Image

gr = pytest.importorskip("gradio")

from im2vec.gradio_app import build_demo, convert  # noqa: E402


def _write(tmp_path, name, mode="RGB", colour=(200, 30, 30), fmt=None, **kw):
    path = tmp_path / name
    Image.new(mode, (64, 64), colour).save(path, fmt, **kw)
    return str(path)


def test_convert_returns_preview_and_downloadable_file(tmp_path):
    preview, svg_path = convert(_write(tmp_path, "logo.png"))
    assert "<img" in preview
    assert svg_path.endswith(".svg")
    with open(svg_path, encoding="utf-8") as f:
        assert "<svg" in f.read()


def test_convert_rejects_missing_input():
    with pytest.raises(gr.Error):
        convert(None)


def test_convert_handles_jpeg(tmp_path):
    preview, svg_path = convert(_write(tmp_path, "logo.jpg", fmt="JPEG", quality=20))
    with open(svg_path, encoding="utf-8") as f:
        assert "<svg" in f.read()


def test_transparent_png_does_not_become_black(tmp_path):
    path = tmp_path / "clear.png"
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    img.paste(Image.new("RGBA", (32, 32), (255, 0, 0, 255)), (16, 16))
    img.save(path)
    _, svg_path = convert(str(path))
    with open(svg_path, encoding="utf-8") as f:
        assert "#000000" not in f.read().lower()


def test_demo_builds():
    assert build_demo() is not None


def test_gradio_app_does_not_require_torch():
    import sys

    assert "torch" not in sys.modules
