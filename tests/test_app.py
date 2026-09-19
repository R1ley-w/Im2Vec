"""Tests for the FastAPI web app (im2vec.app)."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client():
    from im2vec import app as app_module

    with TestClient(app_module.app) as c:
        yield c


def _png_bytes(size=(64, 64), colour=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format="JPEG", quality=20)
    return buf.getvalue()


def test_index_serves_frontend(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Logo to SVG" in resp.text


def test_convert_returns_svg(client):
    resp = client.post(
        "/api/convert", files={"file": ("logo.png", _png_bytes(), "image/png")}
    )
    assert resp.status_code == 200
    assert "<svg" in resp.json()["svg"]


def test_convert_reports_the_real_image_dimensions(client):
    resp = client.post(
        "/api/convert",
        files={"file": ("logo.png", _png_bytes(size=(123, 45)), "image/png")},
    )
    body = resp.json()
    assert body["width"] == 123
    assert body["height"] == 45


def test_convert_accepts_jpeg(client):
    resp = client.post(
        "/api/convert", files={"file": ("logo.jpg", _jpeg_bytes(), "image/jpeg")}
    )
    assert resp.status_code == 200
    assert "<svg" in resp.json()["svg"]


def test_convert_rejects_wrong_type(client):
    resp = client.post(
        "/api/convert", files={"file": ("note.txt", b"hello", "text/plain")}
    )
    assert resp.status_code == 415


def test_convert_rejects_corrupt_image(client):
    resp = client.post(
        "/api/convert", files={"file": ("bad.png", b"not-an-image", "image/png")}
    )
    assert resp.status_code == 422


def test_convert_rejects_oversized_upload(client):
    from im2vec.app import MAX_UPLOAD_BYTES

    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * MAX_UPLOAD_BYTES
    resp = client.post(
        "/api/convert", files={"file": ("big.png", oversized, "image/png")}
    )
    assert resp.status_code == 413


def test_app_starts_without_a_checkpoint(client):
    """The tracer needs no model, so there is no cold start and no download."""
    assert not hasattr(client.app.state, "model")


def test_app_does_not_require_torch():
    import sys

    assert "torch" not in sys.modules


def test_stylesheet_lets_the_hidden_attribute_win(client):
    """Regression for #8: app.js toggles UI with the `hidden` attribute, and
    elements with their own `display` rule ignore it unless this rule exists.
    Without it the spinner stays on and empty panels show."""
    import re

    css = client.get("/static/style.css").text
    rule = re.search(r"\[hidden\]\s*\{([^}]*)\}", css)
    assert rule, "style.css needs a [hidden] rule"
    assert re.search(r"display\s*:\s*none\s*!important", rule.group(1))
