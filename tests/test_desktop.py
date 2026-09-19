"""Tests for the desktop launcher (im2vec.desktop).

Everything here runs headless: the window itself is pywebview's job, so these
cover what the launcher owns — the local server, and saving through the
native dialog.
"""

import io
import json
import socket
import sys
import urllib.request

from PIL import Image

from im2vec.desktop import DesktopApi, LocalServer, find_free_port


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def _post_image(url: str, data: bytes) -> dict:
    boundary = "im2vec-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="logo.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def test_find_free_port_returns_a_bindable_port():
    port = find_free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_local_server_serves_the_frontend_and_converts():
    with LocalServer() as server:
        with urllib.request.urlopen(server.url, timeout=10) as resp:
            assert "Logo to SVG" in resp.read().decode()
        body = _post_image(server.url + "api/convert", _png_bytes())
        assert "<svg" in body["svg"]


def test_local_server_binds_to_loopback_only():
    with LocalServer() as server:
        assert server.url.startswith("http://127.0.0.1:")


def test_local_server_stops_cleanly():
    server = LocalServer().start()
    port = server.port
    server.stop()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))  # released


def test_save_svg_writes_to_the_chosen_path(tmp_path):
    target = tmp_path / "out.svg"
    api = DesktopApi(choose_path=lambda name: str(target))
    saved = api.save_svg("<svg></svg>", "logo.svg")
    assert saved == str(target)
    assert target.read_text(encoding="utf-8") == "<svg></svg>"


def test_save_svg_returns_none_when_cancelled(tmp_path):
    api = DesktopApi(choose_path=lambda name: None)
    assert api.save_svg("<svg></svg>", "logo.svg") is None


def test_save_svg_suggests_an_svg_filename():
    suggested = []
    api = DesktopApi(choose_path=lambda name: suggested.append(name) or None)
    api.save_svg("<svg></svg>", "my logo")
    assert suggested == ["my logo.svg"]


def test_save_svg_strips_path_components_from_the_suggestion():
    suggested = []
    api = DesktopApi(choose_path=lambda name: suggested.append(name) or None)
    api.save_svg("<svg></svg>", "../../etc/evil.svg")
    assert suggested == ["evil.svg"]


def test_save_svg_accepts_dialogs_that_return_a_sequence(tmp_path):
    # pywebview returns a tuple on some platforms and a str on others
    target = tmp_path / "out.svg"
    api = DesktopApi(choose_path=lambda name: (str(target),))
    assert api.save_svg("<svg/>", "logo.svg") == str(target)
    assert target.exists()


def test_save_svg_appends_the_extension_if_the_user_drops_it(tmp_path):
    target = tmp_path / "chosen"
    api = DesktopApi(choose_path=lambda name: str(target))
    saved = api.save_svg("<svg/>", "logo.svg")
    assert saved.endswith("chosen.svg")


def test_importing_the_launcher_does_not_open_a_gui():
    assert "webview" not in sys.modules


def test_self_test_passes_against_a_working_app():
    from im2vec.desktop import self_test

    assert self_test() is True


def test_self_test_fails_when_conversion_is_broken(monkeypatch):
    from im2vec import desktop

    def broken(*args, **kwargs):
        raise RuntimeError("tracer missing from bundle")

    monkeypatch.setattr("im2vec.app.trace_svg", broken)
    assert desktop.self_test() is False


def test_main_self_test_flag_exits_without_a_window(monkeypatch):
    import pytest

    from im2vec import desktop

    monkeypatch.setattr(sys, "argv", ["Im2Vec", "--self-test"])
    with pytest.raises(SystemExit) as exc:
        desktop.main()
    assert exc.value.code == 0
    assert "webview" not in sys.modules
