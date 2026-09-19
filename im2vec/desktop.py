"""Desktop launcher: the web app in a native window.

Starts the FastAPI app (:mod:`im2vec.app`) on a free loopback port and opens
it in a pywebview window, so the tool runs as an ordinary double-clickable
desktop app — offline, no browser, no terminal. This is the entry point the
PyInstaller build packages (``packaging/im2vec.spec``).

Run from source with::

    python -m im2vec.desktop

``--self-test`` starts the bundled server, converts a generated image and
exits 0 or 1 without opening a window. CI runs it against each frozen build,
so a packaging mistake (a missing module, unbundled web assets) fails the
build instead of reaching users.
"""

from __future__ import annotations

import io
import json
import os
import socket
import sys
import threading
import time
from pathlib import PurePath
from typing import Callable, Optional, Sequence, Union

import uvicorn

from .app import app

WINDOW_TITLE = "Im2Vec — Logo to SVG"
WINDOW_SIZE = (1100, 720)
_START_TIMEOUT_S = 15

DialogResult = Union[str, Sequence[str], None]


def find_free_port() -> int:
    """Ask the OS for an unused loopback port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LocalServer:
    """The FastAPI app on a background thread, bound to loopback only."""

    def __init__(self, port: Optional[int] = None):
        self.port = port or find_free_port()
        # log_config=None: uvicorn's default logging config writes to
        # sys.stdout, which is None in a windowed (no-console) build.
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_config=None, log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> "LocalServer":
        self._thread.start()
        deadline = time.monotonic() + _START_TIMEOUT_S
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"Local server failed to start on port {self.port}")
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(_START_TIMEOUT_S)

    def __enter__(self) -> "LocalServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


class DesktopApi:
    """Methods exposed to the page as ``window.pywebview.api``.

    Embedded webviews do not reliably honour ``<a download>``, so the page
    saves through a native Save dialog instead when running on the desktop.
    """

    def __init__(self, choose_path: Callable[[str], DialogResult]):
        # Leading underscore: pywebview does not expose private attributes to JS.
        self._choose_path = choose_path

    def save_svg(self, svg: str, filename: str) -> Optional[str]:
        """Ask where to save, write the SVG there; ``None`` if cancelled."""
        name = PurePath(filename.replace("\\", "/")).name or "logo"
        if not name.lower().endswith(".svg"):
            name += ".svg"

        chosen = self._choose_path(name)
        if chosen and not isinstance(chosen, str):
            chosen = chosen[0] if len(chosen) else None
        if not chosen:
            return None

        if not chosen.lower().endswith(".svg"):
            chosen += ".svg"
        with open(chosen, "w", encoding="utf-8") as f:
            f.write(svg)
        return chosen


def self_test() -> bool:
    """Exercise the bundled app end to end; True if everything works."""
    import urllib.request

    from PIL import Image

    buf = io.BytesIO()
    image = Image.new("RGB", (64, 64), (255, 255, 255))
    image.paste(Image.new("RGB", (32, 32), (200, 30, 30)), (16, 16))
    image.save(buf, "PNG")

    boundary = "im2vec-self-test"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="self-test.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()

    try:
        with LocalServer() as server:
            with urllib.request.urlopen(server.url, timeout=10) as resp:
                if "Logo to SVG" not in resp.read().decode():
                    return False
            with urllib.request.urlopen(server.url + "static/app.js", timeout=10) as resp:
                if "pywebview" not in resp.read().decode():
                    return False
            request = urllib.request.Request(
                server.url + "api/convert",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            with urllib.request.urlopen(request, timeout=30) as resp:
                svg = json.loads(resp.read())["svg"]
            return svg.startswith("<svg") and "<path" in svg
    except Exception:  # noqa: BLE001 - any failure means the build is broken
        return False


def _ensure_std_streams() -> None:
    """Windowed builds have no console: give print/logging somewhere to go."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))


def main() -> None:
    _ensure_std_streams()
    if "--self-test" in sys.argv[1:]:
        sys.exit(0 if self_test() else 1)

    import webview  # imported here so the module stays importable headless

    window_ref: list = []

    def choose_path(name: str) -> DialogResult:
        return window_ref[0].create_file_dialog(
            webview.FileDialog.SAVE,
            save_filename=name,
            file_types=("SVG image (*.svg)",),
        )

    with LocalServer() as server:
        window = webview.create_window(
            WINDOW_TITLE,
            server.url,
            js_api=DesktopApi(choose_path),
            width=WINDOW_SIZE[0],
            height=WINDOW_SIZE[1],
            min_size=(720, 520),
        )
        window_ref.append(window)
        webview.start()


if __name__ == "__main__":
    main()
