"""FastAPI web app: drag-and-drop raster-to-SVG converter.

Serves a small static frontend (``im2vec/web``) and a ``POST /api/convert``
endpoint that traces an uploaded image with vtracer and returns an SVG.

Tracing is deterministic, CPU-only and takes milliseconds, so there is no
model to load, no checkpoint to download and no cold start.

Run with::

    python -m im2vec.app

or::

    uvicorn im2vec.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .tracer import trace_svg

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Logo to SVG")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/convert")
def convert(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, "Only PNG, JPEG, or WebP images are supported")

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is too large (max 10 MB)")

    try:
        svg = trace_svg(data)
        width, height = Image.open(io.BytesIO(data)).size
    except ValueError as exc:
        raise HTTPException(422, f"Could not process the image: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        raise HTTPException(422, f"Could not process the image: {exc}") from exc

    return {"svg": svg, "width": width, "height": height}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Logo to SVG web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
