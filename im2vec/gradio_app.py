"""Gradio app for the raster-to-SVG converter (Hugging Face Space).

Drag-and-drop a PNG/JPEG logo and download the traced SVG.

Conversion is classical vector tracing via vtracer — deterministic, CPU-only
and near-instant. There is no model, so the Space needs no GPU hardware and
has no cold start. See ``im2vec/tracer.py`` and
``docs/adr/0001-classical-tracing-over-generative-model.md``.
"""

from __future__ import annotations

import os
import tempfile
import urllib.parse

import gradio as gr

from .tracer import trace_svg

CHECKER_BG = (
    "background-image:linear-gradient(45deg,#e5e7eb 25%,transparent 25%),"
    "linear-gradient(-45deg,#e5e7eb 25%,transparent 25%),"
    "linear-gradient(45deg,transparent 75%,#e5e7eb 75%),"
    "linear-gradient(-45deg,transparent 75%,#e5e7eb 75%);"
    "background-size:20px 20px;"
    "background-position:0 0,0 10px,10px -10px,-10px 0;"
)


def _render_preview(svg: str) -> str:
    """Embed the SVG in a checkered HTML preview (preserves transparency)."""
    encoded = urllib.parse.quote(svg)
    return (
        f"<div style='{CHECKER_BG} border-radius:8px; "
        "display:flex;align-items:center;justify-content:center;"
        "min-height:320px;padding:16px;'>"
        f"<img src='data:image/svg+xml;utf8,{encoded}' "
        "style='max-width:100%;max-height:420px;'/>"
        "</div>"
    )


def convert(image_path: str | None) -> tuple[str, str | None]:
    """Trace an uploaded raster; returns (preview HTML, svg file path)."""
    if not image_path:
        raise gr.Error("Please upload an image first.")

    with open(image_path, "rb") as f:
        data = f.read()

    try:
        svg = trace_svg(data)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    tmpdir = tempfile.mkdtemp(prefix="logo-to-svg_")
    svg_path = os.path.join(tmpdir, "logo.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return _render_preview(svg), svg_path


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Logo to SVG") as demo:
        gr.Markdown("# Logo to SVG")
        gr.Markdown(
            "Drop a PNG or JPEG logo to convert it into an editable SVG "
            "vector file. Conversion is instant — no model, no queue."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    # "filepath" hands over the uploaded file untouched, which
                    # matters twice: Gradio's own RGB conversion does a plain
                    # `.convert("RGB")` (blackening transparent backgrounds),
                    # and decoding to a PIL image would discard the format the
                    # tracer branches on. See `tracer.params_for_format`.
                    type="filepath",
                    label="Upload logo",
                    sources=["upload"],
                )
                convert_btn = gr.Button("Convert", variant="primary")
            with gr.Column():
                preview = gr.HTML(label="Preview")
                download = gr.DownloadButton("Download .svg")

        convert_btn.click(convert, inputs=image, outputs=[preview, download])

    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
