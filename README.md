# Im2Vec — Logo Raster-to-SVG

Converts a raster logo (PNG/JPEG/WebP) into an editable SVG.

Conversion is **classical vector tracing** via [`vtracer`](https://github.com/visioncortex/vtracer):
deterministic, CPU-only, milliseconds per image, no model and no GPU.

> This project previously used a generative raster→SVG Transformer. It was
> abandoned after measurement — see [Background](#background--the-retired-model)
> below, and [`CONTEXT.md`](CONTEXT.md) for the vocabulary used throughout.

## Download the desktop app

Grab the file for your system from the
[latest release](https://github.com/R1ley-w/Im2Vec/releases/latest) — no
Python or terminal needed. It runs fully offline; images never leave your
machine.

| System | File |
|---|---|
| macOS (Apple Silicon) | `Im2Vec-macOS-AppleSilicon.dmg` — open it and drag **Im2Vec** to Applications |
| Windows 10/11 (64-bit) | `Im2Vec-Windows-x64.exe` — run it directly |
| Linux (x86-64) | `Im2Vec-Linux-x64.tar.gz` — extract, then run `./Im2Vec` |

**First launch warnings.** The builds are not code-signed, so your OS will
warn once:

- **macOS** says the app is from an unidentified developer. Right-click (or
  Control-click) **Im2Vec** → **Open** → **Open**. On recent macOS versions you
  may instead need **System Settings → Privacy & Security → Open Anyway**.
- **Windows** SmartScreen says "Windows protected your PC". Click **More info →
  Run anyway**.
- **Linux** shows no warning.

After the first launch it opens normally.

### Building it yourself

```bash
pip install -r requirements-desktop.txt pyinstaller   # Linux: also pip install "pywebview[qt]"
python -m im2vec.desktop                               # run from source
pyinstaller packaging/im2vec.spec --noconfirm          # build dist/
dist/Im2Vec.app/Contents/MacOS/Im2Vec --self-test      # or dist/Im2Vec(.exe) --self-test
```

PyInstaller can't cross-compile, so each platform is built on that platform.
[`.github/workflows/desktop.yml`](.github/workflows/desktop.yml) builds all
three on every `v*` tag, self-tests each frozen binary, and attaches them to
the release.

## Approach

Tracing trades **fidelity** (how closely the result matches the source pixels)
against **compactness** (how few paths it uses). The best trade depends on
whether the input was lossily compressed: a JPEG's artifacts get faithfully
reproduced as thousands of tiny extra paths, so aggressive speckle filtering
pays for itself — while on an uncompressed image the same filtering only
destroys detail.

Measured over 30 logos at 256×256 (RMSE against the source raster; tokens via
`SVGTokenizer`):

| settings | clean PNG in | JPEG (q15–40) in |
|---|---|---|
| vtracer defaults | **12.67 RMSE / 1,912 tok** | 14.63 RMSE / 14,906 tok |
| `filter_speckle=16, color_precision=5` | 16.93 RMSE / 1,375 tok | **17.43 RMSE / 2,494 tok** |

Tuning costs 4.3 RMSE to save 537 tokens on a PNG (a bad trade) but 2.8 RMSE
to save 12,412 tokens on a JPEG (a good one). So `im2vec/tracer.py` branches on
the detected image format. Format is a *proxy* for compression, not a
measurement of it — a PNG re-encoded from a JPEG gets the lossless branch.

On 10 real logos end-to-end: PNG in → 10.86 RMSE / 0.967 SSIM / median 6 paths;
JPEG q20 in → 12.23 RMSE / 0.939 SSIM / median 6 paths.

## Layout

```
CONTEXT.md           # domain glossary
docs/
  adr/               # architecture decisions
  phase2/            # Phase 2 results and summary.json
im2vec/
  tracer.py          # trace_svg() — the conversion path
  raster.py          # alpha-safe raster normalization
  metrics.py         # fidelity metrics (RMSE / L1 / SSIM / IoU), torch-free
  tokenizer.py       # SVG <-> token sequence; used to measure compactness
  app.py             # FastAPI app + static frontend
  gradio_app.py      # Gradio app
  desktop.py         # desktop launcher: app.py in a native window
  pairs.py           # (clean, messy) trace pair synthesis
  frontier.py        # classical frontier over tracer settings
  excess.py          # droppable / mergeable / distorted excess paths
  data/
    download.py      # fetch SVG datasets from the HF Hub (parquet)
    render.py        # rasterize SVGs -> PNGs
    prepare.py       # one-command download + render
    build_pairs.py   # build the trace-pair dataset (parquet)
    analyze_pairs.py # frontier + excess measurements over the pairs
packaging/           # PyInstaller spec + entry script for the desktop app
tests/
```

## Environment

Everything runs in a plain Python environment — no torch, no CUDA. Use
**Python 3.12**: vtracer 0.6.15's wheel segfaults on Python 3.14.

```bash
pip install -r requirements.txt
```

## Web app (self-hosted)

Two self-hosted frontends, both calling `trace_svg()`. Neither loads a model,
so there is no checkpoint to download and no cold start — a plain CPU box is
enough.

### FastAPI

```bash
python -m im2vec.app --host 0.0.0.0 --port 8000
```

Then open <http://127.0.0.1:8000>. Drop an image on the left; the SVG preview
and download button appear on the right. `POST /api/convert` accepts an
uploaded image and returns `{"svg": ..., "width": ..., "height": ...}`, so it
also works as a plain API.

### Gradio

```bash
python -m im2vec.gradio_app
```

Drag a PNG/JPEG logo in; preview and download the SVG.

### Note on hosting

There is no public demo. Hugging Face now requires a PRO subscription to run
Gradio or Docker Spaces on any hardware including free `cpu-basic` — only
*static* Spaces are free — so the previous Space
(`R1l3y-w/logo-to-svg`) is paused and deprecated. Since tracing needs no GPU
and no model, any small CPU host will run either app above.

If a zero-cost public demo is ever wanted, vtracer has an official
WebAssembly build ([`@visioncortex/vtracer`](https://www.npmjs.com/package/@visioncortex/vtracer))
exposing the same `filterSpeckle`/`colorPrecision` options, which would run
entirely client-side as a free static site.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

## Measuring trace quality

Phase 2 ([#2](https://github.com/R1ley-w/JPG-to-SVG-web-tool/issues/2))
built tooling to measure how JPEG compression bloats traces and how close
tracer settings get to the best possible trade-off. Results are in
[`docs/phase2/results.md`](docs/phase2/results.md). In short, the shipped
JPEG branch is within a few percent of the frontier, and a neural cleanup
model has no room to help
([ADR 0002](docs/adr/0002-no-neural-trace-cleanup.md)).

The tools read the Hugging Face layout of the source datasets,
`data/svg-emoji-hf/` and `data/svg-stack-hf/` (not the flat `data/svg-*/`
directories `prepare` writes):

```python
from huggingface_hub import snapshot_download
snapshot_download("R1l3y-w/im2vec-svg-emoji", repo_type="dataset",
                  local_dir="data/svg-emoji-hf", allow_patterns=["*.svg", "*.md"])
snapshot_download("R1l3y-w/im2vec-svg-stack-sample", repo_type="dataset",
                  local_dir="data/svg-stack-hf", allow_patterns=["*.svg", "*.md"])
```

The PNGs in those repos are not needed. Both repos hold ~46k loose files, so
expect Hugging Face rate limiting (5,000 requests per 5 minutes); re-run the
call if it stops, and completed files are skipped.

```bash
python -m im2vec.data.build_pairs --out data/pairs      # ~65k pairs, ~15 min on 24 cores
python -m im2vec.data.analyze_pairs --pairs data/pairs --out data/analysis
```

## Background — the retired model

Im2Vec began as a generative raster→SVG Transformer: a ResNet encoder pooled an
image into one vector, and a Transformer decoder emitted SVG tokens from it.
Three real bugs were found and fixed along the way (FIGR-8 has no colour
information; `Image.convert("RGB")` blackened transparent backgrounds instead of
compositing; the tokenizer ignored CSS `style="fill:…"`). After all three fixes
and 100 epochs on ~38k coloured examples, it still produced degenerate shapes
with IoU = 0 on a large fraction of validation inputs.

The cause was architectural: a single pooled vector is too weak a conditioning
signal for generating a hundred-plus precise coordinate tokens, and no realistic
amount of data fixes that without changing the architecture. Faithful
reproduction of flat-colour logos, meanwhile, is solved classically — instantly,
deterministically, and with no training.

That code is retired to a gitignored `legacy/` and remains in history at
`f707157`. The old checkpoints stay at
[R1l3y-w/im2vec-logo](https://huggingface.co/R1l3y-w/im2vec-logo) (superseded).

## Background — the SVG datasets

These were sourced to train the retired generative model. Phase 2
([#2](https://github.com/R1ley-w/JPG-to-SVG-web-tool/issues/2)) reused them
as its source of clean SVGs.

| `--dataset` | Source | Rows (train) | Color | License |
|---|---|---|---|---|
| `figr8` | `starvector/FIGR-SVG` | ~1.3M | **monochrome black only** | CC BY-NC 4.0 (non-commercial) |
| `svg-emoji` | `starvector/svg-emoji` | 8,708 | full color | mixed: Twemoji CC-BY 4.0, Noto Emoji Apache-2.0/OFL, OpenMoji CC BY-SA 4.0 (share-alike) |
| `svg-stack` | `starvector/svg-stack` | 2.17M | mostly full color (real logos/icons scraped from GitHub) | mixed: license-filtered permissive GitHub repos (via BigCode's The Stack); per-file provenance untracked |

> **FIGR-8 caveat:** every sampled FIGR-8 icon relies on the SVG default fill
> (black) — there is no color signal anywhere in that dataset.

> **svg-stack caveat:** it's scraped, so ~20% of sampled SVGs tokenize past 512
> tokens and ~3% fail to parse at all.

```bash
python -m im2vec.data.prepare --dataset svg-emoji --split train --out data/svg-emoji/train
python -m im2vec.data.prepare --dataset svg-stack --split train --n 30000 --out data/svg-stack/train
```

`--split` is `train`/`valid`/`test`; `--n` caps the number of SVGs. PNGs are
written alongside the SVGs (same stem) so they pair up automatically.

### Known fix: alpha backgrounds

Every raster load goes through `im2vec.raster.flatten_to_rgb` rather than a
bare `.convert("RGB")`. Plain `Image.convert("RGB")` does **not**
alpha-composite — it discards the alpha channel and exposes whatever RGB
values a renderer stored under fully transparent pixels. cairosvg stores
`(0, 0, 0)` there, so a naive `.convert("RGB")` silently turns every
transparent background solid black. Gradio's own `image_mode="RGB"` coercion
has the same bug, which is why `gradio_app.py` takes `type="filepath"` and
hands the raw upload straight to the tracer. If you add a new image-loading
path, use `flatten_to_rgb`.

## Status

The tracing app is complete, and tracing is the whole conversion path.
Phase 2 measured whether a neural cleanup model could improve traces of
compressed images. It could not beat the classical frontier, so Phase 3 was
discarded
([ADR 0002](docs/adr/0002-no-neural-trace-cleanup.md),
[results](docs/phase2/results.md)).
