---
status: accepted
---

# Classical tracing instead of a generative model

Im2Vec began as a generative raster→SVG Transformer: a ResNet encoder pooled
an image into one vector and a decoder emitted SVG tokens from it. It never
worked. We replaced it with `vtracer`, a deterministic classical vectorizer,
and moved the neural ambition to a narrower, gated problem — see
[#3](https://github.com/R1ley-w/JPG-to-SVG-web-tool/issues/3).

## Why the model was abandoned

Three real bugs were found and fixed along the way (FIGR-8 had no colour
information; `Image.convert("RGB")` blackened transparent backgrounds instead
of compositing; the tokenizer ignored CSS `style="fill:…"`). After all three
fixes and 100 epochs on ~38k coloured examples, the model still produced
degenerate shapes with IoU = 0 on a large fraction of validation inputs.

The cause is architectural, not a matter of data or training time: a single
pooled vector is too weak a conditioning signal for generating a hundred-plus
precise coordinate tokens. No realistic amount of additional data fixes that
without also changing the architecture.

Meanwhile the task itself — faithful reproduction of flat-colour logos — is
solved classically: deterministic, instant, no training, no GPU. Measured on
10 real logos, the tracer achieves 0.967 SSIM with a median of 6 paths.

## Why this is worth recording

A reader finding a `SVGTokenizer`, an HF model repo, and three checkpoints in
this project's history will reasonably ask why there is no neural network in
the conversion path. The answer is that we tried, measured it, and the
classical tool won decisively on the actual problem.

## The consequence that constrains future work

Deep learning is not ruled out — but its remaining opportunity is *cleanup*
(`CONTEXT.md`), not generation, and only where measurement shows the classical
frontier leaves room. Two facts bound that:

- **The obvious formulation does not fit.** Traces of compressed images run to
  a median 11,693 tokens and a maximum of 27,212. The failed model trained at
  512. A token-level seq2seq over those lengths is *harder* than what already
  failed.
- **Part of the gap is unrecoverable.** Fidelity is measured against the
  uncompressed source, so some of it is information the compression destroyed.
  The target is beating the classical frontier, not matching a clean trace.

Phase 3 therefore proceeds only against thresholds fixed in advance
([#3](https://github.com/R1ley-w/JPG-to-SVG-web-tool/issues/3)), so the
decision is made by measurement rather than by attachment to the original idea.
