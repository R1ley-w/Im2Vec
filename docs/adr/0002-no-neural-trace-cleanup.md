---
status: accepted
---

# No neural trace cleanup

[ADR 0001](0001-classical-tracing-over-generative-model.md) left one opening
for deep learning: *cleanup* of messy traces of compressed images, and "only
where measurement shows the classical frontier leaves room". Phase 2
([#2](https://github.com/R1ley-w/JPG-to-SVG-web-tool/issues/2)) took that
measurement against the gates fixed in advance in
[#3](https://github.com/R1ley-w/JPG-to-SVG-web-tool/issues/3). The room is
not there, so Phase 3 is discarded. Full results:
[`docs/phase2/results.md`](../phase2/results.md).

## What was measured

On 1,960 pairs (the valid split's ladder: 392 sources x JPEG q15-q95):

- **Gate 1 fails.** Gate 1 asks for >25% token reduction over the classical
  frontier at equal-or-better RMSE. Measured per image, against a frontier
  over 72 tracer settings, the median is **2%** and only 22% of pairs clear
  25%.
- **Gate 2 passes, but it does not matter.** 73% of excess paths can be
  dropped or merged. Yet an oracle that drops every droppable path is no
  cheaper than the frontier for 95% of pairs: the tokens are in the jagged
  boundaries of the paths that remain, not in the speckle.

## Why gate 1 is read per image

The clean trace usually has better fidelity than any tracer setting
reaches, so "equal-or-better RMSE" needs some RMSE slack before anything
can be compared. A single curve across all images, as in Phase 1, is so
steep near high fidelity that its headroom swings from 43% to -1% as the
slack goes from +1 to +3 RMSE: choosing a slack value decides the result.
Per image, the answer barely moves across that range. The gate is judged
on the stable measure.

## Why not re-scope to heavily compressed input

At q15 the per-image median headroom is 28%, which would pass. It is not
taken up, for three reasons:

- **The band would be chosen after seeing the results**, which is what fixed
  thresholds exist to prevent.
- **That headroom needs regenerated geometry.** Even at q15 a perfect
  drop-only classifier is 64% worse than the frontier (median). Capturing
  the gap means redrawing boundaries: the long-sequence generation 0001
  rules out (q15 messy traces are ~8k tokens at the median).
- **28% is an upper bound.** It measures against the clean trace, which uses
  information the JPEG destroyed.

## Consequences

- The tracer stays the whole conversion path. The shipped JPEG branch
  (`filter_speckle=16, color_precision=5`) is close to the frontier:
  choosing the best of 72 settings for each image, with the source raster
  in hand, saves only 2.8% of tokens at median, and it is already the best
  single setting at q15 and q30. Neither a model nor a per-image settings
  selector is worth building.
- The trace-pair dataset stays local and is not published: it existed to
  train a model. That also removes the licensing review it needed.
- The Phase 2 measurement tools (`im2vec/pairs.py`, `frontier.py`,
  `excess.py`, `im2vec/data/build_pairs.py`, `analyze_pairs.py`) are kept.
  They are how any future claim about trace quality should be checked.
- #3's premise, that the clean trace sits well inside the classical curve,
  came from a 4-setting sweep and a single curve across all images. It
  does not survive a wider sweep.

## What would reopen this

Not a reinterpretation of these gates. Nearly every caveat pushes further
toward failure; a wider sweep, for instance, can only improve the frontier.
The exception is the fidelity metric: **RMSE barely registers jagged
boundaries**, the defect that carries the tokens. If perceived edge quality
comes to matter more than RMSE, that is a new metric and a new gate fixed
in advance, decided again by measurement.
