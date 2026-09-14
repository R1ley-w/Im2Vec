# Phase 2 results — trace pairs and the Phase 3 gates

**Status: complete. Phase 3 discarded.**

Issue [#2](https://github.com/R1ley-w/JPG-to-SVG-web-tool/issues/2). Gates
defined in [#3](https://github.com/R1ley-w/JPG-to-SVG-web-tool/issues/3).
Vocabulary: [`CONTEXT.md`](../../CONTEXT.md). Raw numbers:
[`summary.json`](summary.json).

## Verdict

| gate | threshold | measured (valid ladder, 1,960 pairs) | reading |
|---|---|---|---|
| 1. token headroom over the classical frontier at equal-or-better RMSE | >25% | per-image median **2%**; only **22%** of pairs exceed 25% | **fails** (only q15 would pass) |
| 2. share of excess paths droppable/mergeable | >60% | pooled **73%**; per-image median 69% | **passes** at every quality |

Gate 1 is judged per image, the only stable reading (see
[below](#gate-1--frontier-headroom)). Under #3's own rule, "If (1) fails
there is no room for a model". **Phase 3 is discarded**; see
[Outcome](#outcome) and [ADR 0002](../adr/0002-no-neural-trace-cleanup.md).

The gates also disagree in a way that matters for #3's leading candidate,
per-path classification. Excess **paths** are mostly removable (gate 2), but
removing them does not recover **tokens**. The oracle drop-only trace, which
applies every removal a perfect per-path classifier could make, is no
cheaper than the classical frontier at matched fidelity for 95% of the pairs
where the two can be compared, and strictly more expensive for 78%. The token cost of a messy trace is in the jagged boundaries of the
paths that remain, not in the speckle.

## The dataset

`python -m im2vec.data.build_pairs --out data/pairs` (about 14 min on 24 cores).

| | |
|---|---|
| sources | 46,043 SVGs: `svg-emoji` 10,043 + `svg-stack` 36,000, all verified against HF blob hashes |
| planned pairs | 66,043: 5,000 ladder sources x {15,30,50,75,95}, plus 41,043 at one quality drawn uniformly from [15,95] |
| written | **65,586** (24,825 ladder + 40,761 random) |
| failed | 457 pairs from **317 `svg-stack` sources** that cairosvg cannot render. That is exactly the 271 + 22 + 24 missing PNGs recorded in the source dataset card, so it is the known broken set, not new breakage. |
| size | 2.2 GB, zstd parquet, 37 shards under `<dataset>/<split>/` |
| loose sample | 50 pairs in `data/pairs-loose/`, spread across q15-q95 |

Every row keeps the source SVG, clean PNG, JPEG, both traces, and, for each
trace, token count, path count, RMSE and SSIM against the source raster.
`metadata.json` records the seed, the ladder and the tracer and library
versions. Both traces use vtracer defaults. The tracer's format branch is
bypassed on purpose, because it would filter out the very speckle the pairs
exist to capture.

Stratification: the ladder subset is proportional over (dataset, split,
source-size quartile), chosen by a seeded hash of the source key.

## Compression -> messiness

Median over the 4,965 ladder sources (all splits), which are the same images
at every quality:

| JPEG q | messy/clean tokens | messy/clean paths | messy tokens | JPEG damage (RMSE) | messy - clean trace RMSE |
|---|---|---|---|---|---|
| 15 | 6.16x | 9.0x | 7,995 | 7.30 | +1.70 |
| 30 | 4.56x | 7.0x | 6,159 | 5.41 | +0.83 |
| 50 | 3.58x | 5.3x | 4,792 | 4.26 | +0.42 |
| 75 | 2.47x | 3.4x | 3,360 | 2.98 | +0.16 |
| 95 | 1.14x | 1.2x | 1,502 | 1.53 | +0.03 |

Tokens blow up far faster than fidelity drops: at q50 the messy trace is
3.6x the size for 0.4 RMSE. RMSE barely registers a jagged boundary (see
[caveats](#caveats)), and that jagged boundary is where the tokens go.

## Measurement set

Gates were measured on the **valid** split's ladder: 392 sources (72 emoji,
320 svg-stack) x 5 qualities = 1,960 pairs, 0 errors. `test` was held out
for a Phase 3 model and is unused.

`python -m im2vec.data.analyze_pairs --pairs data/pairs --out data/analysis` (about 6 min).

## Gate 2 — excess paths

`im2vec/excess.py`. **Excess** = messy paths - clean paths. Each messy path
is judged against the source raster on an aliasing-free paint map:

- **droppable**: deleting it does not worsen fidelity
- **mergeable**: repainting it in an adjacent or underlying path's colour,
  which has the same fidelity as a union, does not worsen fidelity
- **distorted**: the remaining excess

Removals are greedy and conservative (merge chains are frozen), so these
are **lower bounds**. The repaired traces were re-rendered with
antialiasing, and their RMSE went **down** (median -0.45, p90 0.00), so the
classification holds up against a real render.

| JPEG q | removable (pooled) | per-image median | per-image p10 | dropped | merged | images >60% |
|---|---|---|---|---|---|---|
| 15 | 76.7% | 73.1% | 51.2% | 50.7% | 26.3% | 77.8% |
| 30 | 73.8% | 71.4% | 48.9% | 47.7% | 26.5% | 73.2% |
| 50 | 71.9% | 69.1% | 42.3% | 46.4% | 26.0% | 70.9% |
| 75 | 67.6% | 66.1% | 33.3% | 44.1% | 24.3% | 64.3% |
| 95 | 61.6% | 61.3% | 28.1% | 46.2% | 21.5% | 52.3% |
| **all** | **73.2%** | **69.2%** | 42.9% | 47.9% | 25.9% | 69.2% |

By dataset: emoji 71.3%, svg-stack 74.1% pooled. Gate 2 passes everywhere.

## Gate 1 — frontier headroom

`im2vec/frontier.py`. The frontier comes from tracing each JPEG under 72
settings (`filter_speckle` {4..48} x `color_precision` {6..3} x
`layer_difference` {16,32,64}), which include vtracer defaults and the
shipped JPEG branch. **Headroom** = 1 - reference tokens / cheapest frontier
tokens at equal-or-better RMSE. The reference is the clean trace, the trace
of the uncompressed raster.

### Why "equal-or-better RMSE" needs care

The clean trace has better fidelity than every one of the 72 settings for
**47%** of pairs, and also on the aggregate curve (clean 1,097 tok @ 12.10;
the best setting reaches 3,584 tok @ 12.77). Where the frontier never
reaches the reference's fidelity, strict headroom is undefined. The
reference only exists by using information the JPEG destroyed.

This leaves two ways to read the gate:

- **Aggregate curve** (one setting for every image, as in Phase 1). Its
  headroom depends entirely on how much RMSE slack the frontier gets: **43%**
  at +1.0, **16%** at +1.5, **-1%** at +3.0. The curve is very steep near
  high fidelity (the last 0.4 RMSE costs 2.5x the tokens), so any threshold
  here really picks a slack value. **This view is too unstable to gate on.**
- **Per image** (each pair against its own frontier). Stable across slack:

| JPEG q | reachable at +0 slack | median headroom | >25% | at +1.0 slack: reachable / median / >25% |
|---|---|---|---|---|
| 15 | 29% | **28.1%** | 53% | 52% / 30.1% / 54% |
| 30 | 43% | 15.5% | 34% | 74% / 16.0% / 35% |
| 50 | 56% | 5.8% | 26% | 84% / 5.7% / 24% |
| 75 | 66% | 0.0% | 16% | 94% / 0.0% / 11% |
| 95 | 73% | -2.0% | 3% | 98% / -4.5% / 1% |
| **all** | 53% | **2.1%** | **22%** | 80% / 1.1% / 21% |

The reachable pairs lean toward milder compression, but the per-quality
rows control for that, and widening slack to +3.0 (97% reachable) leaves
the overall median at 0%. For a typical pair, choosing tracer settings per
image already gets within a few percent of the clean trace's token count at
its fidelity. Only heavily compressed input (q15, and partly q30) shows
>25% headroom.

This headroom is also an **upper bound**: the clean trace is not a trace any
model could produce from the JPEG alone.

### A perfect per-path classifier does not beat the frontier

The same table with the **drop-only oracle trace** as the reference (every
droppable path removed; `dropped_only_*` in `summary.json`): overall median
headroom **-30%**, 0.5% of pairs above 25%, median negative at every quality
(q15 -64%, q95 -3%). Headroom is ≤0 for 95% of comparable pairs at +0 slack
and 98% at +3.0. Dropping speckle removes paths, not tokens. Median
tokens go from 3,584 (defaults) to 2,616, still far above frontier points
of similar fidelity.

## Is the shipped JPEG branch good enough?

Yes. With the source raster in hand, the best of the 72 settings chosen
separately for each image beats the shipped branch (`sp16_cp5_ld16`) by
only **2.8%** of tokens at median, at no worse RMSE. The shipped branch is
itself the best single setting at q15 and q30; at q50, q75 and q95 the
best alternative saves 1.7%, 3.0% and 6.5% (`shipped_branch` in
`summary.json`). A per-image
settings selector is not worth building.

## Outcome

Recorded in [ADR 0002](../adr/0002-no-neural-trace-cleanup.md):

1. **Gate 1 is judged per image**, and fails. The aggregate reading is not
   used because the slack value decides it.
2. **Phase 3 is discarded**, not re-scoped to q≤30. The q15 headroom would
   need regenerated boundary geometry (a drop-only oracle is 64% worse than
   the frontier there), it is an upper bound, and picking the band after
   seeing the results defeats gates fixed in advance.
3. **The pair dataset is not published.** It existed to train a model.
   It stays local, reproducible with the commands below.

## Caveats

- **RMSE is nearly blind to jagged boundaries.** A q18 messy trace with a
  visibly ragged rim and 17x the tokens scored 10.1 against the clean
  trace's 10.0. Both gates inherit this, and it favours the frontier:
  high-filter settings look cheap because RMSE does not penalise their
  jaggedness either.
- **The clean trace is not always better.** vtracer sometimes closes thin
  gaps on the lossless PNG while the JPEG trace keeps them, so messy RMSE is
  sometimes lower than clean.
- **The sweep covers three knobs.** `corner_threshold`, `length_threshold`,
  `splice_threshold`, polygon mode and pre-filters (e.g. median blur) are
  not swept. A wider sweep can only improve the frontier, so it can only
  push gate 1 further toward failing.
- **Merges are scored as repaints**, not real geometric unions. Path counts
  after a merge are exact, but a token count for merged traces was not
  measured.
- **Headroom is measured in-sample** on the valid split. It was not tuned
  on it; the only choice made after seeing the data was how to read gate 1,
  explained above.

## Reproduce

```bash
python -m im2vec.data.build_pairs --out data/pairs --loose data/pairs-loose --loose-n 50
python -m im2vec.data.analyze_pairs --pairs data/pairs --out data/analysis
```

Python 3.12. vtracer 0.6.15 segfaults on Python 3.14.
