# Im2Vec

Converts raster logos and icons into editable SVG vector graphics. The domain
is the quality of a vectorization — what makes one conversion of the same
image better than another.

## Language

### The conversion

**Trace**:
An SVG produced from a raster image by a vectorizer. Always a derived
artifact, never hand-authored.
_Avoid_: Conversion, output, prediction, vectorization

**Tracer**:
The component that produces a trace from a raster.
_Avoid_: Model, converter, engine

**Source raster**:
The image a trace is produced from, and the ground truth a trace is judged
against.
_Avoid_: Input, original, target

### Judging a trace

Fidelity and compactness are **independent axes**. A trace is not "good" or
"bad" along a single scale, and any claim about trace quality that names only
one of the two is incomplete.

**Fidelity**:
How closely a trace, re-rendered back to pixels, matches its source raster.
_Avoid_: Accuracy, quality, correctness

**Compactness**:
How few paths a trace uses to achieve its fidelity.
_Avoid_: Size, simplicity, cleanliness, efficiency

**Frontier**:
The set of traces for which no better fidelity is available without losing
compactness, and vice versa. Tracer settings move along it; they do not
escape it.
_Avoid_: Optimum, best settings, sweet spot

### Compression damage

**Messy**:
Describes a trace with excess paths relative to its fidelity — one that sits
inside the frontier rather than on it. A property of a trace, never of a
source raster.
_Avoid_: Noisy, dirty, fragmented, bloated

**Speckle**:
An excess path corresponding to no feature of the source raster, produced by
a vectorizer faithfully reproducing compression artifacts.
_Avoid_: Noise, artifact, fragment, junk path

**Clean trace**:
The trace obtained from an *uncompressed* source raster. Used as the
reference target when judging traces of compressed versions of the same
image, because it shares the tracer's own vocabulary and style. It is a
reference point, **not** a synonym for a compact or high-fidelity trace.
_Avoid_: Ground truth, ideal, target SVG

**Cleanup**:
Moving a messy trace toward the frontier — recovering compactness without
losing fidelity. Not a synonym for simplification, which sacrifices fidelity
to gain compactness and stays on the frontier.
_Avoid_: Denoising, simplification, optimization, repair
