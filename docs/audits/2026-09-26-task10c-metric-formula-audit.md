# Task10C metric formula audit

Date: 2026-09-26

## Search scope

The repository was searched for the paper source and for explicit definitions
of Eq.(36)--Eq.(41), especially Eq.(38) CD and Eq.(39) GL. The relevant
implementation is [`src/metrics.py`](../../src/metrics.py), whose module
docstring labels ADM/ADSD/CD/GL/RDOA/Ave as Eq.(36)--Eq.(41). Internal reports
such as `docs/STAGE1_BASELINE_REPORT.md` contain metric results but do not
provide the paper equations or a cited reference implementation.

No paper PDF, supplementary formula source, equation image, or independent
trusted implementation was found in the repository. The search therefore
cannot uniquely verify that the current histogram total-variation CD or the
current Sobel orientation GL is the paper's Eq.(38)/(39).

## Provisional implementation interpretation

The current code interprets:

- CD as a 0.5-scaled total-variation distance between normalized histograms;
- GL as the mean absolute difference of Sobel gradient orientations over
  jointly valid pixels;
- RDOA and Ave as arithmetic means of their component diagnostics.

These are implementation descriptions only, not verified paper-metric claims.
The Task10C real-science gate must not label these values as verified paper
metrics until the missing source material is supplied. This is a hard-stop
condition under the Task10C plan; no formula is inferred from the function
names or existing output summaries.
