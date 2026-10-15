# R1.5 run index

Canonical four-pair candidate set for manual review: `r15_20261012_04`.

- `r15_20261012_01`: initial 8×8 center-window screen; no windows passed the fixed edge-buffer gate. Preserved as a zero-candidate diagnostic; no thresholds were relaxed.
- `r15_20261012_02`: denser within-cell deterministic search used for the 00_01 GUI smoke check. Preserved as an intermediate smoke artifact.
- `r15_20261012_03`: same denser search with per-band mask support, all four pairs, aggregate report, spatial map, and cross-pair audit. Preserved as an intermediate artifact.
- `r15_20261012_04`: same selection gates/order and candidate coordinates as the previous full run, with bounded per-band median/MAD diagnostics added for GUI context. Use this run for human review.

All candidate sources remain immutable and `UNVERIFIED`. None of these runs constitutes radiometric ground truth or an R2 experiment.
