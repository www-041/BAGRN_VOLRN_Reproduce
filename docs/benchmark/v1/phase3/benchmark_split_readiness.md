# Benchmark V1 Phase3 readiness

**Four-pair trial split frozen; full 15-pair Benchmark not frozen.** All 160 first-review-approved source candidates remain intact.

- DEVELOPMENT: 43; TEST candidates: 84; UNASSIGNED: 33 (all in excluded 500 m buffer zones). Assigned-only shares: 33.9% / 66.1%. No quota-driven reallocation.
- Formal TEST scoring eligible: **0**. Distinct second-reviewer checks outstanding: **84**. A split is not final approval.
- Leakage audit: **PASS**; assigned-point safe-core violations 0; opposite-role cross-pair proximity conflicts 0; minimum opposite-role left/right projected-point distance 1425.665 m.
- Overall distinct assigned global cells: 38; role-specific DEV/TEST cells: 14/24. Cell distributions and concentration flags are in `split_leakage_audit.json`.
- The eight trial-scene B9 SHA256 and sizes match the Task15-recorded source identities. Remaining scene indices [4, 6, 9, 11, 12] are **not yet reverified** and must be checked before full 13-scene formal freeze.
- Original B1–B16 multiband identities are not frozen by this B9-only check. TEST second review, additional 11-pair annotations, full input version check and final truth export remain required.
- Spatial map: `spatial_split_map.png`; grid is a preselected evaluation design, not tuned on matcher results.

| Pair | DEV | TEST candidates | UNASSIGNED | all assigned cells | DEV cells | TEST cells |
|---|---:|---:|---:|---:|---:|---:|
| 00_01 | 11 | 19 | 10 | 7 | 3 | 4 |
| 02_03 | 9 | 22 | 9 | 17 | 6 | 11 |
| 05_07 | 9 | 26 | 5 | 18 | 6 | 12 |
| 08_10 | 14 | 17 | 9 | 18 | 7 | 11 |
