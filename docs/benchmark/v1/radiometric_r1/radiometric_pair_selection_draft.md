# R1 radiometric pair selection — draft

This is a pre-method, unverified draft. It uses temporal coverage, existing effective-overlap support, metadata quality flags and deterministic pair-ID tie-breaking only. No Ours/VOLRN/Matcher performance was read. Researcher confirmation is required before freezing.

| Pair | Bin | Days | Valid B9 overlap pixels | Quality i/j | Role | Reason |
|---|---|---:|---:|---|---|---|
| 00_01 | T0 | 6.6e-05 | 1291672 | PASS/PASS | UNVERIFIED | required representative pair |
| 10_11 | T0 | 7.1e-05 | 1443251 | PASS/PASS | UNVERIFIED | stratified by quality/support; no algorithm score |
| 04_06 | T1 | 6.001356 | 6760117 | PASS/PASS | UNVERIFIED | stratified by quality/support; no algorithm score |
| 06_08 | T1 | 0.989259 | 5952019 | PASS/PASS | UNVERIFIED | stratified by quality/support; no algorithm score |
| 02_03 | T2 | 18.005909 | 6600806 | PASS/PASS | UNVERIFIED | stratified by quality/support; no algorithm score |
| 02_08 | T2 | 24.996589 | 5780701 | PASS/PASS | UNVERIFIED | stratified by quality/support; no algorithm score |
| 00_10 | T3 | 61.001974 | 1374076 | PASS/PASS | UNVERIFIED | required representative pair |
| 02_10 | T3 | 54.999443 | 9173415 | PASS/PASS | UNVERIFIED | required representative pair |
| 04_11 | T3 | 36.993539 | 5088227 | PASS/PASS | UNVERIFIED | required representative pair |
| 06_11 | T3 | 30.992183 | 9547029 | PASS/PASS | UNVERIFIED | stratified by quality/support; no algorithm score |

All selected pairs still require human ROI review. A shortlist does not establish stable ground or actual land change.
