# LSSTAA numerical diagnostic and DZ01V three-scene exploration

Branch: `research/lsstaa-python-20261014` in `.lsstaa_isolated_20261014`. Official source: `lin-yan-rs/LSSTAA@6bd81a1374927eecf5fd8b3843fa470e2e6c055d`. The previous `lsstaa_three_scene_gate_report.md` remains unchanged.

## Four separate outcomes

| Gate | Outcome |
| --- | --- |
| Official `data1` numerical equivalence, coefficient tolerance ±0.0005 | `OFFICIAL_REFERENCE_MISMATCH` |
| Python foundational checks | `PASS` (21 targeted tests; known affine-network gain/offset, control identity, observation sign/rank, dates, PIF edge semantics, mask/grid/application and frozen-score fixture) |
| Fixed DZ01V 00/01/02, 10× estimation, full-resolution B9 application | `SUCCESS_DIAGNOSTIC`, method `LSSTAA_PY_ADAPTED_DN_PROXY_UNVERIFIED` |
| Full 13-scene Benchmark | `NOT_RUN` |

## Official discrepancy, bounded investigation

`Get_dNDVI_map_v1.m` uses MATLAB `int16` rounding (ties away from zero) and saturation; `GetTemporalDiff_threshold.m` uses `hist(values,50)` with equality at an internal bin edge assigned to the left bin. `Select_PIF_subset.m` computes 1%/99% ranges before removing zero samples; `GetGridSamples_v2.m` uses `ceil`, 102×102 cells and strict `> half` median-cell scanning. Source-backed Python corrections and tests cover these boundaries. No final-coefficient-driven threshold changes were made.

The final official sample has PIF count 71,831, temporal threshold 108.112245, trimmed histogram support 571,067 with modal bin 9 (zero-based), control scene 3 (zero-based 2), and 44 graph edges in each band. The 44 edge subset sizes and graph matrices are in `official_numeric_final_20261015.json`; total selected observations are 8,726 / 8,708 / 8,761 / 8,800 for bands 1–4. Each 18-unknown system has rank 18. Normal-matrix condition numbers are 4.23×10⁷ / 8.47×10⁷ / 1.16×10⁸ / 2.52×10⁹. Direct least squares on **the same equations** differs from the normal-equation answer by at most 2.83×10⁻¹⁰ in offset; the solver is therefore not the observed source of the reference mismatch.

| Official sample band | Maximum gain error | Maximum offset error |
| --- | ---: | ---: |
| 1 | 0.005861 | 3.5934 |
| 2 | 0.005169 | 3.4479 |
| 3 | 0.005416 | 3.2805 |
| 4 | 0.017580 | 34.0698 |

The author archive supplies rounded coefficients but no intermediate PIF mask, threshold, graph or sampled ties. Thus the remaining discrepancy cannot yet be assigned definitively to upstream PIF/reference generation; official numerical equivalence must **not** be claimed. The fixed ±0.0005 tolerance was not relaxed.

## Fixed real experiment

Scenes 00 and 01 are 2025-11-14; scene 02 is 2025-11-20 (absolute offsets 0, 0, 6 days). Input band order is `[B9, B4, B7, B14]`, with B7/B14 used **only as an uncalibrated DN-ratio proxy**, not physical NDVI. The three source bands in each scene have matching native metadata grids; this is not independent evidence of subpixel band registration. Tenfold coarse estimation excludes NoData from means; original canonical B9 data and frozen masks are never modified. The official 14-day connection rule yields 3/3 edges. There are 3,639 PIF pixels. Control=01, selected B9 ties 196/190/161, rank=4/4, normal condition=3.24×10⁹, pre-adjustment RMSE=341.287.

| Scene | B9 gain | B9 offset | Full-resolution valid pixels |
| --- | ---: | ---: | ---: |
| 00 | 1.006874538 | −31.133587 | 10,752,209 |
| 01 (control) | 1.000000000 | 0 | 10,327,283 |
| 02 | 0.955179737 | +562.047227 | 9,988,472 |

Frozen B1 support is unchanged. All values below use the existing frozen metric implementations and exact empirical RDD, not a histogram approximation. Unit remains `UNKNOWN`.

| Pair (valid pixels) | RAW MAE → corrected | RAW RMSE → corrected | RAW MAMD → corrected | RAW MSDD → corrected | RAW RDD → corrected |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00_01 (1,295,654) | 63.128 → **63.753** | 103.759 → **104.308** | 0.00069 → **5.915** | 0.160 → **2.302** | 0.427 → **6.099** |
| 00_02 (1,503,254) | 452.633 → 91.575 | 470.467 → 126.332 | 451.250 → 15.216 | 41.471 → 18.639 | 451.268 → 23.714 |
| 01_02 (642,374) | 428.966 → 98.184 | 448.979 → 134.484 | 426.653 → 0.0773 | 25.856 → 7.145 | 426.674 → 14.239 |

Frozen B2 windows: 32 / 32 / 29 (93 total), every support SHA matched. Median window MAE is 55.990→56.317, 456.856→86.730, and 401.700→92.054 respectively; median RDD is 2.249→6.664, 455.263→46.997, and 399.510→41.535. Per-window seven metrics, structure and range diagnostics are retained in JSON. A separate full-pixel audit confirmed three grids/masks identical to frozen inputs, NaN outside support and exact `float32(gain × RAW + offset)` at every valid pixel.

The v2 run took 87.21 s including frozen B1/B2 scoring; sampled peak RSS was 714,739,712 bytes over 417 samples. Preflight reported ~2.10 GB RAM available and ~20.10 GB disk free. Parameter-stack loading took 20.30 s, PIF/joint solve 0.45 s, three writes 43.69 s, and three pair scoring passes 20.83 s; scoring is not algorithm-solving time. Output checks and hashing were separate post-run operations.

## Artifacts and safeguards

Final run directory: `data/output/benchmark_v1_lsstaa/three_scene_exploratory_20261015_v2/`. It contains `corrected/scene_000..002_B9_lsstaa_float32.tif` and matching `scene_000..002_valid_mask.tif`, `run_status.json`, `b1_b2_scores.json`, `structure_diagnostics.json`, `input_code_attestation.json`, and `output_audit.json`. Final official comparison: `data/output/benchmark_v1_lsstaa/official_numeric_final_20261015.json`. Earlier exploratory and official attempts remain separate and were not overwritten.

Run preflight verified frozen lock 54/54 and isolated code manifest 38/38. Input attestation records SHA256 for all 12 source V-band files, six frozen B9/mask files, the lock/window files, geometry manifests, and the exact uncommitted implementation code used. Targeted tests: 21 passed; `compileall` and `git diff --check`: pass. No full-repository regression suite was run for this isolated phase. No Task15/16, Ours, Unified Benchmark frozen files, original images or previous results were changed; no Git commit or push was requested or made.

This is a **three-scene exploratory diagnostic**, not an official-equivalent LSSTAA reproduction and not a 13-scene ranking result. Stop here pending researcher review; do not tune against these B1/B2 scores.
