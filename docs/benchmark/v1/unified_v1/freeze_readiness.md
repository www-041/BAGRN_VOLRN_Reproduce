# Unified Benchmark V1 input-freeze readiness

The canonical input freeze is `data/output/benchmark_v1_unified/freeze_20261013_v3/`.

| Gate | Result |
|---|---|
| 13 current source B9 SHA256 values vs recorded Task15 identity manifest | PASS, 13/13 |
| Task15 canonical grid / 13 transforms | PASS, 13 transforms; EPSG:32650, 14 m, 7992×12404 |
| Fresh aligned unnormalized inputs | PASS, generated in the isolated output directory only |
| Fresh aligned masks vs all Task15 stage-06 input masks | PASS, exact pixel equality required per scene |
| Canonical overlap inventory | PASS, 78 rows; 48 positive, 30 zero-support |
| Native vs aligned support distinction | PASS, both counts retained; `00_01` differs by 3,982 pixels; `00_10` equal |
| B2 windows | PASS, 1,533 deterministic windows, selected only from input masks and grid locations |
| Metric specification and report schema | PASS; paper CD/GL explicitly unverified |
| Metric spec, synthetic fixture + generator provenance, source/mask manifests | PASS; their hashes are in the freeze lock |
| Unified evaluator, CLI, test and protocol source hashes | See `code_manifest.json`; exact files/hashes and base Git branch/HEAD are recorded |
| SHA lock and post-write verification | PASS; `FROZEN`, `lock_verification.valid=true`, 54 locked files |
| External radiometric algorithm runs / real-scene scores | NOT RUN by design |

This is a freeze of the data identity, geometry, masks, support, pair/window populations and metric contract—not a benchmark score, algorithm result, reflectance calibration, stability truth, or seasonal-generalization claim.

The initial blockwise reprojector prototype was rejected because it differed from a persisted Task15 mask by 16,824 pixels for scene 1. The implementation was changed to Task15-equivalent per-scene local-window float64 bilinear reprojection followed by float32 storage. The final freeze proceeded only after all 13 generated masks exactly matched their corresponding persisted Task15 input masks. The early `freeze_candidate_20261013/` is a failed blockwise prototype; `freeze_candidate_20261013_v2/` and `freeze_20261013_v2/` are intermediate freeze attempts without the final code manifest. They are retained for audit, not canonical use.

The earlier directory `freeze_candidate_20261013/` is an interrupted failed prototype and is **not** the freeze. It is retained as audit evidence and must not be used for scoring.
