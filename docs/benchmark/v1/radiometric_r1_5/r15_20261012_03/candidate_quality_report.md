# DZ01V Radiometric ROI candidate quality report — R1.5

Run ID: `r15_20261012_03`

All records are automatic proposals only. `review_status=UNVERIFIED`; no human truth or approval is created.
Units remain UNKNOWN. No radiometric normalization or formal evaluation was run.

Total automatic candidates: 34
Label suggestions: `{'UNKNOWN': 34}`
Confidence suggestions: `{'LOW': 34}`
Spatial roles: `{'DEVELOPMENT': 10, 'TEST': 14, 'UNASSIGNED': 10}`
APPROVED: 0

## Per-pair counts and filters

| Pair | Candidates | Labels | Roles | Examined windows | Quality accepted before cap | Rejected by reason |
|---|---:|---|---|---:|---:|---|
| 00_01 | 2 | {'UNKNOWN': 2} | {'TEST': 1, 'UNASSIGNED': 1} | 1024 | 2 | `{'B9_VALID_SUPPORT_BELOW_0.95': 674, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 348}` |
| 02_03 | 6 | {'UNKNOWN': 6} | {'DEVELOPMENT': 2, 'TEST': 3, 'UNASSIGNED': 1} | 1024 | 6 | `{'B9_VALID_SUPPORT_BELOW_0.95': 399, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 619}` |
| 05_07 | 15 | {'UNKNOWN': 15} | {'DEVELOPMENT': 6, 'TEST': 6, 'UNASSIGNED': 3} | 1024 | 15 | `{'B9_VALID_SUPPORT_BELOW_0.95': 394, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 615}` |
| 08_10 | 11 | {'UNKNOWN': 11} | {'DEVELOPMENT': 2, 'TEST': 4, 'UNASSIGNED': 5} | 1024 | 11 | `{'B9_VALID_SUPPORT_BELOW_0.95': 387, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 626}` |

## Input coverage and quality

| Pair | Days apart | Input quality | Common V bands present | Missing/unusable common bands | B9 support (min/mean) | Units |
|---|---:|---|---:|---|---|---|
| 00_01 | 6.6e-05 | PASS/PASS | 16 | none | 1.000/1.000 | UNKNOWN |
| 02_03 | 18.005909 | PASS/PASS | 16 | none | 1.000/1.000 | UNKNOWN |
| 05_07 | 0.989259 | RISK/RISK | 16 | none | 1.000/1.000 | UNKNOWN |
| 08_10 | 30.002854 | PASS/PASS | 16 | none | 1.000/1.000 | UNKNOWN |

Band wavelengths are only reported where input metadata supplies them; physical units remain UNKNOWN. Cross-scene geometry and cross-band co-registration are not independently verified.


## Spatial/texture/edge selection audit

Fixed 32 x 32 native B9 windows (448 m at 14 m), deterministic spatial strata, B9 common-valid fraction >= 0.95, strong-edge fraction <= 0.10 after the fixed 2-pixel edge buffer. Thresholds were not relaxed to reach a quota.

Fixed windows and the edge filter can under-sample narrow or fragmented features, boundary-rich landforms, low-texture terrain, shadowed/low-contrast areas, and small targets. A low candidate count is reported as such, not repaired by weakening gates.

Cross-pair audit: `{'candidate_count': 34, 'duplicate_candidate_ids': [], 'overlaps': [], 'opposite_role_proximity': [], 'role_conflicts': [], 'status': 'PASS'}`

### Per-subregion window counts

Counts below distinguish all examined/rejected fixed windows from selected accepted candidates. Rejected candidates remain excluded; they are not labels.

- `r00_c00`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 60, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 4}
- `r00_c01`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 45, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 19}
- `r00_c02`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 41, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 23}
- `r00_c03`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 48, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 16}
- `r00_c04`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 55, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 8}
- `r00_c05`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 61, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 3}
- `r00_c06`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 64}
- `r00_c07`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 64}
- `r01_c00`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 48, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 16}
- `r01_c01`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 13, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 51}
- `r01_c02`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 11, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 53}
- `r01_c03`: rejected windows=62; quality-accepted windows=2; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 18, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 44}
- `r01_c04`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 22, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 42}
- `r01_c05`: rejected windows=62; quality-accepted windows=2; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 27, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 35}
- `r01_c06`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 33, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 31}
- `r01_c07`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 52, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 12}
- `r02_c00`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 47, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 17}
- `r02_c01`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 8, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 56}
- `r02_c02`: rejected windows=64; quality-accepted windows=0; exclusion details={'STRONG_EDGE_FRACTION_ABOVE_0.10': 64}
- `r02_c03`: rejected windows=60; quality-accepted windows=4; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 4, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 56}
- `r02_c04`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 12, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 51}
- `r02_c05`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 16, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 48}
- `r02_c06`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 16, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 47}
- `r02_c07`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 46, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 18}
- `r03_c00`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 49, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 15}
- `r03_c01`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 1, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 63}
- `r03_c02`: rejected windows=63; quality-accepted windows=1; exclusion details={'STRONG_EDGE_FRACTION_ABOVE_0.10': 63}
- `r03_c03`: rejected windows=64; quality-accepted windows=0; exclusion details={'STRONG_EDGE_FRACTION_ABOVE_0.10': 64}
- `r03_c04`: rejected windows=62; quality-accepted windows=2; exclusion details={'STRONG_EDGE_FRACTION_ABOVE_0.10': 62}
- `r03_c05`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 5, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 58}
- `r03_c06`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 14, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 50}
- `r03_c07`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 50, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 13}
- `r04_c00`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 50, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 14}
- `r04_c01`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 14, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 50}
- `r04_c02`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 6, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 57}
- `r04_c03`: rejected windows=64; quality-accepted windows=0; exclusion details={'STRONG_EDGE_FRACTION_ABOVE_0.10': 64}
- `r04_c04`: rejected windows=63; quality-accepted windows=1; exclusion details={'STRONG_EDGE_FRACTION_ABOVE_0.10': 63}
- `r04_c05`: rejected windows=63; quality-accepted windows=1; exclusion details={'STRONG_EDGE_FRACTION_ABOVE_0.10': 63}
- `r04_c06`: rejected windows=62; quality-accepted windows=2; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 3, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 59}
- `r04_c07`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 48, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 16}
- `r05_c00`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 43, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 21}
- `r05_c01`: rejected windows=62; quality-accepted windows=2; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 16, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 46}
- `r05_c02`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 16, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 47}
- `r05_c03`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 13, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 51}
- `r05_c04`: rejected windows=60; quality-accepted windows=4; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 4, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 56}
- `r05_c05`: rejected windows=64; quality-accepted windows=0; exclusion details={'STRONG_EDGE_FRACTION_ABOVE_0.10': 64}
- `r05_c06`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 9, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 55}
- `r05_c07`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 47, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 17}
- `r06_c00`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 50, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 14}
- `r06_c01`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 31, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 33}
- `r06_c02`: rejected windows=62; quality-accepted windows=2; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 21, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 41}
- `r06_c03`: rejected windows=62; quality-accepted windows=2; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 18, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 44}
- `r06_c04`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 16, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 47}
- `r06_c05`: rejected windows=63; quality-accepted windows=1; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 12, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 51}
- `r06_c06`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 16, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 48}
- `r06_c07`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 50, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 14}
- `r07_c00`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 64}
- `r07_c01`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 64}
- `r07_c02`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 64}
- `r07_c03`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 56, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 8}
- `r07_c04`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 47, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 17}
- `r07_c05`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 38, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 26}
- `r07_c06`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 46, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 18}
- `r07_c07`: rejected windows=64; quality-accepted windows=0; exclusion details={'B9_VALID_SUPPORT_BELOW_0.95': 62, 'STRONG_EDGE_FRACTION_ABOVE_0.10': 2}

Edge strata at the fixed 0.10 gate: `{'<=0.10': {'rejected': 191, 'accepted': 34}, '>0.10': {'rejected': 3871}}`

Texture strata use pooled candidate-window texture tertiles (cutpoints 0.000, 1387.139): `{'HIGH': {'rejected': 1345, 'accepted': 20}, 'LOW': {'rejected': 1695}, 'MEDIUM': {'rejected': 1022, 'accepted': 14}}`.

Quality support summary: selected candidate support min=1.000; rejected windows below 0.95=1854; rejected windows above 0.10 edge=3871.

### Role/suggestion caveats

Unsafe or crossing-grid geometries remain UNASSIGNED. Geometry edits invalidate support/role/leakage status and require a full four-pair re-audit.
No candidate is HIGH confidence. No REAL_CHANGE label was suggested because independent cross-scene geometry and cross-band co-registration are unverified. No grayscale brightness cue is interpreted as STABLE, CLOUD, or SHADOW.

## Input identity and protection

Protected small-file SHA256 records captured: 7.
Large imagery is not full-file hashed by this tool; its path/size/mtime and GeoTIFF header fingerprint are recorded in the run manifest. B9 may include a previously recorded Task15 hash; auxiliary-band hashes remain NOT_FROZEN unless present in the manifest.
