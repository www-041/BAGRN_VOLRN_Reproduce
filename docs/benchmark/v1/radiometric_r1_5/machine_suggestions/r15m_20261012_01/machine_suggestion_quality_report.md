# R1.5 machine suggestions

Machine run: `r15m_20261012_01`
Candidate run: `r15_20261012_04`

All entries are machine-only suggestions (`SUGGESTED`), not human review or ground truth. Because pixel units are UNKNOWN, there is no suitable DEM/cloud-quality mask, geometry is not independently verified, and no independent change/land-cover reference is available, raw multi-band and texture evidence is not converted into semantic classes. In particular, no reflectance indices are computed and `CONFIRMED_CHANGE` is never emitted.

## 00_01

Candidate count: 2

| Field | Machine label distribution |
|---|---|
| `suggested_landcover_type` | UNKNOWN: 2; confidence LOW: 2 |
| `suggested_terrain_type` | UNKNOWN: 2; confidence MEDIUM: 2 |
| `suggested_temporal_status` | UNKNOWN: 2; confidence MEDIUM: 2 |
| `suggested_interference_type` | UNKNOWN: 2; confidence MEDIUM: 2 |

## 02_03

Candidate count: 6

| Field | Machine label distribution |
|---|---|
| `suggested_landcover_type` | UNKNOWN: 6; confidence LOW: 6 |
| `suggested_terrain_type` | UNKNOWN: 6; confidence MEDIUM: 6 |
| `suggested_temporal_status` | UNKNOWN: 6; confidence MEDIUM: 6 |
| `suggested_interference_type` | UNKNOWN: 6; confidence MEDIUM: 6 |

## 05_07

Candidate count: 15

| Field | Machine label distribution |
|---|---|
| `suggested_landcover_type` | UNKNOWN: 15; confidence LOW: 15 |
| `suggested_terrain_type` | UNKNOWN: 15; confidence MEDIUM: 15 |
| `suggested_temporal_status` | UNKNOWN: 15; confidence MEDIUM: 15 |
| `suggested_interference_type` | UNKNOWN: 15; confidence MEDIUM: 15 |

## 08_10

Candidate count: 11

| Field | Machine label distribution |
|---|---|
| `suggested_landcover_type` | UNKNOWN: 11; confidence LOW: 11 |
| `suggested_terrain_type` | UNKNOWN: 11; confidence MEDIUM: 11 |
| `suggested_temporal_status` | UNKNOWN: 11; confidence MEDIUM: 11 |
| `suggested_interference_type` | UNKNOWN: 11; confidence MEDIUM: 11 |

## Human agreement

No human labels are generated here. Agreement remains `NOT_EVALUATED` until eligible human review records exist. The provided comparison interface is descriptive agreement only, not classifier accuracy.

## Bias and confusion checks

The generator does not use inter-date brightness similarity to assign `NO_CHANGE`; raw-DN median differences are retained as evidence only. No land-cover/shadow/change confusion rate can be estimated because no semantic labels were emitted or independently reviewed. All four fields are UNKNOWN, so this run demonstrates insufficient current semantic evidence rather than validated classification.

## Sampling / limitations

Review representatives across all four pairs and across DEVELOPMENT/TEST/UNASSIGNED before relying on any suggestion. A deterministic pair/role-balanced sample of 12 candidates is listed in `human_review_sample.csv`. Because these recommendations are conservatively UNKNOWN, inspect whether the measured evidence could support any semantic label; do not treat a low uncertainty or candidate score as class confidence. Fixed candidate windows and strong-edge filters can under-sample narrow boundaries, low-texture areas, shadowed terrain, and homogeneous regions.
