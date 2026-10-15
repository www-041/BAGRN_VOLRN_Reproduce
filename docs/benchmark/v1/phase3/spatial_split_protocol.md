# Benchmark V1 four-pair spatial split protocol

Status: **FROZEN for four trial pairs only**. Full 15-pair Benchmark is **NOT FROZEN**.

- Native projected CRS: EPSG:32650; fixed origin: `(0, 0)` metres; cell size: 10,000 m.
- Index formula: `ix = floor((x - 0)/10000)`, `iy = floor((y - 0)/10000)`; global cell ID `g{ix}_{iy}`.
- Role formula: `DEVELOPMENT if (ix + 2*iy) % 3 == 0 else TEST`. No random seed is used.
- Each edge has a 500 m excluded margin. A point is safe only if `500 < x - ix*10000 < 9500` and `500 < y - iy*10000 < 9500` (strict inequalities).
- Both images' effective projected coordinates (human-adjusted if present) must occupy the same cell safe interior; otherwise UNASSIGNED. Geometry is EPSG:32650, not candidate-local 8x8 grid.
- Opposite-role safe interiors are separated by at least 1,000 m. All assigned left/right point coordinates are tested for this rule, including across pairs.
- This deterministic global rule applies unchanged to the remaining 11 pairs. Do not tune the 10 km cell, 500 m margin, origin or role formula against matcher accuracy. For future pairs, extend GeoJSON coverage with the **same** formula before use; the present polygon file covers the four-pair data envelope only.
- TEST assignment is a candidate role only. A distinct second reviewer must independently visually recheck before formal TEST scoring. No second reviewer or approval is fabricated here.
- The 160 source candidate CSV rows and all first-review history are preserved untouched. The derived assignment CSV retains every source column plus split diagnostics; its `split` column is the Phase3 proposed role, while `source_split` keeps the prior value.
- Concentration warning: fewer than 3 role-specific cells or one cell holding >50% of that role's points. Warnings are descriptive, not grounds to change the frozen rule.

| Pair | DEV | TEST candidates | UNASSIGNED | all assigned cells | DEV cells | TEST cells |
|---|---:|---:|---:|---:|---:|---:|
| 00_01 | 11 | 19 | 10 | 7 | 3 | 4 |
| 02_03 | 9 | 22 | 9 | 17 | 6 | 11 |
| 05_07 | 9 | 26 | 5 | 18 | 6 | 12 |
| 08_10 | 14 | 17 | 9 | 18 | 7 | 11 |

Formal TEST scoring eligibility remains zero until independent second review and separate truth export. No matcher or radiometric experiment was run.
