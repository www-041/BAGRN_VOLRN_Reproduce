# R1 temporal stratification report

This report joins existing frozen manifests. It does not recompute registration, radiometric values, or image overlap.

- Scene combinations: 78 (13 choose 2).
- B9 effective-overlap pairs: 48.
- No-effective-overlap pairs: 30 (`NO_VALID_OVERLAP`, not algorithm FAIL).
- Same-date T0: 7; cross-date T1 (<7d): 13; T2 (7–<30d): 6; T3 (>=30d): 22.
- Cross-date effective overlap pairs: 41.
- Source SHA256: pair candidates `0299c25615f78aa943edcd40e15287be18a887b917f12f6f75d3a2d82a160249`, temporal inventory `139c26be49e65571da91251dad574172369467e41e68177539674efecf4c76a2`, scene inventory `599e5793f4189ae24b7ca28858b5d2f3c4d7f496ad9d11d2f52dfc7ae0b50ea4`.
- Stable-landcover and real-change truth remain UNKNOWN/UNVERIFIED; valid support is not stable-ground truth.

## Pair counts by time stratum

| Stratum | Count | Meaning |
|---|---:|---|
| T0 | 7 | Same date |
| T1 | 13 | Different date, <7 days |
| T2 | 6 | 7 to <30 days |
| T3 | 22 | >=30 days |
| NO_VALID_OVERLAP | 30 | No shared valid B9 pixels |
