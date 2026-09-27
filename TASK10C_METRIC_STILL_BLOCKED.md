# TASK10C_METRIC_STILL_BLOCKED

Date: 2026-09-27

## Hard-stop reason

Task11.5A located and inspected the public 2024 paper source, the cited
Xia et al. metric source, and the exact public author repository commit required
by the plan. The sources confirm the high-level Eq. (38)/(39) structure, but
they do not uniquely determine all implementation-relevant metric semantics.

The unresolved Eq. (38) terms are the histogram channel/range/normalization,
the numeric histogram bin count, and the exact `DeltaH` bin-to-bin distance.
The unresolved Eq. (39) term is the angular wrap rule: the cited C++ code uses
raw absolute subtraction of signed `atan2l` angles, while the paper text does
not state whether a shortest-arc wrap should be applied.

The plan forbids guessing these formulas. Therefore no implementation changes,
real-data replay, VOLRN science gate, or final six-run experiment was performed.

## Evidence artifacts

- [`docs/audits/2026-09-27-paper-metric-source-lock.md`](docs/audits/2026-09-27-paper-metric-source-lock.md)
- [`docs/audits/2026-09-27-paper-metric-source-lock.json`](docs/audits/2026-09-27-paper-metric-source-lock.json)
- Frozen reference repository commit:
  `MenghanXia/ColorConsistency@05d03d87ea67dd92de7ccea09dd450579f6d991e`

## State preserved

- Branch: `exp/2026-09-24-b9-five-scene-validation`
- HEAD at audit start: `3b380a1` (`docs: record Task10C formula audit hard stop`)
- `push_performed=false`
- No merge or push was performed.
- Existing Task0--11 implementation and prior test evidence were not changed.
