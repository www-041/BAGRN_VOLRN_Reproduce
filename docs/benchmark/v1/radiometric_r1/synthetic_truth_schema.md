# Synthetic truth schema

Each `S*_seed*_size*.npz` is a small synthetic-only record. It contains:

| Array | Meaning |
|---|---|
| `case_id`, `seed` | Scenario and deterministic seed |
| `clean_target` | Pre-perturbation synthetic source; not sensor truth |
| `observed` | Synthetic perturbed observation |
| `valid_mask` | Spatial or band-specific support used by the generator |
| `change_mask` | S4 real-change region; empty for S1–S3 |
| `change_amplitude` | Per-pixel/per-band injected change magnitude; zero outside S4 change |
| `saturation_mask` | Values clipped by configured lower/upper bounds |
| `truth_*` | Scenario-specific gain, offset, spatial field, per-band parameter, seed, and S4 amplitude values |
| `source_numeric_units` | Always `SYNTHETIC_ARBITRARY_UNITS` |

Arrays are generated from synthetic patterns only. No existing GeoTIFF, Task15/Task16 product, ROI label, or algorithm output is an input. S4 `clean_target` intentionally differs from `observed` only through injected real change plus the declared observation perturbation; the binary change and amplitude layers remain separate for future metric tests.
