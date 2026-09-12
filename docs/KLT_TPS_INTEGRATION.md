# KLT-TPS Registration Backend

Source algorithm: senior-provided `register_images.py`  
SHA256: `4ff4ff68f52bb58764755c3efd14fcd15fadf6a05f636d784c562855cc98fbac`

Preserved algorithm semantics:
- local normalization
- GFTT corners
- forward/backward Pyramid-LK
- FB threshold filtering
- thin-plate-spline RBFInterpolator
- coarse field + cubic upsampling
- Jacobian folding + max-shift safety
- output-to-source flow
- one cubic resample from ORIGINAL moving DN

Project adaptations:
- Rasterio/in-memory arrays replace GDAL standalone I/O
- geographic overlap context replaces same-shape/same-GeoTransform `check_grid()`
- accepted correspondences are mapped back to moving native pixel coordinates
- one estimated B12 flow is applied to all loaded bands

Phase-I limitation: `klt_tps` supports exactly N=2.

Manual N=2 validation (run by the user after this branch is pushed):

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_registration_pair.py `
  --config configs/dz01_klt_tps_n2_b12.yaml `
  --scene-i 0 --scene-j 1 --validation-band B14 `
  --output-dir outputs/klt_tps_n2_b12
```

This branch does not run real DZ01V data and does not implement N=4/N=6,
multi-scene TPS composition, parameter search, or BAGRN/VOLRN changes.

## TPS-FOLD-D1 diagnostic

TPS-FOLD-D1 is a diagnostic-only path used when the dense TPS geometry gate
rejects a field. It preserves pre-gate controls, displacement statistics,
Jacobian statistics, fold support, and optional visual evidence for manual
inspection.

The diagnostic does not modify TPS fitting, does not relax the Jacobian safety
gate, and never allows an unsafe flow to warp the original imagery. Its purpose
is to distinguish evidence associated with unsupported extrapolation,
inconsistent control displacement, or coordinate/displacement anomalies. It
does not assert which explanation is responsible before a real N=2 run is
manually inspected.

## TPS-SUPPORT-C1 diagnostic

TPS-SUPPORT-C1 is a diagnostic-only, N=2 comparison on the same fixed
HOLDOUT. The three stages are defined exactly as follows:

- Stage 0 = raw TPS geometry-only; an unsafe raw field is never warped.
- Stage 1 = KLT median translation-only.
- Stage 2 = `g + w(Fraw-g)`, using a fixed 64 px inside-only taper.
- fixed HOLDOUT = `configs/dz01_model_c1_holdout_manifest.json`
- registration = B12
- validation = B14
- production backend unchanged

The raw TPS field is fitted once. Stage 1 and Stage 2 are both warped from
the ORIGINAL moving array, and both use the same seven B14 validation windows.
The supported field must pass the existing Jacobian and max-shift safety gate
before Stage 2 is warped. This diagnostic does not search taper widths or
select a HOLDOUT from final validation results.

User-only real command, to be run after reviewing the pushed branch:

```powershell
.\.venv\Scripts\python.exe scripts/diagnose_registration_pair.py `
  --config configs\dz01_klt_tps_n2_b12.yaml `
  --scene-i 0 `
  --scene-j 1 `
  --validation-band B14 `
  --holdout-manifest configs\dz01_model_c1_holdout_manifest.json `
  --tps-support-causal-test `
  --output-dir outputs\klt_tps_n2_b12_tps_support_c1_v1
```

Codex must not execute this command. It is the user's manual real-data
checkpoint, after all synthetic, unit, and full regression tests pass.

Interpret the post-run result using only these three branches:

1. supported safe + better than translation -> design production adoption next
2. supported safe + equal/worse -> local TPS residual has limited/negative value; investigate before adoption
3. supported unsafe -> return to geometry/root-cause; no taper search and no safety relaxation

Even a successful C1 result authorizes only a subsequent production-adoption
design; it does not silently change the production backend here.
