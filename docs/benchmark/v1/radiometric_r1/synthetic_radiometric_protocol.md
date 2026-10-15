# R1 controlled synthetic radiometric protocol

These are controlled numerical fixtures in arbitrary synthetic units, not reflectance simulations and not algorithm results. No normalizer is invoked. The generator writes only bounded NPZ arrays under `data/output/benchmark_v1_radiometric_r1/synthetic/`, refuses to overwrite an existing same-seed/same-size file, and caps each side at 512 pixels.

The frozen YAML is `synthetic_cases.yaml` (`protocol_version=R1-SYNTHETIC-1`, default seed 20261011, default 64×64). Source arrays are deterministic synthetic spatial patterns. The mask includes a small invalid corner. Repeated identical inputs, config, and seed produce identical arrays.

## Scenarios and truth

- **S1 global gain/offset:** `observed = clean_target × gain + offset`.
- **S2 spatial field:** coordinates `x,y` are linear ramps from −0.5 to +0.5. `gain(x,y)=gain_base+gx×x+gy×y`; `offset(x,y)=offset_base+ox×x+oy×y`; `observed=clean_target×gain(x,y)+offset(x,y)`.
- **S3 band-specific:** bands are bands-first. Each band has an independent frozen gain/offset and remains separately represented in truth. Raw cross-band errors are not combined.
- **S4 true change plus observation perturbation:** `changed_target=clean_target+change_amplitude×change_mask`; observation gain/offset are then applied. The clean target, binary change mask, amplitude layer, observation, and valid mask are stored separately.

Optional `noise_std` uses the frozen NumPy generator seed. Optional clipping stores `saturation_mask`, so saturated samples are not confused with recoverable values. Invalid/non-finite samples are excluded by `valid_mask` and represented as zero in the stored numerical arrays; zero is not a radiometric NoData claim.

## Run

```powershell
.\.venv-registration\Scripts\python.exe -m pytest tests\radiometric_r1\test_synthetic.py -q
.\.venv-registration\Scripts\python.exe scripts\benchmark\generate_radiometric_r1_synthetic.py
```

An explicit new run may use `--size 128 --seed 20261012`; it creates a different filename. Existing outputs are never overwritten.
