# Task15/Task16 formula compatibility audit (read-only)

Audit performed by source inspection only. No Task15/Task16 functions were invoked or changed.

| Metric | Existing helper / observed formula | Support and aggregation | R1 treatment |
|---|---|---|---|
| MAMD | `src/multiscene_sift/radiometric_metrics.py::compute_pair_mamd`: absolute difference of means | Caller mask intersected with finite values; Task10D pair rows later weighted by shared-valid pixel count | Same pair formula, but R1 records independent ROI labels and uses equal ROI/pair macro summaries, not pixel-weighted inter-pair mean |
| MSDD | `compute_pair_msdd`: absolute difference of population standard deviations (`np.std`, default `ddof=0`) | Same shared-valid support and pixel-weighted pair aggregate | Same per-pair formula; R1 macro aggregation differs and is explicit |
| RDD | `compute_pair_rdd`: exact `scipy.stats.wasserstein_distance` over 1D valid sample arrays | Same shared-valid support; pair aggregate weighted by valid pixel counts; no histogram approximation | Same exact distance per ROI; R1 reports equal ROI/pair macro results |
| Local MAMD/RDD | `compute_local_pair_metrics` and Task16 streaming helper | 256×256 canonical-origin tiles; only tiles with ≥4096 shared-valid samples; R1 does not silently inherit this Task10D tile support | New fixed human ROI support; no Task10D tile summaries are reused as ROI truth |
| Seam MAE/RMSE/RDD | `compute_seam_zone_metrics`; MAE/RMSE and exact Wasserstein on seam mask | Task9-generated method seam zone and weighted aggregation | Not used as the sole R1 evaluation support; a future comparison must report both fixed geographical ROI and method seam zones |
| CGL / gradient metrics | `compute_cgl` plus `structural_metrics.py` | Sobel, fully valid 3×3 stencil, nonzero-gradient threshold; current structure module also has gradient magnitude NCC | R1 declares a separate gated structure implementation; implementation/support differences must remain named |
| Paper CD/GL | `task10d_metric_protocol.json` records `cd_status=UNVERIFIED`, `gl_status=UNVERIFIED` | Definitions are not established as paper-faithful in current protocol | R1 does not relabel another histogram/gradient diagnostic as CD/GL |

`_shared_values` in `radiometric_metrics.py` intersects supplied masks with `isfinite` for both arrays. Task15/Task16 call the same pair helpers through `radiometric_runner.py` / `task16_volrn_comparison.py`; Task16’s streaming path reads pair/tile windows and retains the same 256-pixel, 4096-sample local gate. The existing helper's aggregate is `weighted_mean` by valid pixel count plus edge median, worst pair, and per-pair rows. These details matter: same formula name does not mean same support or same aggregate as R1.

R1 tests establish formula outputs on synthetic numerical fixtures. No comparison was made to Task15/Task16 outputs and no historical result file was edited. Source reading was limited to helper modules/configuration and does not claim validation against the cited paper's original implementation.
