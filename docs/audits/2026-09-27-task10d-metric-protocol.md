# Task10D Metric Protocol

Task10D freezes an independent radiometric evaluation layer before the real
B9 rerun. It reports shared-valid overlap mean and standard-deviation
differences, exact one-dimensional Wasserstein-1 distances, fixed 256-pixel
local tiles, the Task9 seam transition zone, and circular Sobel gradient
orientation loss. All primary metrics are lower-is-better and are reported
without a composite winner score.

The protocol is stored as JSON in
`data/output/b9_five_scene_validation/task10d_metric_protocol.json` and its
SHA-256 is embedded in each Task10D scientific run. The loader rejects runtime
definition overrides, including changes to the 256-pixel local grid, the 4096
pixel support threshold, the SciPy Wasserstein implementation, seam threshold,
or circular angle formula.

These metrics are deliberately not paper Eq.(38) CD or Eq.(39) GL. The
available paper source does not expose enough information to verify those two
equations. Their status therefore remains `UNVERIFIED` in this protocol and
in every run summary; the Task10D metrics must not be presented as a paper
reproduction of CD or GL.
