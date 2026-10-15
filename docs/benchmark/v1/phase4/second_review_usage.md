# Phase4 independent second review: PowerShell

Run these commands from the repository root with `.venv-registration` active. Every path in the tool defaults relative to the script's repository root, not the terminal's current directory.

Read-only current status:

```powershell
python scripts/export_benchmark_v1_test_truth.py --preflight
```

Open the first pair for a **different, actual second reviewer** (replace the example identity with that person's own name/ID; do not reuse `www` if `www` performed first review):

```powershell
python scripts/annotate_benchmark_v1.py --second-review --pair 00_01 --reviewer "SECOND_PERSON"
```

Other trial pairs use the same command with `--pair 02_03`, `05_07`, or `08_10`. The top list contains only the frozen TEST candidates for that pair. Select one; the frozen role and global cell are visible but cannot be changed. The two cyan crosses show its current locations. Inspect each image at original resolution (Zoom + shows finer detail); click either image to adjust its position and enter an adjustment reason. Enter your own localization-uncertainty estimate, indicate whether the two human localizations materially disagree, and explicitly confirm inspection of **both** images. Then select `Approve second review`, `Reject`, or `Doubtful`. An approval dialog asks you to confirm independent inspection. The action writes one append-only event; there is no bulk approval and no automatic approval from matching score, first-review HIGH, or ±0.5-pixel text.

After each save, check Pending, approved-valid, doubtful, rejected, spatially ineligible and dispute counts at the top. Closing and reopening replays the independent event log. Original Phase2 candidates, first-review files and Phase3 assignments are never rewritten by this mode. Events are stored at `docs/benchmark/v1/phase4/second_reviews/<pair_id>.jsonl`; do not manually edit, truncate or copy synthetic test events into this directory.

If a proposed adjustment ever crosses the original frozen safe-core edge, fails the same-cell or valid-mask check, or conflicts with DEVELOPMENT isolation, that candidate is permanently `NOT_ELIGIBLE_AFTER_COORDINATE_ADJUSTMENT` under this freeze. Moving it back does not restore formal TEST eligibility. A material coordinate disagreement is `DISPUTE_PENDING` until an independent human adjudication process outside this exporter resolves it. Do not alter original coordinates or regenerate Phase3 to bypass these outcomes.

Formal export is a separate later step. It requires a researcher-signed non-use JSON declaration described in [second_review_schema.md](second_review_schema.md); it must truthfully identify the research runs and attest that no TEST point was used for fitting, RANSAC, threshold selection, or tuning. Do not create that declaration before the relevant research process can substantiate it. When available, use a **new empty output directory**:

```powershell
python scripts/export_benchmark_v1_test_truth.py --event-dir docs/benchmark/v1/phase4/second_reviews --attestation docs/benchmark/v1/phase4/test_nonuse_attestation.json --output-dir docs/benchmark/v1/phase4/formal_test_export
```

For now the read-only preflight should report 84 pending and zero formal truth. Software checks file identities, event structure and declared reviewer names; it cannot independently authenticate the physical person or prove historical non-use.
