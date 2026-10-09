# GMQ Momentum Entry Timing / Chase Avoidance V4.2.1 — BASE Replay Diagnostics

## Why this follow-up exists

The V4.2 artifact audit completed successfully, but its manifest correctly records `historical_backtest_run: false`. It reads previous V2, V3 and V4.1 artifacts. It does not perform a new BASE replay.

This follow-up performs one new, unmodified BASE replay. It then builds the legacy V2/V3 feature dataset and the V4.1 execution-verified feature dataset from the **same BASE state, market panels, warm-up start and evaluation start**. That provides a direct, same-run comparison of the two sample definitions.

## Scope and safety

- Diagnostic-only. No candidate sizing policy is applied.
- No thresholds or hyperparameters are optimized.
- No production source file is replaced or edited.
- `engine.py`, `portfolio.py`, `config.py`, `data.py`, `signals.py`, and live state/order/NAV files are read by the current research code only.
- The script adds an order ID marker only to pending BUY dictionaries in the in-memory research state. It does not write that metadata to production files.
- The current engine BUY-fill event does not carry a persistent order ID. When multiple pending orders share market/ticker/tranche in the same step, the script reports the association as FIFO-inferred/ambiguous. It does not claim an exact order-ID join.
- `applied_sizing_events` is reported as **not run**, not as zero. This diagnostic does not test policy application.

## Files to add

1. Put `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_diagnostic.py` in the repository root.
2. Put `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay.yml` in `.github/workflows/`.
3. Keep the existing V4.2 artifact-audit files and all production files unchanged.

No existing file needs to be replaced.

## Run it from a phone

1. Open the repository on GitHub.
2. Add the Python file to the root using **Add file → Upload files**.
3. Open `.github/workflows/`, add the new workflow file, and commit the two new research files.
4. Open **Actions**.
5. Choose **GMQ Momentum Entry Timing / Chase Avoidance V4.2.1 — BASE Replay Diagnostics**.
6. Tap **Run workflow**. Keep the default `years=14` and `start=2018-01-01` for comparison with the prior study.
7. After the workflow completes, download the artifact `gmq-momentum-entry-timing-chase-avoidance-v4-2-replay-diagnostic`.
8. Read `..._replay_report.md` first. Then inspect the stage ledger, cycle-match ledger, order/fill ledger, population comparison and identity comparison.

The workflow first invokes the existing V4.2 script to download the latest successful V2/V3/V4.1 artifacts. It then runs the BASE replay. If a historical artifact is missing or expired, the report marks it unavailable instead of substituting zero counts.

## What this diagnostic measures

- Raw closed trade/cycle rows from the replay state.
- Continuation rows labelled `döngü yenilendi (devam)`.
- BUY order signals, BUY fill events, cancelled/no-fill orders, pending-at-end orders, orphan fills and same-key FIFO ambiguities.
- Raw closed cycles matched/unmatched to BUY fills by market/ticker/tranche/entry date.
- Legacy and V4.1 feature rows before and after invalid-row cleanup and the start-date filter.
- Duplicate/missing legacy four-field identities versus V4.1 five-field identities.
- Same-run identity overlap (with the explicit caveat that four-field overlap ignores tranche).
- Chase, dangerous and healthy counts, policy-trigger counts, and the minimum class-size warning.
- Historical artifact counts, common-date row counts and four-field identity overlap sensitivity.

## Validation boundary

The local packaging checks compile the script and run synthetic tests. They do **not** prove that the real historical replay will complete. The GitHub Actions run is the first test against the repository's live market-data code path. If replay or the cycle-count parity assertion fails, do not treat partial outputs as validated results and do not promote any policy.
