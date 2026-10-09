# Validation report — GMQ V4.2.1 BASE Replay Diagnostics

## Source artifact verification

- Previous V4.2 artifact run: https://github.com/evsayar33-oss/global_momentum_quant/actions/runs/37974041171
- Artifact: `gmq-momentum-entry-timing-chase-avoidance-v4-2-diagnostic-only`
- Reported SHA-256: `b899dc8e512686b1ccfc04ad283211b8065656b28fb6810231f297afae21cd70`
- The downloaded artifact ZIP was hashed locally and the result matched the digest exactly.
- That prior run completed successfully but performed an artifact audit only. It did not run the new BASE replay (`historical_backtest_run: false`).

## Local checks run for this follow-up package

- `python -m py_compile gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_diagnostic.py` — PASS.
- `python gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_diagnostic.py --self-test` — PASS.
- Synthetic instrumentation test — PASS. It verifies order-to-fill handling, no-fill disappearance, no orphan fills in the fixture, and status transitions.
- Identity test — PASS. It confirms a legacy four-field key can collapse two different tranche rows while a five-field key distinguishes them.
- Cycle reconciliation test — PASS. A continuation-cycle row without a corresponding BUY fill remains unmatched.
- Low-sample classification gate — PASS. Class counts 13/3 remain `INCONCLUSIVE_LOW_N`.
- YAML parse — PASS.
- Historical artifact summary helper — PASS. It reads the available previous artifacts and reproduces the common-window counts: V2 8,825; V3 8,825; V4.1 3,430. Four-key identity overlap with V4.1 is six and is labelled sensitivity-only.
- Final package ZIP integrity (`unzip -t`) — PASS after the final rebuild. The final ZIP hash is provided alongside the download link, rather than embedded inside the ZIP itself.

## Not yet verified

- A fresh BASE replay against the current repository market-data path has not yet been run by the new V4.2.1 workflow.
- Real-data order/fill/cycle parity and the V2/V3-vs-V4.1 same-run sample bridge remain unverified until that workflow runs.
- No profitability, predictive validity, or production-promotion claim is made.

## Production safety

The package contains new research files only. It does not replace production modules. Candidate sizing policies are not applied, thresholds are not optimized, and applied sizing counts are intentionally marked not run.
