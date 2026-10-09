# GMQ Momentum Entry Timing / Chase Avoidance V4.2 — Diagnostic Only

## Purpose

V4.2 is a **read-only research-artifact audit**. It compares the population evidence in prior V2, V3 and V4.1 research artifacts. It does not optimize thresholds, re-run candidate sizing policies, modify production modules, or claim a strategy is profitable.

## Files

- `gmq_momentum_entry_timing_chase_avoidance_v4_2_diagnostic.py`
- `.github/workflows/gmq_momentum_entry_timing_chase_avoidance_v4_2_diagnostic.yml`
- `README_GMQ_MOMENTUM_ENTRY_TIMING_V4_2_DIAGNOSTIC.md`

## What it audits

- Prediction rows, composite identity uniqueness, duplicate identity rows and missing identity fields.
- Published raw-cycle and pre-start-filter counts when those counts exist in a version's `population_audit.csv`.
- Feature/prediction rows in the evaluation window.
- Chase, dangerous-chase and healthy-chase counts when the source artifact records them.
- Policy trigger counts separately from applied sizing event counts.
- Exact composite identity overlap between versions when both prediction files expose the same key columns.
- Available cost-stress summary fields and missing evidence.
- Whether dangerous/healthy class counts are below a diagnostic floor of 30 each. This is a warning gate, not proof of predictive ability.

## Important audit limitations

The uploaded V4.1 result ZIP contains aggregate `execution_audit.csv` sizing counts. It does **not** contain a record-level BUY order → fill → closed-cycle ledger. Therefore the diagnostic must not convert missing order/fill information into a zero unmatched-count. It records those counts as unavailable. A 100% match rate for applied sizing events is not the same as a 100% historical order/fill/cycle reconciliation.

If a prior version's artifacts do not exist or have expired, the workflow reports which version is missing. It does not fabricate V2 or V3 counts. If those artifacts originate from different commits, data snapshots or input settings, exact sample comparisons require that limitation to be reviewed.

## GitHub Actions: Android-friendly run

1. Add the Python file to the repository root.
2. Add the workflow at `.github/workflows/gmq_momentum_entry_timing_chase_avoidance_v4_2_diagnostic.yml`.
3. Do not replace `engine.py`, `portfolio.py`, `config.py`, `signals.py`, or any live state/orders/NAV file.
4. Open the repository's **Actions** tab, select **GMQ Momentum Entry Timing / Chase Avoidance V4.2 — Diagnostic Only**, and tap **Run workflow**.
5. Keep `fetch_latest_artifacts` enabled. The workflow reads the latest successful V2, V3 and V4.1 artifacts using the standard workflow token with `actions: read` permission. It does not commit or update files in the repository.
6. Download the workflow artifact named `gmq-momentum-entry-timing-chase-avoidance-v4-2-diagnostic-only`. Start with `..._report.md`, then review the population, stage, identity and policy-event CSVs.

If an older artifact is missing or expired, this workflow does not re-run V2/V3/V4.1. This is deliberate: it keeps the run short and diagnostic-only. In that case, provide the missing version result ZIP as a local input to the Python command or retain/re-upload that artifact through the existing research process.

## Local usage

```bash
python -m pip install -r gmq_momentum_entry_timing_requirements.txt
python -m py_compile gmq_momentum_entry_timing_chase_avoidance_v4_2_diagnostic.py
python gmq_momentum_entry_timing_chase_avoidance_v4_2_diagnostic.py --self-test

python gmq_momentum_entry_timing_chase_avoidance_v4_2_diagnostic.py \
  --v4-1-zip gmq-momentum-entry-timing-chase-avoidance-v4-1-results.zip \
  --out research/results_momentum_entry_timing_chase_avoidance_v4_2_diagnostic
```

To compare all versions from local archives, pass `--v2-zip`, `--v3-zip` and `--v4-1-zip`. Each argument can point to a result ZIP or an extracted directory.

## Output files

- `gmq_momentum_entry_timing_chase_avoidance_v4_2_report.md`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_population_summary.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_stage_ledger.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_identity_comparison.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_policy_events.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_artifact_sources.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_manifest.json`

## Acceptance standard

A report is not a promotion result. Missing version artifacts, missing record-level execution evidence, duplicate identities, missing key fields, or insufficient dangerous/healthy class counts remain explicit limitations. There is no automatic `PASS` for classification performance. A full historical run and an independent record-level execution reconciliation are still required before considering any production change.
