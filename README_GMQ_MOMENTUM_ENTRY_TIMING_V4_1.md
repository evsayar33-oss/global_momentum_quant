# GMQ Momentum Entry Timing / Chase Avoidance V4.1

## Included
- `gmq_momentum_entry_timing_chase_avoidance_v4_1.py`
- `.github/workflows/gmq_momentum_entry_timing_chase_avoidance_v4_1.yml`

## Changes
1. Independent V4.1 research script; V4 is not overwritten.
2. Uses `gmq_early_adverse_predictor.build_trade_dataset()` for the verified closed-cycle dataset.
3. All policy/BASE simulations use the same warm-up start (`start_full`); metrics are calculated only from the requested evaluation start.
4. Cost stress re-runs BASE and candidate at every identical cost level and compares them over the same evaluation dates.
5. Adds a population audit CSV to make the execution-verified sample narrowing visible.
6. Requires at least 30 dangerous and 30 healthy chase observations before the classification gate can pass.
7. Keeps the execution mapping hard-fail check. No production files are edited.

## Install
1. Upload `gmq_momentum_entry_timing_chase_avoidance_v4_1.py` to the repository root.
2. Upload the YML to `.github/workflows/gmq_momentum_entry_timing_chase_avoidance_v4_1.yml`.
3. Commit both files and run **Actions → GMQ Momentum Entry Timing / Chase Avoidance V4.1 — Validation Fix**.

## Limits
Python syntax compilation was checked while packaging. The full historical workflow must still run to validate the repository-specific imports, actual data, execution mapping, and performance. This package does not claim V4.1 is profitable or production-ready.
