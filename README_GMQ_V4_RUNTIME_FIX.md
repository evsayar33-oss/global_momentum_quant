# GMQ V4 Runtime Fix

This package replaces only the V4 GitHub Actions workflow. It fixes the specific error where V4 imports `build_trade_dataset()` from the wrong helper module. The workflow applies a guarded one-line patch to the checked-out research script during the runner job, verifies the function exists in `gmq_early_adverse_predictor.py`, compiles the script, runs synthetic smoke, then runs the research.

Install:
1. Extract the ZIP.
2. Replace `.github/workflows/gmq_momentum_entry_timing_chase_avoidance_v4.yml` in the repository with the included file.
3. Commit and run the workflow from Actions.

Production files are not changed. This fixes the reported import error but cannot guarantee that no other runtime/data errors exist.
