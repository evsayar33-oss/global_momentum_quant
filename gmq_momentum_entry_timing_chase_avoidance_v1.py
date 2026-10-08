#!/usr/bin/env python3
"""
Global Momentum Quant — Momentum Entry Timing / Chase Avoidance V1

RESEARCH ONLY.

Purpose
-------
Test whether a small entry-time intervention improves the existing BASE
momentum strategy by avoiding entries that arrive unusually extended or gap
far above the prior signal close.

Research basis
--------------
The prior BASE failure analysis found:
- late_extension: 1,470 trades; 11.38% of all trades; 14.56% of total loss
  contribution.
- gap_risk: 537 trades; 4.16% of all trades; 3.86% of total loss contribution.
- momentum_extension_z > 2.5: 293 trades; mean return +0.48%.
- gap_atr > 2.0: 160 trades; mean return +0.63%.

Those fixed thresholds are diagnostic anchors, not OOS-optimized thresholds.

Chronology / point-in-time rule
-------------------------------
- `momentum_extension_z` is known at signal close.
- `gap_atr` is known at the next entry open.
- In this research wrapper, the realized next-open condition is applied to the
  pending BUY amount for the simulated entry. This is equivalent to deciding at
  entry open, not at signal close.
- No future exit/path variable is used to decide the entry.

Production protection
---------------------
Production `engine.py`, `portfolio.py`, `config.py`, `signals.py`, state,
orders, NAV and trade files are never modified.
The research uses the repository's real `engine.MarketData` and
`portfolio.process_day` path and only changes BUY amount inside an in-memory
research wrapper.

Policies (pre-declared; no optimizer)
-------------------------------------
BASE              : no intervention
EXT_SOFT          : extension_z > 2.5 -> 0.75x
GAP_SOFT          : gap_atr > 2.0 -> 0.75x
CHASE_AND_SOFT    : extension_z > 2.5 AND gap_atr > 2.0 -> 0.75x
CHASE_OR_SOFT     : extension_z > 2.5 OR gap_atr > 2.0 -> 0.75x
CHASE_AND_STRONG  : extension_z > 2.5 AND gap_atr > 2.0 -> 0.50x
GAP_SKIP          : gap_atr > 2.0 -> 0.00x
EXT_SKIP          : extension_z > 2.5 -> 0.00x

Primary policy is CHASE_AND_SOFT. The strong/skip variants are fixed
sensitivity tests, not optimization.

Outputs
-------
- gmq_momentum_entry_timing_chase_avoidance_v1_raporu.md
- gmq_momentum_entry_timing_chase_avoidance_v1_metrics.csv
- gmq_momentum_entry_timing_chase_avoidance_v1_folds.csv
- gmq_momentum_entry_timing_chase_avoidance_v1_monthly.csv
- gmq_momentum_entry_timing_chase_avoidance_v1_stress.csv
- gmq_momentum_entry_timing_chase_avoidance_v1_diagnostics.csv
- gmq_momentum_entry_timing_chase_avoidance_v1_policy_usage.csv
- gmq_momentum_entry_timing_chase_avoidance_v1_predictions.csv
- gmq_momentum_entry_timing_chase_avoidance_v1_trades.csv

Run from repository root.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PRIMARY_COST_RT = 0.35
STRESS_COSTS = (0.35, 0.50, 0.75, 1.00, 1.25)

def load_research_helper():
    import gmq_tail_risk_alpha_conditional_sizing as mod
    return mod

def normalize_date_series(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, errors="coerce")
    try:
        x = x.dt.tz_localize(None)
    except TypeError:
        pass
    return x.dt.normalize()

DEFAULT_YEARS = 14
DEFAULT_START = "2018-01-01"
PRIMARY_POLICY = "CHASE_AND_SOFT"
EXT_THRESHOLD = 2.5
GAP_ATR_THRESHOLD = 2.0
FOLD_MONTHS = 6

POLICIES = {
    "BASE": {"mode": "none", "multiplier": 1.00},
    "EXT_SOFT": {"mode": "extension", "multiplier": 0.75},
    "GAP_SOFT": {"mode": "gap", "multiplier": 0.75},
    "CHASE_AND_SOFT": {"mode": "and", "multiplier": 0.75},
    "CHASE_OR_SOFT": {"mode": "or", "multiplier": 0.75},
    "CHASE_AND_STRONG": {"mode": "and", "multiplier": 0.50},
    "GAP_SKIP": {"mode": "gap", "multiplier": 0.00},
    "EXT_SKIP": {"mode": "extension", "multiplier": 0.00},
}


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_timing_flags(ds: pd.DataFrame) -> pd.DataFrame:
    required = ["gap_atr", "momentum_extension_z", "signal_date", "entry_date", "ret_pct", "market", "ticker"]
    missing = [c for c in required if c not in ds.columns]
    if missing:
        raise ValueError(f"Eksik timing feature: {missing}")
    x = ds.copy()
    x["signal_date"] = normalize_date_series(x["signal_date"])
    x["entry_date"] = normalize_date_series(x["entry_date"])
    x["gap_atr"] = safe_num(x["gap_atr"])
    x["momentum_extension_z"] = safe_num(x["momentum_extension_z"])
    x["ret_pct"] = safe_num(x["ret_pct"])

    x["flag_extension"] = x["momentum_extension_z"] > EXT_THRESHOLD
    x["flag_gap"] = x["gap_atr"] > GAP_ATR_THRESHOLD
    x["flag_chase_and"] = x["flag_extension"] & x["flag_gap"]
    x["flag_chase_or"] = x["flag_extension"] | x["flag_gap"]
    x["entry_gap_known_at_entry"] = x["gap_atr"].notna()
    x["extension_known_at_signal"] = x["momentum_extension_z"].notna()
    return x


def policy_condition(x: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(False, index=x.index)
    if mode == "extension":
        return x["flag_extension"].fillna(False)
    if mode == "gap":
        return x["flag_gap"].fillna(False)
    if mode == "and":
        return x["flag_chase_and"].fillna(False)
    if mode == "or":
        return x["flag_chase_or"].fillna(False)
    raise KeyError(mode)


def apply_policy(x: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in POLICIES:
        raise KeyError(name)
    spec = POLICIES[name]
    z = x.copy()
    cond = policy_condition(z, spec["mode"])
    # Missing timing data never causes a penalty: research must fail closed to
    # BASE, not invent an adverse condition.
    z["timing_condition"] = cond.astype(bool)
    z["sizing_multiplier"] = np.where(cond, float(spec["multiplier"]), 1.0)
    z["policy"] = name
    return z


def make_lookup(pred: pd.DataFrame) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for r in pred.to_dict("records"):
        d = pd.Timestamp(r["signal_date"]).strftime("%Y-%m-%d")
        out[(str(r["market"]), str(r["ticker"]), d)] = float(r["sizing_multiplier"])
    return out


def policy_usage(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in POLICIES:
        z = apply_policy(pred, name)
        cond = z["timing_condition"]
        rows.append({
            "policy": name,
            "mode": POLICIES[name]["mode"],
            "multiplier_when_triggered": POLICIES[name]["multiplier"],
            "n": len(z),
            "condition_n": int(cond.sum()),
            "condition_pct": float(cond.mean() * 100.0),
            "mean_multiplier": float(z["sizing_multiplier"].mean()),
            "median_multiplier": float(z["sizing_multiplier"].median()),
            "mean_trigger_ret_pct": float(z.loc[cond, "ret_pct"].mean()) if cond.any() else np.nan,
            "win_trigger_pct": float((z.loc[cond, "ret_pct"] > 0).mean() * 100.0) if cond.any() else np.nan,
            "tail10_trigger_pct": float((z.loc[cond, "ret_pct"] <= -10).mean() * 100.0) if cond.any() else np.nan,
            "tail15_trigger_pct": float((z.loc[cond, "ret_pct"] <= -15).mean() * 100.0) if cond.any() else np.nan,
        })
    return pd.DataFrame(rows)


def diagnostics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, mode in [
        ("extension_only", "extension"),
        ("gap_only", "gap"),
        ("chase_and", "and"),
        ("chase_or", "or"),
    ]:
        cond = policy_condition(pred, mode)
        g = pred.loc[cond].copy()
        rows.append({
            "scenario": name,
            "n": len(g),
            "share_pct": float(len(g) / max(len(pred), 1) * 100.0),
            "win_rate_pct": float((g["ret_pct"] > 0).mean() * 100.0) if len(g) else np.nan,
            "mean_ret_pct": float(g["ret_pct"].mean()) if len(g) else np.nan,
            "median_ret_pct": float(g["ret_pct"].median()) if len(g) else np.nan,
            "mean_loss_pct": float(g.loc[g["ret_pct"] < 0, "ret_pct"].mean()) if (g["ret_pct"] < 0).any() else np.nan,
            "tail10_rate_pct": float((g["ret_pct"] <= -10).mean() * 100.0) if len(g) else np.nan,
            "tail15_rate_pct": float((g["ret_pct"] <= -15).mean() * 100.0) if len(g) else np.nan,
            "bist_mean_ret_pct": float(g.loc[g.market == "bist", "ret_pct"].mean()) if (g.market == "bist").any() else np.nan,
            "us_mean_ret_pct": float(g.loc[g.market == "us", "ret_pct"].mean()) if (g.market == "us").any() else np.nan,
        })
    return pd.DataFrame(rows)


def six_month_calendar(last_date: pd.Timestamp, start: pd.Timestamp) -> pd.DataFrame:
    rows = []
    s = start.normalize()
    fold = 0
    end_limit = last_date.normalize() + pd.Timedelta(days=1)
    while s < end_limit:
        e = min(s + pd.DateOffset(months=FOLD_MONTHS), end_limit)
        rows.append({"fold": fold, "oos_start": s, "oos_end": e})
        s = e
        fold += 1
    return pd.DataFrame(rows)


def fold_metrics(runs: Dict[str, Tuple[dict, pd.Series]], calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_daily = runs["BASE"][1]
    for row in calendar.to_dict("records"):
        t0 = pd.Timestamp(row["oos_start"])
        t1 = pd.Timestamp(row["oos_end"])
        b = base_daily[(base_daily.index >= t0) & (base_daily.index < t1)]
        if len(b) < 2:
            continue
        bm = metrics_from_daily(b)
        b_ret = float(b.iloc[-1] / b.iloc[0] - 1)
        for name, (_st, daily) in runs.items():
            s = daily[(daily.index >= t0) & (daily.index < t1)]
            if len(s) < 2:
                continue
            m = metrics_from_daily(s)
            period_ret = float(s.iloc[-1] / s.iloc[0] - 1)
            rows.append({
                "fold": int(row["fold"]),
                "oos_start": t0,
                "oos_end": t1,
                "policy": name,
                "period_return_pct": period_ret * 100.0,
                "base_period_return_pct": b_ret * 100.0,
                "period_return_delta_pp": (period_ret - b_ret) * 100.0,
                "maxdd_pct": m.get("maxdd_pct", np.nan),
                "base_maxdd_pct": bm.get("maxdd_pct", np.nan),
                "pf": m.get("profit_factor", np.nan),
            })
    return pd.DataFrame(rows)


def matched_monthly(base: pd.Series, candidate: pd.Series) -> pd.DataFrame:
    b = base.resample("ME").last().pct_change()
    c = candidate.resample("ME").last().pct_change()
    out = pd.concat([b.rename("base_ret"), c.rename("candidate_ret")], axis=1).dropna()
    out["delta_pp"] = (out["candidate_ret"] - out["base_ret"]) * 100.0
    return out


def bootstrap(values: pd.Series, n_boot: int = 10000, seed: int = 20261008) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(x) < 10:
        return {"mean_pp": np.nan, "ci5_pp": np.nan, "ci95_pp": np.nan, "p_le0": np.nan, "positive_pct": np.nan, "n_months": len(x)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return {
        "mean_pp": float(x.mean()),
        "ci5_pp": float(np.quantile(means, 0.05)),
        "ci95_pp": float(np.quantile(means, 0.95)),
        "p_le0": float((means <= 0).mean()),
        "positive_pct": float((x > 0).mean() * 100.0),
        "n_months": len(x),
    }


def write_report(out_dir: Path, dataset: pd.DataFrame, pred: pd.DataFrame, metrics: pd.DataFrame, folds: pd.DataFrame, monthly: pd.DataFrame, stress: pd.DataFrame, diagnostics_df: pd.DataFrame, usage: pd.DataFrame, boot: dict) -> None:
    primary = metrics.loc[metrics.policy == PRIMARY_POLICY].iloc[0]
    p_folds = folds.loc[folds.policy == PRIMARY_POLICY]
    positive_folds = float((p_folds["period_return_delta_pp"] > 0).mean() * 100.0) if len(p_folds) else np.nan
    stress_p = stress.loc[stress.policy == PRIMARY_POLICY]
    stress_positive = int((stress_p["cagr_delta_vs_base_pp"] > 0).sum()) if len(stress_p) else 0
    passed = bool(
        primary["cagr_delta_vs_base_pp"] >= 0
        and np.isfinite(boot["ci5_pp"]) and boot["ci5_pp"] > 0
        and np.isfinite(positive_folds) and positive_folds >= 60
        and stress_positive >= 4
    )
    verdict = "PROMISING — NEEDS FINAL HOLDOUT / PARITY" if passed else "REJECT FOR PROMOTION"
    lines = [
        "# Global Momentum Quant — Momentum Entry Timing / Chase Avoidance V1",
        "",
        "**RESEARCH ONLY — PRODUCTION DOSYALARINA DOKUNULMADI**",
        "",
        "## 1. Hipotez",
        "BASE doğru momentum adaylarını seçiyor olabilir; ancak bazı girişler sinyal kapanışından sonraki açılışta fiyatın aşırı genişlemesi nedeniyle chase/late-entry karakterine dönüşebilir. Bu test mevcut seçimleri değiştirmeden yalnızca BUY miktarını entry-time chase koşulunda azaltır.",
        "",
        "## 2. Sabit diagnostik eşikler",
        f"- `momentum_extension_z > {EXT_THRESHOLD:.1f}` → extension flag",
        f"- `gap_atr > {GAP_ATR_THRESHOLD:.1f}` → entry-gap flag",
        "- Threshold'lar önceki BASE failure analysis diagnostik bucket'larından alınmıştır; bu test içinde yeniden optimize edilmemiştir.",
        "",
        "## 3. Policy family",
        usage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 4. Ex-ante chase diagnostics",
        diagnostics_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. OOS portfolio metrics",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. Six-month robustness blocks",
        folds.to_markdown(index=False, floatfmt=".4f") if not folds.empty else "—",
        "",
        "## 7. Matched-month bootstrap — primary",
        f"Months: **{boot['n_months']}**",
        f"Mean monthly delta: **{boot['mean_pp']:.4f} pp**",
        f"Bootstrap CI 5–95: **[{boot['ci5_pp']:.4f}, {boot['ci95_pp']:.4f}] pp**",
        f"P(mean delta <= 0): **{boot['p_le0']:.4f}**",
        f"Positive months: **{boot['positive_pct']:.2f}%**",
        "",
        "## 8. Cost stress",
        stress.to_markdown(index=False, floatfmt=".4f"),
        f"Positive stress scenarios for primary: **{stress_positive}/{len(stress_p)}**",
        "",
        "## 9. Decision gate",
        "Primary gate: non-negative CAGR delta + bootstrap CI5 > 0 + positive six-month folds >= 60% + >=4/5 cost-stress scenarios positive.",
        f"Positive six-month fold ratio for primary: **{positive_folds:.2f}%**",
        "",
        f"# FINAL VERDICT\n**{verdict}**",
        "",
        "No production file, live state, order ledger or NAV file is modified by this research.",
    ]
    (out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_raporu.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def synthetic_smoke(out_dir: Path) -> None:
    rng = np.random.default_rng(42)
    n = 12000
    dates = pd.bdate_range("2016-01-04", periods=600)
    signal_dates = np.repeat(dates, n // len(dates))
    signal_dates = signal_dates[:n]
    gap = rng.exponential(0.7, n)
    ext = rng.normal(0.5, 1.2, n)
    ret = rng.normal(3.0, 7.0, n) - 1.5 * (gap > 2.0) - 1.2 * (ext > 2.5) - 1.8 * ((gap > 2.0) & (ext > 2.5))
    ds = pd.DataFrame({
        "market": np.where(np.arange(n) % 2 == 0, "bist", "us"),
        "ticker": [f"T{i % 100:03d}" for i in range(n)],
        "signal_date": signal_dates,
        "entry_date": signal_dates + pd.Timedelta(days=1),
        "exit_date": signal_dates + pd.Timedelta(days=21),
        "ret_pct": ret,
        "gap_atr": gap,
        "momentum_extension_z": ext,
    })
    p = build_timing_flags(ds)
    u = policy_usage(p)
    d = diagnostics(p)
    for name in POLICIES:
        z = apply_policy(p, name)
        assert len(z) == len(p)
        assert ((z["sizing_multiplier"] >= 0) & (z["sizing_multiplier"] <= 1)).all()
    p.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_predictions.csv", index=False)
    u.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_policy_usage.csv", index=False)
    d.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_diagnostics.csv", index=False)
    (out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_metrics.csv").write_text("synthetic_smoke,ok\n", encoding="utf-8")
    print("SMOKE OK — timing flags / policy path")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--out", default="research/results_momentum_entry_timing_chase_avoidance_v1")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start).normalize()

    if args.synthetic:
        synthetic_smoke(out_dir)
        return

    helper = load_research_helper()
    build_feature_dataset = helper.build_feature_dataset
    metrics_from_daily = helper.metrics_from_daily
    run_engine_backtest = helper.run_engine_backtest

    import config as C
    import data as DA
    import engine as E
    import portfolio as P

    years = max(int(args.years), 5)
    panels = {
        "bist": DA.get_panel("bist", years=years, force_full=True),
        "us": DA.get_panel("us", years=years, force_full=True),
    }
    if panels["bist"].empty or panels["us"].empty:
        raise RuntimeError("BIST veya US paneli boş.")
    fx = DA.get_fx(years)
    if fx.empty:
        raise RuntimeError("USD/TRY paneli boş.")
    W = {mk: DA.to_wide(panel) for mk, panel in panels.items()}
    md_b = E.MarketData("bist", W["bist"], spec=C.MARKETS["bist"]["spec"])
    md_u = E.MarketData("us", W["us"], spec=C.MARKETS["us"]["spec"])

    warm = max(260, P.C.RP_WINDOW + 5)
    start_full = max(pd.Timestamp(md_b.dates[warm]), pd.Timestamp(md_u.dates[warm]))
    base_state, _ = run_engine_backtest(md_b, md_u, fx, start_full, None, PRIMARY_COST_RT, None, C.CAPITAL_TL)["BASE"]
    ds = build_feature_dataset(base_state, {"bist": md_b, "us": md_u}, W)
    ds = ds.dropna(subset=["signal_date", "ret_pct"]).copy()
    ds["signal_date"] = normalize_date_series(ds["signal_date"])
    ds["exit_date"] = normalize_date_series(ds["exit_date"])
    ds = ds[ds["signal_date"] >= start].sort_values(["signal_date", "market", "ticker"]).reset_index(drop=True)
    if ds.empty:
        raise RuntimeError("Başlangıç tarihinden sonra trade yok.")

    pred = build_timing_flags(ds)
    usage = policy_usage(pred)
    diag = diagnostics(pred)
    usage.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_policy_usage.csv", index=False)
    diag.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_diagnostics.csv", index=False)

    lookups = {}
    policy_parts = []
    for name in POLICIES:
        z = apply_policy(pred, name)
        lookups[name] = make_lookup(z)
        policy_parts.append(z)
    all_policy = pd.concat(policy_parts, ignore_index=True)
    pred.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_predictions.csv", index=False)
    all_policy.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_trades.csv", index=False)

    last_date = pd.Timestamp(pred["signal_date"].max())
    runs = run_engine_backtest(md_b, md_u, fx, start, last_date, PRIMARY_COST_RT, lookups, C.CAPITAL_TL)
    base_daily = runs["BASE"][1]
    bm = metrics_from_daily(base_daily)
    metrics_rows = []
    for name, (_state, daily) in runs.items():
        m = metrics_from_daily(daily)
        metrics_rows.append({
            "policy": name,
            **m,
            "cagr_delta_vs_base_pp": m["cagr_pct"] - bm["cagr_pct"],
            "maxdd_delta_vs_base_pp": m["maxdd_pct"] - bm["maxdd_pct"],
            "pf_delta_vs_base": m["profit_factor"] - bm["profit_factor"],
        })
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_metrics.csv", index=False)

    calendar = six_month_calendar(last_date, start)
    folds = fold_metrics(runs, calendar)
    folds.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_folds.csv", index=False)

    primary_daily = runs[PRIMARY_POLICY][1]
    mdlt = matched_monthly(base_daily, primary_daily)
    monthly = mdlt.reset_index().rename(columns={"index": "month"})
    monthly.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_monthly.csv", index=False)
    boot = bootstrap(mdlt["delta_pp"])

    stress_rows = []
    for cost in STRESS_COSTS:
        sr = run_engine_backtest(md_b, md_u, fx, start, last_date, float(cost), {"BASE": None, PRIMARY_POLICY: lookups[PRIMARY_POLICY]}, C.CAPITAL_TL)
        c_b = metrics_from_daily(sr["BASE"][1])
        c_p = metrics_from_daily(sr[PRIMARY_POLICY][1])
        stress_rows.append({
            "cost_rt_pct": float(cost),
            "policy": PRIMARY_POLICY,
            "base_cagr_pct": c_b["cagr_pct"],
            "cand_cagr_pct": c_p["cagr_pct"],
            "cagr_delta_vs_base_pp": c_p["cagr_pct"] - c_b["cagr_pct"],
            "base_maxdd_pct": c_b["maxdd_pct"],
            "cand_maxdd_pct": c_p["maxdd_pct"],
            "base_pf": c_b["profit_factor"],
            "cand_pf": c_p["profit_factor"],
        })
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v1_stress.csv", index=False)

    write_report(out_dir, ds, pred, metrics, folds, monthly, stress, diag, usage, boot)
    print("=== GMQ Momentum Entry Timing / Chase Avoidance V1 ===")
    print(metrics[["policy", "cagr_pct", "maxdd_pct", "profit_factor", "cagr_delta_vs_base_pp"]].to_string(index=False))
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
