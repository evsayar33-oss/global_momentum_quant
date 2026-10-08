#!/usr/bin/env python3
"""
Global Momentum Quant — Cross-Sectional Alpha Quality / Capital Efficiency V1

RESEARCH ONLY.

Purpose
-------
Test whether capital efficiency improves when the existing BASE portfolio
reduces the BUY allocation of the weakest selected names inside each
market/day cross-section, using only signal-date information already present
in the production-aligned trade dataset.

Core hypothesis
---------------
The portfolio already selects a cross-sectional winner set. The hypothesis is
NOT that we can forecast absolute future returns. Instead, among the names that
BASE has already selected on the same signal day, the weakest-ranked selections
may deserve less capital than the strongest selections.

Point-in-time / leakage rules
-----------------------------
1) Production files are NOT modified.
2) Historical trades are generated once through the live production engine.
3) Quality components are computed only from signal-date fields.
4) Cross-sectional ranks are computed WITHIN market + signal_date using only
   the trades selected for that same day. No future observation is used.
5) No ML model, optimizer, parameter search, or OOS tuning is used.
6) The policy family is frozen before the backtest.
7) Only BUY amounts are scaled in the research wrapper. Sells, stops, ranking,
   tranche timing, risk parity and cash mechanics stay unchanged.
8) Primary evaluation begins at 2018-01-01. Six-month fold reports are
   descriptive robustness blocks, not training folds.

Capital-efficiency score
------------------------
For each selected trade, build a fixed weighted score from same-day
cross-sectional percent ranks:

  50% score_pctile rank       (higher BASE conviction)
  20% ret_21d rank            (medium-horizon momentum)
  15% stock_vs_market_21d rank(relative momentum)
  10% ret_63d rank            (longer-horizon confirmation)
   5% inverse momentum_extension_z rank (less chase = better)

Each component is ranked within market + signal_date across that day's
selected names. The score is therefore relative and regime-robust rather than
an absolute return forecast.

Policies (pre-declared)
------------------------
BASE                 = 1.00x for all
CE_GENTLE            = 0.90x for bottom 20% of daily market cohort
CE_BALANCED          = 0.80x for bottom 20%
CE_STRONG            = 0.70x for bottom 20%
CE_BOTTOM30_GENTLE   = 0.90x for bottom 30%

The extra bottom-30% policy is a pre-declared sensitivity check, not an
optimizer. The PRIMARY policy is CE_BALANCED.

Outputs
-------
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_raporu.md
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_metrics.csv
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_folds.csv
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_monthly.csv
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_stress.csv
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_quintiles.csv
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_policy_usage.csv
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_predictions.csv
- gmq_cross_sectional_alpha_quality_capital_efficiency_v1_trades.csv

Run from repository root.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from gmq_tail_risk_alpha_conditional_sizing import (
    PRIMARY_COST_RT,
    STRESS_COSTS,
    build_feature_dataset,
    metrics_from_daily,
    normalize_date_series,
    run_engine_backtest,
)

WF_START = pd.Timestamp("2018-01-01")
REFIT_MONTHS = 6
PRIMARY_POLICY = "CE_BALANCED"
POLICY_DEFS = {
    "BASE": {"bottom_pct": 0.00, "multiplier": 1.00, "family": "base"},
    "CE_GENTLE": {"bottom_pct": 0.20, "multiplier": 0.90, "family": "bottom20"},
    "CE_BALANCED": {"bottom_pct": 0.20, "multiplier": 0.80, "family": "bottom20"},
    "CE_STRONG": {"bottom_pct": 0.20, "multiplier": 0.70, "family": "bottom20"},
    "CE_BOTTOM30_GENTLE": {"bottom_pct": 0.30, "multiplier": 0.90, "family": "bottom30"},
}
COMPONENTS = {
    "score_pctile": 0.50,
    "ret_21d_pct": 0.20,
    "stock_vs_market_21d_pp": 0.15,
    "ret_63d_pct": 0.10,
    "momentum_extension_z": 0.05,
}
REQUIRED = list(COMPONENTS)


def rank01(s: pd.Series, higher_better: bool = True) -> pd.Series:
    """Dense-free percentile rank with deterministic midpoint ties."""
    x = pd.to_numeric(s, errors="coerce")
    r = x.rank(method="average", pct=True, ascending=higher_better)
    return r.clip(0.0, 1.0)


def build_cross_sectional_quality(ds: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in ds.columns]
    if missing:
        raise ValueError(f"Eksik quality feature: {missing}")
    x = ds.copy()
    x["signal_date"] = normalize_date_series(x["signal_date"])
    x["market"] = x["market"].astype(str)
    x["ticker"] = x["ticker"].astype(str)

    parts = []
    for (market, date), g in x.groupby(["market", "signal_date"], sort=True, observed=True):
        z = g.copy()
        z["q_score_rank"] = rank01(z["score_pctile"], True)
        z["q_ret21_rank"] = rank01(z["ret_21d_pct"], True)
        z["q_rel_rank"] = rank01(z["stock_vs_market_21d_pp"], True)
        z["q_ret63_rank"] = rank01(z["ret_63d_pct"], True)
        z["q_extension_rank"] = rank01(z["momentum_extension_z"], False)
        z["capital_efficiency_score"] = (
            COMPONENTS["score_pctile"] * z["q_score_rank"]
            + COMPONENTS["ret_21d_pct"] * z["q_ret21_rank"]
            + COMPONENTS["stock_vs_market_21d_pp"] * z["q_rel_rank"]
            + COMPONENTS["ret_63d_pct"] * z["q_ret63_rank"]
            + COMPONENTS["momentum_extension_z"] * z["q_extension_rank"]
        )
        z["cohort_n"] = len(z)
        parts.append(z)
    if not parts:
        raise RuntimeError("Cross-sectional cohort dataset boş.")
    out = pd.concat(parts, ignore_index=True)
    out["ce_rank_pctile"] = out.groupby(["market", "signal_date"], observed=True)["capital_efficiency_score"].rank(method="average", pct=True)
    return out


def apply_policy(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in POLICY_DEFS:
        raise KeyError(name)
    spec = POLICY_DEFS[name]
    x = df.copy()
    # A zero-bottom BASE policy is always full allocation.
    if spec["bottom_pct"] <= 0:
        x["allocation_condition"] = False
        x["sizing_multiplier"] = 1.0
        return x
    # Weakest cross-sectional cohort: lower CE rank is worse.
    threshold = spec["bottom_pct"]
    x["allocation_condition"] = x["ce_rank_pctile"] <= threshold + 1e-12
    x["sizing_multiplier"] = np.where(
        x["allocation_condition"], spec["multiplier"], 1.0
    )
    return x


def make_lookup(pred: pd.DataFrame) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for r in pred.to_dict("records"):
        d = pd.Timestamp(r["signal_date"]).strftime("%Y-%m-%d")
        out[(str(r["market"]), str(r["ticker"]), d)] = float(r["sizing_multiplier"])
    return out


def make_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ce_quintile"] = x.groupby("market", observed=True)["capital_efficiency_score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if len(s) >= 5 else np.nan
    )
    rows = []
    for (mk, q), g in x.dropna(subset=["ce_quintile"]).groupby(["market", "ce_quintile"], observed=True):
        rows.append({
            "market": mk,
            "quintile": int(q),
            "n": len(g),
            "mean_ce_score": float(g["capital_efficiency_score"].mean()),
            "win_rate_pct": float((g["ret_pct"] > 0).mean() * 100.0),
            "mean_ret_pct": float(g["ret_pct"].mean()),
            "mean_loss_pct": float(g.loc[g["ret_pct"] < 0, "ret_pct"].mean()) if (g["ret_pct"] < 0).any() else np.nan,
            "tail10_rate_pct": float((g["ret_pct"] <= -10).mean() * 100.0),
            "tail15_rate_pct": float((g["ret_pct"] <= -15).mean() * 100.0),
        })
    return pd.DataFrame(rows)


def fold_calendar(last_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    s = WF_START
    fold = 0
    while s < last_date:
        e = min(s + pd.DateOffset(months=REFIT_MONTHS), last_date + pd.Timedelta(days=1))
        rows.append({"fold": fold, "oos_start": s, "oos_end": e})
        fold += 1
        s = e
    return pd.DataFrame(rows)


def fold_metrics(runs: Dict[str, Tuple[dict, pd.Series]], calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in calendar.to_dict("records"):
        t0 = pd.Timestamp(row["oos_start"])
        t1 = pd.Timestamp(row["oos_end"])
        b = runs["BASE"][1]
        for name, (_st, s) in runs.items():
            sb = s[(s.index >= t0) & (s.index < t1)]
            bb = b[(b.index >= t0) & (b.index < t1)]
            if len(sb) < 2 or len(bb) < 2:
                continue
            m = metrics_from_daily(sb)
            bm = metrics_from_daily(bb)
            rows.append({
                "fold": int(row["fold"]),
                "oos_start": t0,
                "oos_end": t1,
                "policy": name,
                "cagr_annualized_pct": float(((sb.iloc[-1] / sb.iloc[0]) ** (365.25 / max((sb.index[-1] - sb.index[0]).days, 1)) - 1) * 100),
                "period_return_pct": float((sb.iloc[-1] / sb.iloc[0] - 1) * 100),
                "base_period_return_pct": float((bb.iloc[-1] / bb.iloc[0] - 1) * 100),
                "period_return_delta_pp": float((sb.iloc[-1] / sb.iloc[0] - bb.iloc[-1] / bb.iloc[0]) * 100),
                "maxdd_pct": m["maxdd_pct"],
                "base_maxdd_pct": bm["maxdd_pct"],
                "pf": m["profit_factor"],
            })
    return pd.DataFrame(rows)


def bootstrap_monthly(delta_pp: pd.Series, n_boot: int = 10000, seed: int = 20261008):
    x = pd.to_numeric(delta_pp, errors="coerce").dropna().to_numpy(float)
    if len(x) < 10:
        return {"mean_pp": np.nan, "ci5_pp": np.nan, "ci95_pp": np.nan, "p_le0": np.nan, "positive_month_pct": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return {
        "mean_pp": float(x.mean()),
        "ci5_pp": float(np.quantile(means, 0.05)),
        "ci95_pp": float(np.quantile(means, 0.95)),
        "p_le0": float((means <= 0).mean()),
        "positive_month_pct": float((x > 0).mean() * 100.0),
    }


def usage_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in POLICY_DEFS:
        z = apply_policy(pred, name)
        rows.append({
            "policy": name,
            "bottom_pct": POLICY_DEFS[name]["bottom_pct"] * 100,
            "multiplier": POLICY_DEFS[name]["multiplier"],
            "n": len(z),
            "condition_n": int(z["allocation_condition"].sum()),
            "condition_pct": float(z["allocation_condition"].mean() * 100),
            "mean_multiplier": float(z["sizing_multiplier"].mean()),
            "median_multiplier": float(z["sizing_multiplier"].median()),
        })
    return pd.DataFrame(rows)


def write_report(out_dir: Path, ds: pd.DataFrame, preds: pd.DataFrame, metrics: pd.DataFrame, folds: pd.DataFrame, monthly: pd.DataFrame, stress: pd.DataFrame, quintiles: pd.DataFrame, usage: pd.DataFrame, boot: dict) -> None:
    primary = metrics.loc[metrics.policy == PRIMARY_POLICY].iloc[0]
    base = metrics.loc[metrics.policy == "BASE"].iloc[0]
    p_folds = folds.loc[folds.policy == PRIMARY_POLICY].copy()
    positive_fold_pct = float((p_folds["period_return_delta_pp"] > 0).mean() * 100.0) if len(p_folds) else np.nan
    stress_p = stress.loc[stress.policy == PRIMARY_POLICY]
    stress_positive = int((stress_p["cagr_delta_vs_base_pp"] > 0).sum())
    verdict = "PROMISING — NEEDS HOLDOUT/WF FOLLOW-UP" if (
        primary["cagr_delta_vs_base_pp"] >= 0
        and boot["ci5_pp"] > 0
        and positive_fold_pct >= 60
        and stress_positive >= 4
    ) else "REJECT FOR PROMOTION"
    lines = [
        "# Global Momentum Quant — Cross-Sectional Alpha Quality / Capital Efficiency V1",
        "",
        "**RESEARCH ONLY — PRODUCTION DOSYALARINA DOKUNULMADI**",
        "",
        "## 1. Hipotez",
        "BASE zaten cross-sectional winner setini seçiyor. Bu araştırma mutlak gelecekteki getiriyi tahmin etmek yerine, aynı market/sinyal gününde seçilmiş hisseler içindeki göreli kalite farkının sermaye tahsisinde kullanılabilir olup olmadığını test eder.",
        "",
        "## 2. Sabit score",
        "- 50% `score_pctile`",
        "- 20% `ret_21d_pct`",
        "- 15% `stock_vs_market_21d_pp`",
        "- 10% `ret_63d_pct`",
        "- 5% inverse `momentum_extension_z`",
        "Tüm bileşenler aynı market + signal_date cross-section içinde rank edilir. Gelecek veri kullanılmaz.",
        "",
        "## 3. Policy family",
        usage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 4. Production-aligned OOS result",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. Cross-sectional quintile diagnostics",
        quintiles.to_markdown(index=False, floatfmt=".4f") if not quintiles.empty else "—",
        "",
        "## 6. Six-month robustness blocks",
        folds.to_markdown(index=False, floatfmt=".4f") if not folds.empty else "—",
        "",
        "## 7. Matched-month bootstrap — primary",
        f"Mean monthly delta: **{boot['mean_pp']:.4f} pp**",
        f"Bootstrap 5–95% CI: **[{boot['ci5_pp']:.4f}, {boot['ci95_pp']:.4f}] pp**",
        f"P(mean delta <= 0): **{boot['p_le0']:.4f}**",
        f"Positive months: **{boot['positive_month_pct']:.2f}%**",
        "",
        "## 8. Cost stress — primary",
        stress.to_markdown(index=False, floatfmt=".4f"),
        f"Positive stress scenarios: **{stress_positive}/{len(stress_p)}**",
        "",
        "## 9. Decision gate",
        "Promotion gate: non-negative primary CAGR delta + bootstrap 5% bound > 0 + positive six-month fold ratio >= 60% + >=4/5 cost-stress scenarios positive.",
        "",
        f"# FINAL VERDICT\n**{verdict}**",
        "",
        "Notably, this V1 has no ML model and no fitted threshold. It is deliberately a simple cross-sectional capital-efficiency test.",
    ]
    (out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_raporu.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def synthetic_smoke(out_dir: Path) -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2017-01-02", periods=2500)
    rows = []
    for d in dates:
        for mk in ("bist", "us"):
            n = 10
            base = np.arange(n) / max(n - 1, 1)
            score = base * 100
            ret21 = base * 20 + rng.normal(0, 2, n)
            rel = base * 10 + rng.normal(0, 1.5, n)
            ret63 = base * 25 + rng.normal(0, 3, n)
            ext = rng.normal(0, 1, n) - base
            ret = 0.5 + 4.0 * base + rng.normal(0, 5, n)
            for i in range(n):
                rows.append({
                    "market": mk,
                    "ticker": f"T{i:03d}",
                    "signal_date": d,
                    "entry_date": d + pd.Timedelta(days=1),
                    "exit_date": d + pd.Timedelta(days=21),
                    "ret_pct": ret[i],
                    "score_pctile": score[i],
                    "ret_21d_pct": ret21[i],
                    "stock_vs_market_21d_pp": rel[i],
                    "ret_63d_pct": ret63[i],
                    "momentum_extension_z": ext[i],
                })
    ds = pd.DataFrame(rows)
    pred = build_cross_sectional_quality(ds)
    quint = make_quintiles(pred)
    usage = usage_table(pred)
    assert not pred.empty and pred["capital_efficiency_score"].between(0, 1).all()
    assert (pred.groupby(["market", "signal_date"], observed=True)["ce_rank_pctile"].min() > 0).all()
    primary = apply_policy(pred, PRIMARY_POLICY)
    assert (primary.loc[primary["allocation_condition"], "sizing_multiplier"] == 0.8).all()
    pred.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_predictions.csv", index=False)
    quint.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_quintiles.csv", index=False)
    usage.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_policy_usage.csv", index=False)
    print("SMOKE OK — cross-sectional rank / policy path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=14)
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out", default="research/results_cross_sectional_alpha_quality_capital_efficiency_v1")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    global WF_START
    WF_START = pd.Timestamp(args.start).normalize()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        synthetic_smoke(out_dir)
        return

    import config as C
    import data as DA
    import engine as E
    import portfolio as P

    years = max(int(args.years), 5)
    panels = {"bist": DA.get_panel("bist", years=years, force_full=True), "us": DA.get_panel("us", years=years, force_full=True)}
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
    ds["signal_date"] = normalize_date_series(ds["signal_date"])
    ds["exit_date"] = normalize_date_series(ds["exit_date"])
    ds = ds[ds["signal_date"] >= WF_START].copy()
    if ds.empty:
        raise RuntimeError("WF başlangıcından sonra trade yok.")

    pred = build_cross_sectional_quality(ds)
    usage = usage_table(pred)
    quint = make_quintiles(pred)
    lookups = {}
    policy_pred_parts = []
    for name in POLICY_DEFS:
        z = apply_policy(pred, name)
        z["policy"] = name
        policy_pred_parts.append(z)
        lookups[name] = make_lookup(z)
    all_policy_trades = pd.concat(policy_pred_parts, ignore_index=True)

    last_date = pd.Timestamp(ds["signal_date"].max())
    runs = run_engine_backtest(md_b, md_u, fx, WF_START, last_date, PRIMARY_COST_RT, lookups, C.CAPITAL_TL)
    base_daily = runs["BASE"][1]
    base_m = metrics_from_daily(base_daily)
    metrics_rows = []
    for name, (_st, daily) in runs.items():
        m = metrics_from_daily(daily)
        metrics_rows.append({
            "policy": name,
            **m,
            "cagr_delta_vs_base_pp": m["cagr_pct"] - base_m["cagr_pct"],
            "maxdd_delta_vs_base_pp": m["maxdd_pct"] - base_m["maxdd_pct"],
            "pf_delta_vs_base": m["profit_factor"] - base_m["profit_factor"],
        })
    metrics = pd.DataFrame(metrics_rows)

    cal = fold_calendar(last_date)
    folds = fold_metrics(runs, cal)
    primary_daily = runs[PRIMARY_POLICY][1]
    mdlt = pd.concat([
        base_daily.resample("ME").last().pct_change().rename("base_ret"),
        primary_daily.resample("ME").last().pct_change().rename("primary_ret"),
    ], axis=1).dropna()
    mdlt["delta_pp"] = (mdlt["primary_ret"] - mdlt["base_ret"]) * 100
    monthly = mdlt.reset_index().rename(columns={"index": "month"})
    boot = bootstrap_monthly(mdlt["delta_pp"])

    stress_rows = []
    primary_lookup = lookups[PRIMARY_POLICY]
    for cost in STRESS_COSTS:
        sr = run_engine_backtest(md_b, md_u, fx, WF_START, last_date, float(cost), {"BASE": None, PRIMARY_POLICY: primary_lookup}, C.CAPITAL_TL)
        bm = metrics_from_daily(sr["BASE"][1])
        cm = metrics_from_daily(sr[PRIMARY_POLICY][1])
        stress_rows.append({
            "cost_rt_pct": float(cost),
            "policy": PRIMARY_POLICY,
            "base_cagr_pct": bm["cagr_pct"],
            "cand_cagr_pct": cm["cagr_pct"],
            "cagr_delta_vs_base_pp": cm["cagr_pct"] - bm["cagr_pct"],
            "base_maxdd_pct": bm["maxdd_pct"],
            "cand_maxdd_pct": cm["maxdd_pct"],
            "base_pf": bm["profit_factor"],
            "cand_pf": cm["profit_factor"],
        })
    stress = pd.DataFrame(stress_rows)

    pred.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_predictions.csv", index=False)
    all_policy_trades.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_trades.csv", index=False)
    metrics.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_metrics.csv", index=False)
    folds.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_folds.csv", index=False)
    monthly.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_monthly.csv", index=False)
    stress.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_stress.csv", index=False)
    quint.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_quintiles.csv", index=False)
    usage.to_csv(out_dir / "gmq_cross_sectional_alpha_quality_capital_efficiency_v1_policy_usage.csv", index=False)

    write_report(out_dir, ds, pred, metrics, folds, monthly, stress, quint, usage, boot)
    print("=== GMQ Cross-Sectional Alpha Quality / Capital Efficiency V1 ===")
    print(metrics[["policy", "cagr_pct", "maxdd_pct", "profit_factor", "cagr_delta_vs_base_pp"]].to_string(index=False))
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
