#!/usr/bin/env python3
"""
Global Momentum Quant — Alpha-Quality Conditional Allocation V1
Walk-Forward Validation

RESEARCH ONLY.

This test validates the already-defined Alpha-Quality Conditional Allocation V1
hypothesis out-of-sample using an expanding-window walk-forward process.

Primary candidate
-----------------
COND_STRONG:
    alpha percentile <= 25% AND tail-risk percentile <= 50%
    -> BUY multiplier 0.50

No new model family, threshold, multiplier or feature is optimized here.
The original V1 policy definitions are frozen before the walk-forward run.

Walk-forward protocol
---------------------
- Historical production trade set is generated once with the real engine path.
- Walk-forward starts: 2018-01-01.
- Refit cadence: every 6 calendar months.
- For fold [t0, t1): fit only trades with signal_date < t0 AND exit_date < t0.
- The fold's model and training-only empirical CDF are then frozen for [t0, t1).
- No OOS observation is used to fit models or thresholds for that fold.
- Candidate BUY sizing is applied only inside the research wrapper.
- Production source files are never modified.

Outputs
-------
- gmq_alpha_quality_walk_forward_validation_v1_raporu.md
- gmq_alpha_quality_walk_forward_validation_v1_metrics.csv
- gmq_alpha_quality_walk_forward_validation_v1_folds.csv
- gmq_alpha_quality_walk_forward_validation_v1_monthly.csv
- gmq_alpha_quality_walk_forward_validation_v1_stress.csv
- gmq_alpha_quality_walk_forward_validation_v1_model_metrics.csv
- gmq_alpha_quality_walk_forward_validation_v1_threshold_drift.csv
- gmq_alpha_quality_walk_forward_validation_v1_policy_usage.csv
- gmq_alpha_quality_walk_forward_validation_v1_predictions.csv
- gmq_alpha_quality_walk_forward_validation_v1_trades.csv

Run from repository root.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from gmq_tail_risk_alpha_conditional_sizing import (
    PRIMARY_COST_RT,
    STRESS_COSTS,
    STRICT_FEATURES,
    build_feature_dataset,
    fit_models,
    metrics_from_daily,
    normalize_date_series,
    predict_models,
    run_engine_backtest,
)
from gmq_alpha_quality_conditional_allocation_v1 import POLICIES, policy_multiplier

ROOT = Path(__file__).resolve().parent
WF_START = pd.Timestamp("2018-01-01")
REFIT_MONTHS = 6
MIN_TRAIN_N = 1000
PRIMARY_POLICY = "COND_STRONG"


def safe_auc(y: pd.Series, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan")


def safe_pr_auc(y: pd.Series, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if y.nunique() == 2 else float("nan")


def make_fold_calendar(last_date: pd.Timestamp) -> pd.DataFrame:
    rows = []
    start = WF_START
    fold = 0
    while start < last_date:
        end = min(start + pd.DateOffset(months=REFIT_MONTHS), last_date + pd.Timedelta(days=1))
        rows.append(
            {
                "fold": fold,
                "fit_cutoff": start,
                "oos_start": start,
                "oos_end": end,
            }
        )
        fold += 1
        start = end
    return pd.DataFrame(rows)


def fold_period_metrics(daily: pd.Series) -> Dict[str, float]:
    s = daily.dropna().astype(float).sort_index()
    if len(s) < 2:
        return {}
    r = s.pct_change().dropna()
    pos = r[r > 0]
    neg = -r[r < 0]
    pf = float(pos.sum() / neg.sum()) if len(neg) and neg.sum() > 0 else float("nan")
    peak = s.cummax()
    dd = s / peak - 1.0
    return {
        "period_return_pct": float((s.iloc[-1] / s.iloc[0] - 1.0) * 100.0),
        "maxdd_pct": float(dd.min() * 100.0),
        "profit_factor": pf,
        "daily_win_pct": float((r > 0).mean() * 100.0) if len(r) else float("nan"),
        "n_days": len(r),
    }


def bootstrap_stats(delta_pp: pd.Series, n_boot: int = 10000, seed: int = 20261008):
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


def build_fold_predictions(ds: pd.DataFrame, calendar: pd.DataFrame):
    pred_parts = []
    model_rows = []
    threshold_rows = []
    usage_rows = []
    policy_fold_frames = []

    for row in calendar.to_dict("records"):
        fold = int(row["fold"])
        t0 = pd.Timestamp(row["oos_start"])
        t1 = pd.Timestamp(row["oos_end"])

        fit = ds[(ds["signal_date"] < t0) & (ds["exit_date"] < t0)].copy()
        hold = ds[(ds["signal_date"] >= t0) & (ds["signal_date"] < t1)].copy()
        if hold.empty:
            continue
        if len(fit) < MIN_TRAIN_N:
            raise RuntimeError(f"Fold {fold}: insufficient expanding train set: {len(fit)} < {MIN_TRAIN_N}")

        risk_models, alpha_model, mm = fit_models(fit)
        fit_pred = predict_models(fit, risk_models, alpha_model)
        hold_pred = predict_models(hold, risk_models, alpha_model)

        # Training-only empirical CDFs, performed explicitly per market.
        fit_pred["sample"] = "train"
        hold_pred["sample"] = "walk_forward_oos"
        for market in sorted(set(fit_pred["market"].astype(str)) | set(hold_pred["market"].astype(str))):
            tr = fit_pred[fit_pred["market"].astype(str) == market]
            ho_idx = hold_pred[hold_pred["market"].astype(str) == market].index
            if len(ho_idx) == 0:
                continue
            a = tr["pred_alpha_pct"].to_numpy(float)
            rr = tr["tail_risk_score"].to_numpy(float)
            vals_a = hold_pred.loc[ho_idx, "pred_alpha_pct"].to_numpy(float)
            vals_r = hold_pred.loc[ho_idx, "tail_risk_score"].to_numpy(float)
            # Vectorized mid-rank CDF against FIT only.
            hold_pred.loc[ho_idx, "alpha_pctile_mkt"] = [
                ((a <= v).sum() - 0.5 * (a == v).sum()) / len(a) if np.isfinite(v) else np.nan
                for v in vals_a
            ]
            hold_pred.loc[ho_idx, "risk_pctile_mkt"] = [
                ((rr <= v).sum() - 0.5 * (rr == v).sum()) / len(rr) if np.isfinite(v) else np.nan
                for v in vals_r
            ]

            # Record the actual score thresholds implied by the training CDF.
            threshold_rows.append(
                {
                    "fold": fold,
                    "market": market,
                    "fit_n": len(tr),
                    "alpha_q25_score": float(np.nanquantile(a, 0.25)),
                    "risk_q50_score": float(np.nanquantile(rr, 0.50)),
                    "fit_alpha_pred_mean": float(np.nanmean(a)),
                    "fit_alpha_pred_std": float(np.nanstd(a)),
                    "fit_risk_score_mean": float(np.nanmean(rr)),
                    "fit_risk_score_std": float(np.nanstd(rr)),
                }
            )

        hold_pred["fold"] = fold
        hold_pred["fit_cutoff"] = t0
        hold_pred["oos_end"] = t1
        pred_parts.append(hold_pred)

        # Fold-specific model diagnostics. Fit metrics plus genuine OOS AUC/PR-AUC.
        mm = mm.copy()
        for target in ("tail_5", "tail_10", "tail_15"):
            y = (hold["ret_pct"] <= float(target.split("_")[1])).astype(int)
            p = hold_pred[f"p_{target}"].to_numpy(float)
            model_rows.append(
                {
                    "fold": fold,
                    "fit_cutoff": t0,
                    "oos_end": t1,
                    "model": f"risk_{target}",
                    "fit_n": len(fit),
                    "oos_n": len(hold),
                    "auc_oos": safe_auc(y, p),
                    "pr_auc_oos": safe_pr_auc(y, p),
                    "auc_train": float(mm.loc[mm.model == f"risk_{target}", "auc_train"].iloc[0]),
                }
            )
        model_rows.append(
            {
                "fold": fold,
                "fit_cutoff": t0,
                "oos_end": t1,
                "model": "alpha_ridge",
                "fit_n": len(fit),
                "oos_n": len(hold),
                "r2_train": float(mm.loc[mm.model == "alpha_ridge", "r2_train"].iloc[0]),
                "mae_train": float(mm.loc[mm.model == "alpha_ridge", "mae_train"].iloc[0]),
                "alpha_oos_mean_pred_pct": float(hold_pred["pred_alpha_pct"].mean()),
            }
        )

        # Freeze the full V1 policy family for this fold.
        for name, spec in POLICIES.items():
            z = policy_multiplier(hold_pred, spec)
            z["fold"] = fold
            z["policy"] = name
            z["fit_cutoff"] = t0
            z["oos_end"] = t1
            policy_fold_frames.append(z)
            cond = z["allocation_condition"].astype(bool)
            usage_rows.append(
                {
                    "fold": fold,
                    "policy": name,
                    "family": spec["family"],
                    "oos_n": len(z),
                    "condition_n": int(cond.sum()),
                    "condition_pct": float(cond.mean() * 100.0),
                    "mean_multiplier": float(z["sizing_multiplier"].mean()),
                    "median_multiplier": float(z["sizing_multiplier"].median()),
                    "alpha_cut": spec["alpha_cut"],
                    "risk_cut": spec["risk_cut"],
                    "multiplier": spec["multiplier"],
                }
            )

    pred = pd.concat(pred_parts, ignore_index=True)
    models = pd.DataFrame(model_rows)
    thresholds = pd.DataFrame(threshold_rows)
    usage = pd.DataFrame(usage_rows)
    policy_df = pd.concat(policy_fold_frames, ignore_index=True)
    return pred, models, thresholds, usage, policy_df


def make_lookups(policy_df: pd.DataFrame) -> Dict[str, Dict[Tuple[str, str, str], float]]:
    lookups: Dict[str, Dict[Tuple[str, str, str], float]] = {k: {} for k in POLICIES}
    cols = ["policy", "market", "ticker", "signal_date", "sizing_multiplier"]
    for r in policy_df[cols].to_dict("records"):
        key = (str(r["market"]), str(r["ticker"]), pd.Timestamp(r["signal_date"]).strftime("%Y-%m-%d"))
        lookups[str(r["policy"])] [key] = float(r["sizing_multiplier"])
    return lookups


def attach_fold_to_daily(daily: pd.Series, calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in daily.items():
        d, nav = r
        fold = calendar[(calendar.oos_start <= pd.Timestamp(d)) & (pd.Timestamp(d) < calendar.oos_end)]
        if fold.empty:
            continue
        rows.append({"date": pd.Timestamp(d), "nav": float(nav), "fold": int(fold.iloc[0].fold)})
    return pd.DataFrame(rows)


def fold_performance(runs: Dict[str, Tuple[dict, pd.Series]], calendar: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_df = attach_fold_to_daily(runs["BASE"][1], calendar)
    for name, (_state, daily) in runs.items():
        cand_df = attach_fold_to_daily(daily, calendar)
        for fold in calendar["fold"]:
            b = base_df[base_df.fold == fold].set_index("date")["nav"]
            c = cand_df[cand_df.fold == fold].set_index("date")["nav"]
            common = b.index.intersection(c.index)
            if len(common) < 2:
                continue
            bm = fold_period_metrics(b.loc[common])
            cm = fold_period_metrics(c.loc[common])
            rows.append(
                {
                    "policy": name,
                    "fold": int(fold),
                    "oos_start": str(calendar.loc[calendar.fold == fold, "oos_start"].iloc[0].date()),
                    "oos_end": str(calendar.loc[calendar.fold == fold, "oos_end"].iloc[0].date()),
                    "base_period_return_pct": bm["period_return_pct"],
                    "cand_period_return_pct": cm["period_return_pct"],
                    "period_delta_pp": cm["period_return_pct"] - bm["period_return_pct"],
                    "base_maxdd_pct": bm["maxdd_pct"],
                    "cand_maxdd_pct": cm["maxdd_pct"],
                    "base_pf": bm["profit_factor"],
                    "cand_pf": cm["profit_factor"],
                    "base_daily_win_pct": bm["daily_win_pct"],
                    "cand_daily_win_pct": cm["daily_win_pct"],
                    "base_n_days": bm["n_days"],
                }
            )
    return pd.DataFrame(rows)


def market_usage(policy_df: pd.DataFrame, primary: str) -> pd.DataFrame:
    x = policy_df[policy_df.policy == primary].copy()
    x["condition"] = x["allocation_condition"].astype(bool)
    rows = []
    for fold, g in x.groupby("fold"):
        for market, h in g.groupby("market"):
            rows.append(
                {
                    "fold": int(fold),
                    "market": str(market),
                    "oos_n": len(h),
                    "condition_n": int(h.condition.sum()),
                    "condition_pct": float(h.condition.mean() * 100.0),
                    "mean_multiplier": float(h.sizing_multiplier.mean()),
                }
            )
    return pd.DataFrame(rows)


def report(
    out_dir: Path,
    ds: pd.DataFrame,
    pred: pd.DataFrame,
    models: pd.DataFrame,
    thresholds: pd.DataFrame,
    usage: pd.DataFrame,
    folds: pd.DataFrame,
    metrics: pd.DataFrame,
    monthly: pd.DataFrame,
    stress: pd.DataFrame,
    primary_boot: dict,
    market_use: pd.DataFrame,
):
    primary_row = metrics[metrics.policy == PRIMARY_POLICY].iloc[0].to_dict()
    primary_folds = folds[folds.policy == PRIMARY_POLICY]
    pos_fold = float((primary_folds.period_delta_pp > 0).mean() * 100.0) if len(primary_folds) else np.nan
    stress_primary = stress[stress.policy == PRIMARY_POLICY]
    stress_pos = int((stress_primary.cagr_delta_vs_base_pp > 0).sum()) if len(stress_primary) else 0
    ci_positive = bool(np.isfinite(primary_boot["ci5_pp"]) and primary_boot["ci5_pp"] > 0)
    overall_positive = bool(primary_row["cagr_delta_vs_base_pp"] >= 0)
    fold_positive = bool(pos_fold >= 60.0)
    stress_ok = bool(stress_pos >= 4)
    verdict = "WALK-FORWARD PASS CANDIDATE" if (overall_positive and ci_positive and fold_positive and stress_ok) else "WALK-FORWARD FAIL / DO NOT PROMOTE"

    lines = [
        "# Global Momentum Quant — Alpha-Quality Conditional Allocation V1 — Walk-Forward Validation",
        "",
        f"_Üretim: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')} · WF start: {WF_START.date()} · refit: {REFIT_MONTHS} months · primary policy: {PRIMARY_POLICY}_",
        "",
        "## 1. Amaç",
        "Alpha-Quality Conditional Allocation V1 hipotezinin tek bir 2022+ holdout yerine expanding-window walk-forward yöntemiyle zaman içinde tekrarlanıp tekrarlanmadığı test edilmiştir.",
        "",
        "## 2. Leakage / protocol",
        f"- Toplam BASE trade dataset: **{len(ds):,}**",
        f"- Walk-forward OOS trade satırı: **{len(pred):,}**",
        f"- İlk OOS tarihi: **{WF_START.date()}**",
        f"- Refit: **{REFIT_MONTHS} ayda bir**",
        "- Her fold'da fit seti yalnızca `signal_date < fold_start` ve `exit_date < fold_start` işlemleridir.",
        "- Fold içindeki model ve training-only CDF eşikleri tüm OOS blok boyunca kilitlidir.",
        "- Hiçbir OOS gözlemi sonraki fold fit'inde kullanılmadan önce gerçekleşmiş exit şartını sağlamalıdır.",
        "- Production engine / portfolio / config / signals / state dosyaları değiştirilmez.",
        "",
        "## 3. Frozen policy family",
        "| Policy | Condition | Multiplier |",
        "|---|---|---:|",
    ]
    for name, spec in POLICIES.items():
        cond = "none" if name == "BASE" else ("alpha <= Q25" if spec["family"] == "PURE_ALPHA" else "alpha <= Q25 AND risk <= Q50")
        lines.append(f"| {name} | {cond} | {spec['multiplier']:.2f} |")
    lines += [
        "",
        "No new policy threshold or multiplier is selected from walk-forward OOS results.",
        "",
        "## 4. Primary OOS performance",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. Fold stability",
        f"Primary positive-fold ratio: **{pos_fold:.2f}%**",
        folds.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. Monthly paired bootstrap — primary",
        f"Mean monthly delta: **{primary_boot['mean_pp']:.4f} pp**",
        f"Bootstrap 5–95% CI: **[{primary_boot['ci5_pp']:.4f}, {primary_boot['ci95_pp']:.4f}] pp**",
        f"P(mean delta <= 0): **{primary_boot['p_le0']:.4f}**",
        f"Positive-month ratio: **{primary_boot['positive_month_pct']:.2f}%**",
        "",
        "## 7. Cost stress — primary",
        stress.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Positive stress scenarios for {PRIMARY_POLICY}: **{stress_pos}/{len(stress_primary)}**",
        "",
        "## 8. Model stability",
        models.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 9. Threshold drift",
        thresholds.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 10. Policy usage",
        usage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 11. Primary market usage",
        market_use.to_markdown(index=False, floatfmt=".4f") if not market_use.empty else "—",
        "",
        "## 12. Decision rule",
        "Primary policy is considered a walk-forward candidate only when overall CAGR delta is non-negative, monthly bootstrap 5% bound is above zero, positive-fold ratio is at least 60%, and at least 4/5 primary cost-stress scenarios retain positive CAGR delta.",
        "",
        "# FINAL VERDICT",
        f"**{verdict}**",
        "",
        "Production integration is NOT performed by this research package.",
    ]
    (out_dir / "gmq_alpha_quality_walk_forward_validation_v1_raporu.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def synthetic_smoke(out_dir: Path):
    rng = np.random.default_rng(123)
    n = 5000
    dates = pd.bdate_range("2013-01-01", periods=n)
    market = np.where(np.arange(n) % 2, "us", "bist")
    ds = pd.DataFrame({
        "market": market,
        "ticker": [f"T{i%50:03d}" for i in range(n)],
        "signal_date": dates,
        "entry_date": dates + pd.Timedelta(days=1),
        "exit_date": dates + pd.Timedelta(days=21),
        "ret_pct": rng.normal(2.5, 6.0, n),
    })
    for c in STRICT_FEATURES:
        ds[c] = rng.normal(0, 1, n)
    ds["score_pctile"] = np.clip(rng.normal(50, 15, n), 0, 100)
    ds["ret_5d_pct"] = rng.normal(3, 7, n)
    ds["ret_21d_pct"] = rng.normal(7, 12, n)
    ds["ret_63d_pct"] = rng.normal(15, 20, n)
    ds["vol20_ann_pct"] = np.abs(rng.normal(22, 7, n))
    ds["atr14_pct"] = np.abs(rng.normal(4, 1.2, n))
    ds["market_breadth_5d"] = rng.uniform(.1,.9,n)
    ds["market_breadth_21d"] = rng.uniform(.1,.9,n)
    ds["market_median_1d_pct"] = rng.normal(0, 2, n)
    ds["market_median_21d_pct"] = rng.normal(0, 4, n)
    ds["market_median_63d_pct"] = rng.normal(0, 8, n)
    ds["market_dispersion_1d_pct"] = np.abs(rng.normal(3, 1, n))
    ds["stock_vs_market_21d_pp"] = rng.normal(0, 8, n)
    ds["momentum_extension_z"] = rng.normal(0, 1, n)
    ds["signal_date"] = normalize_date_series(ds["signal_date"])
    ds["entry_date"] = normalize_date_series(ds["entry_date"])
    ds["exit_date"] = normalize_date_series(ds["exit_date"])
    cal = make_fold_calendar(ds["signal_date"].max())
    pred, mm, th, usage, policy_df = build_fold_predictions(ds, cal)
    assert not pred.empty and not mm.empty and not th.empty and not usage.empty and not policy_df.empty
    lookups = make_lookups(policy_df)
    assert set(lookups) == set(POLICIES)
    primary = policy_df[policy_df.policy == PRIMARY_POLICY]
    assert not primary.empty
    pred.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_predictions.csv", index=False)
    mm.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_model_metrics.csv", index=False)
    th.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_threshold_drift.csv", index=False)
    usage.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_policy_usage.csv", index=False)
    policy_df.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_trades.csv", index=False)
    print("SMOKE OK — synthetic walk-forward model/protocol path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=14)
    ap.add_argument("--start", default="2018-01-01", help="walk-forward OOS start; default 2018-01-01")
    ap.add_argument("--refit-months", type=int, default=6)
    ap.add_argument("--out", default="research/results_alpha_quality_walk_forward_validation_v1")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    globals()["WF_START"] = pd.Timestamp(args.start).normalize()
    globals()["REFIT_MONTHS"] = int(args.refit_months)
    if REFIT_MONTHS <= 0:
        raise ValueError("--refit-months pozitif olmalı")

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

    base_state, _ = run_engine_backtest(
        md_b, md_u, fx, start_full, None, PRIMARY_COST_RT, None, C.CAPITAL_TL
    )["BASE"]
    ds = build_feature_dataset(base_state, {"bist": md_b, "us": md_u}, W)
    ds["signal_date"] = normalize_date_series(ds["signal_date"])
    ds["exit_date"] = normalize_date_series(ds["exit_date"])
    last_date = max(pd.Timestamp(ds["signal_date"].max()), pd.Timestamp(md_b.dates[-1]), pd.Timestamp(md_u.dates[-1]))
    if WF_START >= last_date:
        raise RuntimeError(f"WF start {WF_START.date()} >= dataset end {last_date.date()}")

    calendar = make_fold_calendar(last_date)
    pred, model_metrics, thresholds, usage, policy_df = build_fold_predictions(ds, calendar)
    lookups = make_lookups(policy_df)

    # Primary OOS research using the SAME real production mechanics.
    runs = run_engine_backtest(
        md_b,
        md_u,
        fx,
        WF_START,
        last_date,
        PRIMARY_COST_RT,
        lookups,
        C.CAPITAL_TL,
    )

    # Overall metrics and paired monthly diagnostics.
    base_daily = runs["BASE"][1]
    base_m = metrics_from_daily(base_daily)
    metric_rows = []
    for name, (_st, daily) in runs.items():
        m = metrics_from_daily(daily)
        row = {"policy": name, **m}
        row["cagr_delta_vs_base_pp"] = m["cagr_pct"] - base_m["cagr_pct"]
        row["maxdd_delta_vs_base_pp"] = m["maxdd_pct"] - base_m["maxdd_pct"]
        row["pf_delta_vs_base"] = m["profit_factor"] - base_m["profit_factor"]
        metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    folds = fold_performance(runs, calendar)
    monthly = None
    primary_boot = bootstrap_stats(pd.Series([], dtype=float))
    primary_daily = runs[PRIMARY_POLICY][1]
    mdlt = pd.concat(
        [base_daily.resample("ME").last().pct_change().rename("base"), primary_daily.resample("ME").last().pct_change().rename("primary")],
        axis=1,
    ).dropna()
    mdlt["delta_pp"] = (mdlt["primary"] - mdlt["base"]) * 100.0
    monthly = mdlt.reset_index().rename(columns={"date":"month"})
    primary_boot = bootstrap_stats(mdlt["delta_pp"])

    # Cost stress only for the pre-declared PRIMARY policy to control compute
    # while preserving a full robustness check.
    primary_lookup = {k: v for k, v in lookups[PRIMARY_POLICY].items()}
    stress_rows = []
    for cost in STRESS_COSTS:
        stress_runs = run_engine_backtest(
            md_b,
            md_u,
            fx,
            WF_START,
            last_date,
            float(cost),
            {"BASE": None, PRIMARY_POLICY: primary_lookup},
            C.CAPITAL_TL,
        )
        b = metrics_from_daily(stress_runs["BASE"][1])
        c = metrics_from_daily(stress_runs[PRIMARY_POLICY][1])
        stress_rows.append(
            {
                "cost_rt_pct": float(cost),
                "policy": PRIMARY_POLICY,
                "base_cagr_pct": b["cagr_pct"],
                "cand_cagr_pct": c["cagr_pct"],
                "cagr_delta_vs_base_pp": c["cagr_pct"] - b["cagr_pct"],
                "base_maxdd_pct": b["maxdd_pct"],
                "cand_maxdd_pct": c["maxdd_pct"],
                "base_pf": b["profit_factor"],
                "cand_pf": c["profit_factor"],
            }
        )
    stress = pd.DataFrame(stress_rows)

    # Diagnostics / outputs.
    market_use = market_usage(policy_df, PRIMARY_POLICY)
    pred.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_predictions.csv", index=False)
    model_metrics.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_model_metrics.csv", index=False)
    thresholds.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_threshold_drift.csv", index=False)
    usage.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_policy_usage.csv", index=False)
    folds.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_folds.csv", index=False)
    metrics.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_metrics.csv", index=False)
    monthly.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_monthly.csv", index=False)
    stress.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_stress.csv", index=False)
    policy_df.to_csv(out_dir / "gmq_alpha_quality_walk_forward_validation_v1_trades.csv", index=False)

    # One compact fold summary for the primary candidate.
    primary_fold = folds[folds.policy == PRIMARY_POLICY].copy()
    if primary_fold.empty:
        raise RuntimeError("Primary fold performance boş.")

    report(
        out_dir,
        ds,
        pred,
        model_metrics,
        thresholds,
        usage,
        primary_fold,
        metrics,
        monthly,
        stress,
        primary_boot,
        market_use,
    )

    print("=== GMQ Alpha-Quality Conditional Allocation V1 — Walk-Forward Validation ===")
    print(f"dataset={len(ds):,}  wf_oos={len(pred):,}  folds={len(calendar):,}  wf_start={WF_START.date()}")
    print(metrics[["policy", "cagr_pct", "maxdd_pct", "profit_factor", "cagr_delta_vs_base_pp", "maxdd_delta_vs_base_pp"]].to_string(index=False))
    print(f"Primary bootstrap: mean={primary_boot['mean_pp']:.4f}pp CI=[{primary_boot['ci5_pp']:.4f},{primary_boot['ci95_pp']:.4f}] P<=0={primary_boot['p_le0']:.4f}")
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
