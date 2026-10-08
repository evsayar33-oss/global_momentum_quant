#!/usr/bin/env python3
"""
Global Momentum Quant — Alpha-Quality Conditional Allocation V1

RESEARCH ONLY.

Purpose
-------
The previous Tail-Risk x Alpha and Asymmetric Tail-Risk allocation tests showed
that high tail-risk is not automatically bad: high-alpha/high-risk trades can
carry strong positive expectancy, and even low-alpha/high-risk trades remained
positive. The weakest observed quadrant was LOW_ALPHA + LOW_RISK.

This research therefore isolates ALPHA QUALITY rather than penalising tail risk
itself. Two pre-declared families are tested:

A) PURE ALPHA QUALITY
   Penalise only the bottom alpha quartile.

B) ALPHA x LOW-RISK CONDITIONAL
   Penalise only trades in the bottom alpha quartile AND at/below the training
   median tail-risk score. This directly targets the previously observed weak
   LOW_ALPHA + LOW_RISK quadrant while leaving LOW_ALPHA + HIGH_RISK untouched.

No OOS optimization is performed.

Critical design rules
---------------------
1) Production files are NOT modified.
2) The research backtest imports the repository's live engine.py, portfolio.py,
   data.py and config.py.
3) BASE and candidate runs use the same production mechanics: tranches,
   stagger, refresh, transaction costs, risk parity, cash and catastrophe stop.
4) FIT consists only of realized trades with signal_date AND exit_date before
   2022-01-01. OOS starts at 2022-01-01.
5) Alpha/risk features are signal-date only. No future/path variables are fed
   into allocation decisions.
6) Alpha model is fixed Ridge(alpha=10) on a clipped return target [-25%,25%].
7) Tail-risk models are fixed LogisticRegression(C=0.25, balanced) and are used
   only to define the LOW-RISK conditional branch and diagnostic quadrants.
8) Alpha and risk percentiles are market-specific CDFs built from FIT only.
9) Allocation thresholds are fixed BEFORE OOS: alpha <= 25th percentile and,
   for the conditional family, risk <= 50th percentile.
10) Only BUY order amounts are scaled inside a research wrapper. Sells, stops,
    ranking, tranche timing, risk parity, cash mechanics and production state
    are unchanged.
11) Candidate multipliers are fixed and small in number; this is a hypothesis
    test, not an optimizer.

Pre-declared policies
---------------------
BASE
ALPHA_GENTLE    bottom-alpha => 0.85
ALPHA_BALANCED  bottom-alpha => 0.70
ALPHA_STRONG    bottom-alpha => 0.50
COND_GENTLE     bottom-alpha AND low-risk => 0.85
COND_BALANCED   bottom-alpha AND low-risk => 0.70
COND_STRONG     bottom-alpha AND low-risk => 0.50

Outputs
-------
- gmq_alpha_quality_conditional_allocation_v1_raporu.md
- gmq_alpha_quality_conditional_allocation_v1_metrics.csv
- gmq_alpha_quality_conditional_allocation_v1_monthly.csv
- gmq_alpha_quality_conditional_allocation_v1_trades.csv
- gmq_alpha_quality_conditional_allocation_v1_predictions.csv
- gmq_alpha_quality_conditional_allocation_v1_model_metrics.csv
- gmq_alpha_quality_conditional_allocation_v1_policy_usage.csv
- gmq_alpha_quality_conditional_allocation_v1_stress.csv
- gmq_alpha_quality_conditional_allocation_v1_alpha_bins.csv
- gmq_alpha_quality_conditional_allocation_v1_risk_bins.csv
- gmq_alpha_quality_conditional_allocation_v1_quadrants.csv
- gmq_alpha_quality_conditional_allocation_v1_market.csv

Run on GitHub Actions or locally from repository root.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Reuse the already validated production-path research helper. This keeps the
# engine wrapper identical to the previous test and avoids a second divergent
# implementation of production backtest mechanics.
from gmq_tail_risk_alpha_conditional_sizing import (
    PRIMARY_COST_RT,
    SPLIT_DATE,
    STRESS_COSTS,
    STRICT_FEATURES,
    add_market_percentiles,
    bootstrap_ci,
    build_feature_dataset,
    fit_models,
    make_bin_table,
    metrics_from_daily,
    normalize_date_series,
    predict_models,
    run_engine_backtest,
)

ROOT = Path(__file__).resolve().parent

POLICIES = {
    "BASE": {
        "family": "BASE",
        "alpha_cut": 0.0,
        "risk_cut": 1.0,
        "multiplier": 1.00,
    },
    "ALPHA_GENTLE": {
        "family": "PURE_ALPHA",
        "alpha_cut": 0.25,
        "risk_cut": 1.0,
        "multiplier": 0.85,
    },
    "ALPHA_BALANCED": {
        "family": "PURE_ALPHA",
        "alpha_cut": 0.25,
        "risk_cut": 1.0,
        "multiplier": 0.70,
    },
    "ALPHA_STRONG": {
        "family": "PURE_ALPHA",
        "alpha_cut": 0.25,
        "risk_cut": 1.0,
        "multiplier": 0.50,
    },
    "COND_GENTLE": {
        "family": "ALPHA_X_LOW_RISK",
        "alpha_cut": 0.25,
        "risk_cut": 0.50,
        "multiplier": 0.85,
    },
    "COND_BALANCED": {
        "family": "ALPHA_X_LOW_RISK",
        "alpha_cut": 0.25,
        "risk_cut": 0.50,
        "multiplier": 0.70,
    },
    "COND_STRONG": {
        "family": "ALPHA_X_LOW_RISK",
        "alpha_cut": 0.25,
        "risk_cut": 0.50,
        "multiplier": 0.50,
    },
}


def policy_multiplier(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    x = df.copy()
    alpha_pct = pd.to_numeric(x["alpha_pctile_mkt"], errors="coerce")
    risk_pct = pd.to_numeric(x["risk_pctile_mkt"], errors="coerce")
    alpha_low = alpha_pct <= float(spec["alpha_cut"])
    if spec["family"] == "ALPHA_X_LOW_RISK":
        condition = alpha_low & (risk_pct <= float(spec["risk_cut"]))
    elif spec["family"] == "PURE_ALPHA":
        condition = alpha_low
    else:
        condition = pd.Series(False, index=x.index)
    x["allocation_condition"] = condition.fillna(False)
    x["sizing_multiplier"] = np.where(condition, float(spec["multiplier"]), 1.0)
    return x


def policy_usage(h: pd.DataFrame, name: str, spec: dict) -> dict:
    cond = h["allocation_condition"].astype(bool)
    mult = h["sizing_multiplier"].astype(float)
    return {
        "policy": name,
        "family": spec["family"],
        "alpha_cut_pctile": spec["alpha_cut"],
        "risk_cut_pctile": spec["risk_cut"],
        "target_multiplier": spec["multiplier"],
        "oos_n": len(h),
        "condition_n": int(cond.sum()),
        "condition_pct": float(cond.mean() * 100.0),
        "mean_multiplier": float(mult.mean()),
        "median_multiplier": float(mult.median()),
        "pct_below_1": float((mult < 0.999999).mean() * 100.0),
    }


def trade_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    rows = []
    for market, g in x.groupby("market", dropna=False):
        neg = g.loc[g["ret_pct"] < 0, "ret_pct"]
        tail15 = g.loc[g["ret_pct"] <= -15, "ret_pct"]
        rows.append(
            {
                "market": str(market),
                "n": len(g),
                "win_rate_pct": float((g["ret_pct"] > 0).mean() * 100.0),
                "mean_ret_pct": float(g["ret_pct"].mean()),
                "median_ret_pct": float(g["ret_pct"].median()),
                "pf": float(g.loc[g.ret_pct > 0, "ret_pct"].sum() / (-neg.sum())) if len(neg) and neg.sum() < 0 else np.nan,
                "tail15_rate_pct": float((g["ret_pct"] <= -15).mean() * 100.0),
                "tail15_loss_share_pct": float((-tail15.sum()) / (-neg.sum()) * 100.0) if len(tail15) and len(neg) and neg.sum() < 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def quadrant_table(h: pd.DataFrame) -> pd.DataFrame:
    x = h.copy()
    x["alpha_side"] = np.where(x["alpha_pctile_mkt"] > 0.50, "HIGH_ALPHA", "LOW_ALPHA")
    x["risk_side"] = np.where(x["risk_pctile_mkt"] > 0.50, "HIGH_RISK", "LOW_RISK")
    x["quadrant"] = x["alpha_side"] + " + " + x["risk_side"]
    rows = []
    for q, g in x.groupby("quadrant", dropna=False):
        neg = g.loc[g["ret_pct"] < 0, "ret_pct"]
        rows.append(
            {
                "quadrant": str(q),
                "n": len(g),
                "share_pct": float(len(g) / len(x) * 100.0),
                "win_rate_pct": float((g["ret_pct"] > 0).mean() * 100.0),
                "mean_ret_pct": float(g["ret_pct"].mean()),
                "median_ret_pct": float(g["ret_pct"].median()),
                "pf": float(g.loc[g.ret_pct > 0, "ret_pct"].sum() / (-neg.sum())) if len(neg) and neg.sum() < 0 else np.nan,
                "tail15_rate_pct": float((g["ret_pct"] <= -15).mean() * 100.0),
                "abs_negative_loss_sum_pct": float(-neg.sum()) if len(neg) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("quadrant").reset_index(drop=True)


def market_condition_table(h: pd.DataFrame) -> pd.DataFrame:
    x = h.copy()
    x["is_cond"] = (x["alpha_pctile_mkt"] <= 0.25) & (x["risk_pctile_mkt"] <= 0.50)
    x["is_alpha_low"] = x["alpha_pctile_mkt"] <= 0.25
    rows = []
    for market, g in x.groupby("market"):
        for label, m in (
            ("ALL", np.ones(len(g), dtype=bool)),
            ("ALPHA_LOW", g["is_alpha_low"].to_numpy()),
            ("ALPHA_LOW_LOW_RISK", g["is_cond"].to_numpy()),
        ):
            z = g.iloc[np.flatnonzero(m)] if len(g) else g.iloc[0:0]
            if z.empty:
                continue
            neg = z.loc[z["ret_pct"] < 0, "ret_pct"]
            rows.append(
                {
                    "market": str(market),
                    "segment": label,
                    "n": len(z),
                    "share_of_market_pct": float(len(z) / len(g) * 100.0),
                    "win_rate_pct": float((z["ret_pct"] > 0).mean() * 100.0),
                    "mean_ret_pct": float(z["ret_pct"].mean()),
                    "pf": float(z.loc[z.ret_pct > 0, "ret_pct"].sum() / (-neg.sum())) if len(neg) and neg.sum() < 0 else np.nan,
                    "tail15_rate_pct": float((z["ret_pct"] <= -15).mean() * 100.0),
                }
            )
    return pd.DataFrame(rows)


def monthly_matrix(runs: Dict[str, Tuple[dict, pd.Series]]) -> pd.DataFrame:
    names = list(runs)
    base = runs["BASE"][1]
    base_m = base.resample("ME").last().pct_change().rename("BASE")
    rows = []
    for name in names:
        if name == "BASE":
            continue
        s = runs[name][1].resample("ME").last().pct_change().rename(name)
        d = pd.concat([base_m, s], axis=1).dropna()
        if d.empty:
            continue
        delta = (d[name] - d["BASE"]) * 100.0
        mean_d, ci5, ci95 = bootstrap_ci(delta, n_boot=5000, seed=20261008)
        rng = np.random.default_rng(20261009)
        if len(delta) >= 10:
            draws = rng.choice(delta.to_numpy(), size=(2000, len(delta)), replace=True).mean(axis=1)
            p_le0 = float((draws <= 0).mean())
        else:
            p_le0 = np.nan
        rows.append(
            {
                "policy": name,
                "common_months": len(delta),
                "mean_monthly_delta_pp": mean_d,
                "bootstrap_ci5_pp": ci5,
                "bootstrap_ci95_pp": ci95,
                "bootstrap_p_delta_le0": p_le0,
                "positive_delta_month_pct": float((delta > 0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def make_synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 6000
    d = pd.date_range("2014-01-01", periods=n, freq="B")
    market = np.where(np.arange(n) % 2 == 0, "bist", "us")
    score = rng.normal(0, 1, n)
    risk = rng.normal(0, 1, n)
    # Only a smoke-test mechanism: low alpha + low risk is deliberately weaker.
    ret = 2.5 + 4.0 * score + rng.normal(0, 7, n)
    ret -= ((score < -0.65) & (risk < 0.0)) * 4.0
    out = pd.DataFrame({
        "market": market,
        "ticker": [f"S{i%100:03d}" for i in range(n)],
        "signal_date": d,
        "entry_date": d + pd.Timedelta(days=1),
        "exit_date": d + pd.Timedelta(days=22),
        "ret_pct": ret,
    })
    for c in STRICT_FEATURES:
        out[c] = rng.normal(0, 1, n)
    out["score_pctile"] = 50 + 20 * score
    out["ret_5d_pct"] = score * 5 + rng.normal(0, 2, n)
    out["ret_21d_pct"] = score * 8 + rng.normal(0, 4, n)
    out["ret_63d_pct"] = score * 12 + rng.normal(0, 6, n)
    out["vol20_ann_pct"] = 20 + 4 * risk + rng.normal(0, 2, n)
    out["atr14_pct"] = 3 + np.abs(risk)
    out["market_breadth_5d"] = 0.5 + rng.normal(0, 0.1, n)
    out["market_breadth_21d"] = 0.5 + rng.normal(0, 0.1, n)
    out["market_median_1d_pct"] = rng.normal(0, 1, n)
    out["market_median_21d_pct"] = rng.normal(0, 3, n)
    out["market_median_63d_pct"] = rng.normal(0, 7, n)
    out["market_dispersion_1d_pct"] = np.abs(rng.normal(0, 2, n))
    out["stock_vs_market_21d_pp"] = rng.normal(0, 5, n)
    out["momentum_extension_z"] = score / (np.abs(risk) + 1.0)
    return out


def report(
    out_dir: Path,
    ds_stats: dict,
    model_metrics: pd.DataFrame,
    alpha_bins: pd.DataFrame,
    risk_bins: pd.DataFrame,
    quadrants: pd.DataFrame,
    market_table: pd.DataFrame,
    policy_usage_df: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    monthly_df: pd.DataFrame,
    stress_df: pd.DataFrame,
) -> None:
    lines = [
        "# Global Momentum Quant — Alpha-Quality Conditional Allocation V1",
        "",
        f"_Research only · primary RT cost {PRIMARY_COST_RT:.2f}% · FIT/OOS cutoff {SPLIT_DATE.date()}_",
        "",
        "## 1. Hipotez",
        "Önceki testlerde tail-risk tek başına allocation azaltma için yeterli olmadı. Bu test tail-risk'i cezalandırmak yerine alpha kalitesini hedefler.",
        "İki sabit aile test edilir: (A) yalnızca bottom-alpha quartile azaltma; (B) yalnızca bottom-alpha + low-risk quadrant azaltma.",
        "",
        "## 2. Veri / leakage",
        f"- Toplam trade: **{ds_stats.get('all_n', 0):,}**",
        f"- FIT: **{ds_stats.get('train_n', 0):,}**",
        f"- OOS: **{ds_stats.get('holdout_n', 0):,}**",
        "- FIT trade'lerinde signal_date ve exit_date cutoff öncesindedir.",
        "- Allocation feature'ları yalnızca signal-date strict feature setinden gelir.",
        "- Production config/engine/portfolio/signals/state/order dosyaları değiştirilmez.",
        "",
        "## 3. Model",
        "- Alpha: Ridge(alpha=10), clipped return target [-25%, +25%].",
        "- Tail diagnostics: LogisticRegression(C=0.25, class_weight=balanced) for -5/-10/-15% tails.",
        "- Market percentile CDF'leri yalnızca FIT dağılımından hesaplanır.",
        "- OOS eşikleri sabittir: alpha <= 25th percentile; conditional family için risk <= 50th percentile.",
        "",
        "## 4. Model metrics",
        model_metrics.to_markdown(index=False, floatfmt=".4f") if not model_metrics.empty else "—",
        "",
        "## 5. OOS alpha bins",
        alpha_bins.to_markdown(index=False, floatfmt=".4f") if not alpha_bins.empty else "—",
        "",
        "## 6. OOS risk bins",
        risk_bins.to_markdown(index=False, floatfmt=".4f") if not risk_bins.empty else "—",
        "",
        "## 7. OOS alpha × risk quadrants",
        quadrants.to_markdown(index=False, floatfmt=".4f") if not quadrants.empty else "—",
        "",
        "## 8. OOS market segments",
        market_table.to_markdown(index=False, floatfmt=".4f") if not market_table.empty else "—",
        "",
        "## 9. Policy usage",
        policy_usage_df.to_markdown(index=False, floatfmt=".4f") if not policy_usage_df.empty else "—",
        "",
        "## 10. OOS portfolio results",
        policy_metrics.to_markdown(index=False, floatfmt=".4f") if not policy_metrics.empty else "—",
        "",
        "## 11. Monthly paired bootstrap",
        monthly_df.to_markdown(index=False, floatfmt=".4f") if not monthly_df.empty else "—",
        "",
        "## 12. Cost stress",
        stress_df.to_markdown(index=False, floatfmt=".4f") if not stress_df.empty else "—",
        "",
        "## 13. Decision framework",
        "PRIMARY karar ölçütü sadece MaxDD değildir. Adayın OOS CAGR kaybı, PF, worst-12M, paired monthly bootstrap ve maliyet stresindeki yönü birlikte değerlendirilmelidir.",
        "Özellikle ALPHA_X_LOW_RISK familyasının gerçekten weak quadrantı hedefleyip hedeflemediği ve bunun portföy CAGR'ını BASE'e göre bozup bozmadığı kontrol edilmelidir.",
        "Bu raporun hiçbir sonucu canlı production dosyalarına otomatik olarak uygulanmaz.",
        "",
        "**FINAL STATUS: RESEARCH ONLY — NO PRODUCTION CHANGE**",
    ]
    (out_dir / "gmq_alpha_quality_conditional_allocation_v1_raporu.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=14)
    ap.add_argument("--out", default="research/results_alpha_quality_conditional_allocation_v1")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--cost", type=float, default=PRIMARY_COST_RT)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        ds = make_synthetic()
        ds["signal_date"] = normalize_date_series(ds["signal_date"])
        ds["entry_date"] = normalize_date_series(ds["entry_date"])
        ds["exit_date"] = normalize_date_series(ds["exit_date"])
        fit = ds[ds.signal_date < SPLIT_DATE].copy()
        hold = ds[ds.signal_date >= SPLIT_DATE].copy()
        models, alpha_model, model_metrics = fit_models(fit)
        pred = add_market_percentiles(predict_models(fit, models, alpha_model), predict_models(hold, models, alpha_model))
        hp = pred[pred["sample"] == "holdout"].copy()
        alpha_bins = make_bin_table(hp, "pred_alpha_pct")
        risk_bins = make_bin_table(hp, "tail_risk_score")
        quadrants = quadrant_table(hp)
        market_df = market_condition_table(hp)
        usages = []
        for name, spec in POLICIES.items():
            z = policy_multiplier(hp, spec)
            usages.append(policy_usage(z, name, spec))
        usage_df = pd.DataFrame(usages)
        # Synthetic smoke has no real engine path, so no fake portfolio metrics.
        metrics = pd.DataFrame([{"policy": "SYNTHETIC_SMOKE", "status": "PASS"}])
        monthly = pd.DataFrame()
        stress = pd.DataFrame()
        hp.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_predictions.csv", index=False)
        model_metrics.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_model_metrics.csv", index=False)
        alpha_bins.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_alpha_bins.csv", index=False)
        risk_bins.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_risk_bins.csv", index=False)
        quadrants.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_quadrants.csv", index=False)
        market_df.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_market.csv", index=False)
        usage_df.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_policy_usage.csv", index=False)
        metrics.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_metrics.csv", index=False)
        monthly.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_monthly.csv", index=False)
        stress.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_stress.csv", index=False)
        report(out_dir, {"all_n": len(ds), "train_n": len(fit), "holdout_n": len(hold)}, model_metrics, alpha_bins, risk_bins, quadrants, market_df, usage_df, metrics, monthly, stress)
        print("SMOKE OK — synthetic only")
        return

    import config as C
    import data as DA
    import engine as E
    import portfolio as P

    years = max(int(args.years), 3)
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

    # One BASE history creates the realized trade set. No candidate result is
    # used to fit the model.
    base_state, _ = run_engine_backtest(
        md_b, md_u, fx, start_full, None, PRIMARY_COST_RT, None, C.CAPITAL_TL
    )["BASE"]
    ds = build_feature_dataset(base_state, {"bist": md_b, "us": md_u}, W)
    ds["is_fit"] = (ds["signal_date"] < SPLIT_DATE) & (ds["exit_date"] < SPLIT_DATE)
    ds["is_oos"] = ds["signal_date"] >= SPLIT_DATE
    train = ds[ds["is_fit"]].copy()
    hold = ds[ds["is_oos"]].copy()
    if len(train) < 500 or len(hold) < 500:
        raise RuntimeError(f"Yetersiz FIT/OOS trade: fit={len(train)} oos={len(hold)}")

    risk_models, alpha_model, model_metrics = fit_models(train)
    train_pred = predict_models(train, risk_models, alpha_model)
    hold_pred = predict_models(hold, risk_models, alpha_model)
    pred = add_market_percentiles(train_pred, hold_pred)
    hold_p = pred[pred["sample"] == "holdout"].copy()

    alpha_bins = make_bin_table(hold_p, "pred_alpha_pct")
    risk_bins = make_bin_table(hold_p, "tail_risk_score")
    quadrants = quadrant_table(hold_p)
    market_df = market_condition_table(hold_p)

    lookups: Dict[str, Dict[Tuple[str, str, str], float]] = {}
    usage_rows = []
    policy_frames = []
    for name, spec in POLICIES.items():
        z = policy_multiplier(hold_p, spec)
        usage_rows.append(policy_usage(z, name, spec))
        x = z.copy()
        x["policy"] = name
        x["family"] = spec["family"]
        policy_frames.append(x)
        key_lookup: Dict[Tuple[str, str, str], float] = {}
        for r in z[["market", "ticker", "signal_date", "sizing_multiplier"]].to_dict("records"):
            key = (str(r["market"]), str(r["ticker"]), pd.Timestamp(r["signal_date"]).strftime("%Y-%m-%d"))
            key_lookup[key] = float(r["sizing_multiplier"])
        lookups[name] = key_lookup

    usage_df = pd.DataFrame(usage_rows)
    policy_trades = pd.concat(policy_frames, ignore_index=True)
    policy_trades.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_trades.csv", index=False)
    pred.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_predictions.csv", index=False)
    model_metrics.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_model_metrics.csv", index=False)
    alpha_bins.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_alpha_bins.csv", index=False)
    risk_bins.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_risk_bins.csv", index=False)
    quadrants.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_quadrants.csv", index=False)
    market_df.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_market.csv", index=False)
    usage_df.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_policy_usage.csv", index=False)

    hold_end = max(md_b.dates[-1], md_u.dates[-1])
    runs = run_engine_backtest(
        md_b,
        md_u,
        fx,
        SPLIT_DATE,
        hold_end,
        PRIMARY_COST_RT,
        lookups,
        C.CAPITAL_TL,
    )

    rows = []
    base_daily = runs["BASE"][1]
    base_metrics = metrics_from_daily(base_daily)
    for name, (_state, daily) in runs.items():
        m = metrics_from_daily(daily)
        row = {"policy": name, **m}
        row["cagr_delta_pp"] = m["cagr_pct"] - base_metrics["cagr_pct"]
        row["maxdd_delta_pp"] = m["maxdd_pct"] - base_metrics["maxdd_pct"]
        md = pd.concat(
            [
                base_daily.resample("ME").last().pct_change().rename("base"),
                daily.resample("ME").last().pct_change().rename("cand"),
            ],
            axis=1,
        ).dropna()
        if name == "BASE":
            row.update(
                {
                    "bootstrap_monthly_delta_pp": 0.0,
                    "bootstrap_ci5_pp": 0.0,
                    "bootstrap_ci95_pp": 0.0,
                    "bootstrap_p_delta_le0": 0.0,
                }
            )
        elif not md.empty:
            delta = (md["cand"] - md["base"]) * 100.0
            mean_d, ci5, ci95 = bootstrap_ci(delta, n_boot=5000, seed=20261010)
            rng = np.random.default_rng(20261011)
            draws = rng.choice(delta.to_numpy(), size=(2000, len(delta)), replace=True).mean(axis=1) if len(delta) >= 10 else np.array([np.nan])
            row.update(
                {
                    "bootstrap_monthly_delta_pp": mean_d,
                    "bootstrap_ci5_pp": ci5,
                    "bootstrap_ci95_pp": ci95,
                    "bootstrap_p_delta_le0": float(np.nanmean(draws <= 0)),
                }
            )
        else:
            row.update(
                {
                    "bootstrap_monthly_delta_pp": np.nan,
                    "bootstrap_ci5_pp": np.nan,
                    "bootstrap_ci95_pp": np.nan,
                    "bootstrap_p_delta_le0": np.nan,
                }
            )
        rows.append(row)
    policy_metrics = pd.DataFrame(rows)

    monthly_df = monthly_matrix(runs)

    # Fixed cost stress: no re-fit and no threshold changes.
    stress_rows = []
    for cost in STRESS_COSTS:
        cost_runs = run_engine_backtest(
            md_b, md_u, fx, SPLIT_DATE, hold_end, cost, lookups, C.CAPITAL_TL
        )
        b = metrics_from_daily(cost_runs["BASE"][1])
        for name, (_s, d) in cost_runs.items():
            cm = metrics_from_daily(d)
            stress_rows.append(
                {
                    "cost_rt_pct": cost,
                    "policy": name,
                    "cagr_pct": cm["cagr_pct"],
                    "cagr_delta_vs_base_pp": cm["cagr_pct"] - b["cagr_pct"],
                    "maxdd_pct": cm["maxdd_pct"],
                    "pf": cm["profit_factor"],
                    "worst_12m_pct": cm["worst_12m_pct"],
                }
            )
    stress_df = pd.DataFrame(stress_rows)

    policy_metrics.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_metrics.csv", index=False)
    monthly_df.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_monthly.csv", index=False)
    stress_df.to_csv(out_dir / "gmq_alpha_quality_conditional_allocation_v1_stress.csv", index=False)

    report(
        out_dir,
        {"all_n": len(ds), "train_n": len(train), "holdout_n": len(hold)},
        model_metrics,
        alpha_bins,
        risk_bins,
        quadrants,
        market_df,
        usage_df,
        policy_metrics,
        monthly_df,
        stress_df,
    )

    print("=== GMQ Alpha-Quality Conditional Allocation V1 ===")
    print(f"all={len(ds):,} fit={len(train):,} oos={len(hold):,}")
    print(policy_metrics[["policy", "cagr_pct", "maxdd_pct", "profit_factor", "cagr_delta_pp", "maxdd_delta_pp"]].to_string(index=False))
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
