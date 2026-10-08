#!/usr/bin/env python3
"""
Global Momentum Quant — Asymmetric Tail-Risk Allocation V1

RESEARCH ONLY.

Hypothesis
----------
The previous tail-risk sizing tests applied a broad haircut and lost CAGR.
The current hypothesis is narrower:

    "High tail-risk is not automatically bad when alpha is also high.
     The economically dangerous state is specifically LOW alpha + HIGH tail risk."

Therefore this experiment does NOT reduce every risky trade. It scales BUY
amounts only inside one pre-declared mismatch quadrant:

    alpha_pctile < 0.40 AND risk_pctile >= 0.60

All other combinations remain at 100% allocation.

Pre-declared policies
---------------------
BASE                    danger multiplier = 1.00
ASYM_GENTLE              danger multiplier = 0.85
ASYM_BALANCED            danger multiplier = 0.70
ASYM_STRONG              danger multiplier = 0.50

No optimizer is used. Thresholds and multipliers are fixed before OOS.

Critical integrity rules
------------------------
1) Production config.py / engine.py / portfolio.py / signals.py are not edited.
2) This file reuses the validated production-path helpers from the previous
   Tail-Risk × Alpha Conditional Sizing V1 research module.
3) Model fit is strictly pre-2022-01-01 and requires signal_date AND exit_date
   to be before the cutoff.
4) OOS begins at 2022-01-01. No OOS tuning is performed.
5) Model features are signal-date strict features only.
6) Alpha/risk percentiles are market-specific and reference TRAIN only.
7) Only BUY order amounts are scaled in the research wrapper. Sells, stops,
   ranking, tranche schedule, risk parity, cash mechanics and catastrophe
   stops are untouched.
8) Cost stress reuses the same fixed policies; no re-optimization per cost.

Dependencies inside the repository
-----------------------------------
- gmq_tail_risk_alpha_conditional_sizing.py
- gmq_early_adverse_predictor.py
- production: data.py, engine.py, portfolio.py, config.py

Outputs
-------
- gmq_asymmetric_tail_risk_allocation_raporu.md
- gmq_asymmetric_tail_risk_allocation_metrics.csv
- gmq_asymmetric_tail_risk_allocation_stress.csv
- gmq_asymmetric_tail_risk_allocation_monthly.csv
- gmq_asymmetric_tail_risk_allocation_trades.csv
- gmq_asymmetric_tail_risk_allocation_predictions.csv
- gmq_asymmetric_tail_risk_allocation_model_metrics.csv
- gmq_asymmetric_tail_risk_allocation_policy_usage.csv
- gmq_asymmetric_tail_risk_allocation_quadrants.csv
- gmq_asymmetric_tail_risk_allocation_alpha_risk_matrix.csv
- gmq_asymmetric_tail_risk_allocation_market.csv

Examples
--------
python gmq_asymmetric_tail_risk_allocation_v1.py --years 14
python gmq_asymmetric_tail_risk_allocation_v1.py --synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# The validated production-path helper is imported only when the real
# repository backtest is requested. The synthetic smoke test stays standalone.
BASE_RESEARCH = None

def load_base_research():
    global BASE_RESEARCH
    if BASE_RESEARCH is not None:
        return BASE_RESEARCH
    try:
        import gmq_tail_risk_alpha_conditional_sizing as mod
    except Exception as exc:  # pragma: no cover - clear runtime guidance
        raise RuntimeError(
            "Eksik bağımlılık: gmq_tail_risk_alpha_conditional_sizing.py repo kökünde olmalı."
        ) from exc
    BASE_RESEARCH = mod
    return BASE_RESEARCH

ROOT = Path(__file__).resolve().parent
SPLIT_DATE = pd.Timestamp("2022-01-01")
PRIMARY_COST_RT = 0.35
STRESS_COSTS = (0.35, 0.50, 0.75, 1.00, 1.25)

# Fixed before OOS. The asymmetric intervention targets only the lower-alpha /
# higher-risk mismatch. No parameter search is performed.
ALPHA_LOW_CUTOFF = 0.40
RISK_HIGH_CUTOFF = 0.60
POLICIES = {
    "BASE": {"danger_multiplier": 1.00},
    "ASYM_GENTLE": {"danger_multiplier": 0.85},
    "ASYM_BALANCED": {"danger_multiplier": 0.70},
    "ASYM_STRONG": {"danger_multiplier": 0.50},
}


def classify_quadrant(alpha_pct: float, risk_pct: float) -> str:
    if not np.isfinite(alpha_pct) or not np.isfinite(risk_pct):
        return "UNKNOWN"
    alpha_high = alpha_pct >= ALPHA_LOW_CUTOFF
    risk_high = risk_pct >= RISK_HIGH_CUTOFF
    if (not alpha_high) and risk_high:
        return "LOW_ALPHA_HIGH_RISK"
    if alpha_high and risk_high:
        return "HIGH_ALPHA_HIGH_RISK"
    if alpha_high and (not risk_high):
        return "HIGH_ALPHA_LOW_RISK"
    return "LOW_ALPHA_LOW_RISK"


def add_asymmetric_features(df: pd.DataFrame, danger_multiplier: float) -> pd.DataFrame:
    x = df.copy()
    x["asym_quadrant"] = [
        classify_quadrant(a, r)
        for a, r in zip(x["alpha_pctile_mkt"], x["risk_pctile_mkt"])
    ]
    danger = x["asym_quadrant"].eq("LOW_ALPHA_HIGH_RISK")
    x["danger_quadrant"] = danger
    x["sizing_multiplier"] = np.where(danger, float(danger_multiplier), 1.0)
    return x


def make_lookup(pred: pd.DataFrame) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for r in pred.to_dict("records"):
        if str(r.get("sample")) != "holdout":
            continue
        date = pd.Timestamp(r["signal_date"]).strftime("%Y-%m-%d")
        key = (str(r["market"]), str(r["ticker"]), date)
        out[key] = float(r["sizing_multiplier"])
    return out


def safe_pct(num: float, den: float) -> float:
    return float(num / den * 100.0) if den else 0.0


def trade_metrics(df: pd.DataFrame) -> Dict[str, float]:
    x = df.copy()
    x["ret_pct"] = pd.to_numeric(x["ret_pct"], errors="coerce")
    x = x.dropna(subset=["ret_pct"])
    if x.empty:
        return {"n": 0}
    neg = x.loc[x["ret_pct"] < 0, "ret_pct"]
    pos = x.loc[x["ret_pct"] > 0, "ret_pct"]
    tail15 = x.loc[x["ret_pct"] <= -15.0, "ret_pct"]
    return {
        "n": int(len(x)),
        "win_pct": float((x["ret_pct"] > 0).mean() * 100.0),
        "mean_ret_pct": float(x["ret_pct"].mean()),
        "median_ret_pct": float(x["ret_pct"].median()),
        "mean_loss_pct": float(neg.mean()) if len(neg) else 0.0,
        "profit_factor": float(pos.sum() / (-neg.sum())) if len(neg) and neg.sum() < 0 else np.nan,
        "tail15_rate_pct": float((x["ret_pct"] <= -15.0).mean() * 100.0),
        "abs_negative_loss_pct": float(-neg.sum()) if len(neg) else 0.0,
        "tail15_loss_share_pct": safe_pct(float(-tail15.sum()), float(-neg.sum())) if len(tail15) and len(neg) else 0.0,
        "positive_return_sum_pct": float(pos.sum()) if len(pos) else 0.0,
    }


def quadrant_table(hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_neg = float(-hold.loc[hold["ret_pct"] < 0, "ret_pct"].sum())
    total_pos = float(hold.loc[hold["ret_pct"] > 0, "ret_pct"].sum())
    for q, g in hold.groupby("asym_quadrant", dropna=False, observed=False):
        tm = trade_metrics(g)
        neg_loss = tm.get("abs_negative_loss_pct", 0.0)
        pos_sum = tm.get("positive_return_sum_pct", 0.0)
        rows.append({
            "quadrant": str(q),
            **tm,
            "share_of_oos_pct": safe_pct(len(g), len(hold)),
            "share_of_negative_loss_pct": safe_pct(neg_loss, total_neg),
            "share_of_positive_return_pct": safe_pct(pos_sum, total_pos),
        })
    return pd.DataFrame(rows).sort_values("quadrant").reset_index(drop=True)


def alpha_risk_matrix(hold: pd.DataFrame) -> pd.DataFrame:
    x = hold.copy()
    x["alpha_bin"] = pd.qcut(x["pred_alpha_pct"], q=5, labels=False, duplicates="drop")
    x["risk_bin"] = pd.qcut(x["tail_risk_score"], q=5, labels=False, duplicates="drop")
    rows = []
    for (ab, rb), g in x.groupby(["alpha_bin", "risk_bin"], observed=True):
        tm = trade_metrics(g)
        rows.append({
            "alpha_bin": int(ab),
            "risk_bin": int(rb),
            **tm,
        })
    return pd.DataFrame(rows).sort_values(["alpha_bin", "risk_bin"]).reset_index(drop=True)


def market_table(hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, g in hold.groupby("market"):
        tm = trade_metrics(g)
        danger = g[g["danger_quadrant"]]
        dm = trade_metrics(danger)
        rows.append({
            "market": str(market),
            **{f"all_{k}": v for k, v in tm.items()},
            "danger_n": dm.get("n", 0),
            "danger_share_pct": safe_pct(len(danger), len(g)),
            "danger_mean_ret_pct": dm.get("mean_ret_pct", np.nan),
            "danger_win_pct": dm.get("win_pct", np.nan),
            "danger_tail15_rate_pct": dm.get("tail15_rate_pct", np.nan),
            "danger_abs_negative_loss_pct": dm.get("abs_negative_loss_pct", 0.0),
        })
    return pd.DataFrame(rows)


def policy_usage(hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, spec in POLICIES.items():
        x = add_asymmetric_features(hold, spec["danger_multiplier"])
        m = x["sizing_multiplier"]
        rows.append({
            "policy": name,
            "danger_multiplier": spec["danger_multiplier"],
            "oos_n": len(x),
            "danger_n": int(x["danger_quadrant"].sum()),
            "danger_share_pct": float(x["danger_quadrant"].mean() * 100.0),
            "mean_multiplier": float(m.mean()),
            "median_multiplier": float(m.median()),
            "pct_below_1": float((m < 0.999999).mean() * 100.0),
            "pct_at_danger_multiplier": float((m == spec["danger_multiplier"]).mean() * 100.0),
        })
    return pd.DataFrame(rows)


def metrics_from_daily(daily: pd.Series) -> Dict[str, float]:
    return load_base_research().metrics_from_daily(daily)


def bootstrap_summary(base_daily: pd.Series, cand_daily: pd.Series) -> Dict[str, float]:
    br = load_base_research()
    md = br.monthly_delta(base_daily, cand_daily)
    if md.empty:
        return {
            "common_months": 0,
            "mean_monthly_delta_pp": np.nan,
            "median_monthly_delta_pp": np.nan,
            "ci5_pp": np.nan,
            "ci95_pp": np.nan,
            "p_delta_le0": np.nan,
            "positive_month_pct": np.nan,
        }
    mean_delta, ci5, ci95 = br.bootstrap_ci(md["delta_pp"], seed=20261007)
    arr = md["delta_pp"].to_numpy(dtype=float)
    rng = np.random.default_rng(20261008)
    if len(arr) >= 10:
        draws = rng.choice(arr, size=(5000, len(arr)), replace=True).mean(axis=1)
        p = float((draws <= 0).mean())
    else:
        p = np.nan
    return {
        "common_months": int(len(md)),
        "mean_monthly_delta_pp": float(mean_delta),
        "median_monthly_delta_pp": float(np.median(arr)),
        "ci5_pp": float(ci5),
        "ci95_pp": float(ci95),
        "p_delta_le0": p,
        "positive_month_pct": float((arr > 0).mean() * 100.0),
    }


def run_research(years: int, start: pd.Timestamp, out_dir: Path) -> None:
    br = load_base_research()
    import config as C
    import data as DA
    import engine as E
    import portfolio as P

    panels = {
        "bist": DA.get_panel("bist", years=years, force_full=True),
        "us": DA.get_panel("us", years=years, force_full=True),
    }
    if panels["bist"].empty or panels["us"].empty:
        raise RuntimeError("BIST veya US paneli boş.")
    fx = DA.get_fx(years)
    if fx.empty:
        raise RuntimeError("USD/TRY verisi boş.")

    W = {mk: DA.to_wide(panel) for mk, panel in panels.items()}
    md_b = E.MarketData("bist", W["bist"], spec=C.MARKETS["bist"]["spec"])
    md_u = E.MarketData("us", W["us"], spec=C.MARKETS["us"]["spec"])

    warm = max(260, P.C.RP_WINDOW + 5)
    start_full = max(pd.Timestamp(md_b.dates[warm]), pd.Timestamp(md_u.dates[warm]))

    # Step 1 — one untouched BASE production-path run to build realized trades.
    base_full_state, _base_full_daily = br.run_engine_backtest(
        md_b, md_u, fx, start_full, None, PRIMARY_COST_RT, None, C.CAPITAL_TL
    )["BASE"]

    ds = br.build_feature_dataset(
        base_full_state,
        {"bist": md_b, "us": md_u},
        W,
    )
    ds["is_fit"] = (ds["signal_date"] < SPLIT_DATE) & (ds["exit_date"] < SPLIT_DATE)
    ds["is_oos"] = ds["signal_date"] >= SPLIT_DATE
    train = ds[ds["is_fit"]].copy()
    hold = ds[ds["is_oos"] & (ds["signal_date"] >= start)].copy()
    if len(train) < 500 or len(hold) < 500:
        raise RuntimeError(f"Yetersiz FIT/OOS trade: fit={len(train)} oos={len(hold)}")

    # Step 2 — exactly the previously validated fixed model stack.
    risk_models, alpha_model, model_metrics = br.fit_models(train)
    train_pred = br.predict_models(train, risk_models, alpha_model)
    hold_pred = br.predict_models(hold, risk_models, alpha_model)
    pred = br.add_market_percentiles(train_pred, hold_pred)
    pred.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_predictions.csv", index=False)
    model_metrics.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_model_metrics.csv", index=False)

    hold = pred[pred["sample"] == "holdout"].copy()
    hold = add_asymmetric_features(hold, POLICIES["ASYM_GENTLE"]["danger_multiplier"])

    # Diagnostics before the portfolio replay.
    qtab = quadrant_table(hold)
    matrix = alpha_risk_matrix(hold)
    mkt = market_table(hold)
    usage = policy_usage(hold)
    qtab.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_quadrants.csv", index=False)
    matrix.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_alpha_risk_matrix.csv", index=False)
    mkt.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_market.csv", index=False)
    usage.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_policy_usage.csv", index=False)

    # Save the policy trade table once. The realized ret_pct is the same across
    # policies; the sizing multiplier is the only research treatment.
    policy_trade_rows = []
    for name, spec in POLICIES.items():
        x = add_asymmetric_features(hold, spec["danger_multiplier"])
        x["policy"] = name
        policy_trade_rows.append(x)
    policy_trades = pd.concat(policy_trade_rows, ignore_index=True)
    policy_trades.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_trades.csv", index=False)

    lookups = {}
    for name, spec in POLICIES.items():
        lookups[name] = make_lookup(add_asymmetric_features(hold, spec["danger_multiplier"]))

    # Step 3 — exact production-path OOS replay.
    oos_end = max(md_b.dates[-1], md_u.dates[-1])
    runs = br.run_engine_backtest(
        md_b,
        md_u,
        fx,
        start,
        oos_end,
        PRIMARY_COST_RT,
        lookups,
        C.CAPITAL_TL,
    )

    base_daily = runs["BASE"][1]
    base_metrics = metrics_from_daily(base_daily)
    metric_rows = []
    monthly_rows = []

    for name, (state, daily) in runs.items():
        m = metrics_from_daily(daily)
        row = {"policy": name, **m}
        if name == "BASE":
            row.update({
                "cagr_delta_pp": 0.0,
                "maxdd_delta_pp": 0.0,
                "bootstrap_monthly_delta_pp": 0.0,
                "bootstrap_ci5_pp": 0.0,
                "bootstrap_ci95_pp": 0.0,
                "bootstrap_p_delta_le0": 0.0,
                "bootstrap_positive_month_pct": 0.0,
            })
        else:
            bs = bootstrap_summary(base_daily, daily)
            row.update({
                "cagr_delta_pp": m["cagr_pct"] - base_metrics["cagr_pct"],
                "maxdd_delta_pp": m["maxdd_pct"] - base_metrics["maxdd_pct"],
                "bootstrap_monthly_delta_pp": bs["mean_monthly_delta_pp"],
                "bootstrap_ci5_pp": bs["ci5_pp"],
                "bootstrap_ci95_pp": bs["ci95_pp"],
                "bootstrap_p_delta_le0": bs["p_delta_le0"],
                "bootstrap_positive_month_pct": bs["positive_month_pct"],
            })
            md = BASE_RESEARCH.monthly_delta(base_daily, daily)
            md = md.reset_index().rename(columns={"index": "month"})
            md["policy"] = name
            monthly_rows.append(md)

        # Diagnostics from the realized trade log are secondary; the primary
        # acceptance decision remains portfolio-level OOS performance.
        all_trades = []
        for mk in ("bist", "us"):
            all_trades.extend(state.get("markets", {}).get(mk, {}).get("trades", []) or [])
        trdf = pd.DataFrame(all_trades)
        if not trdf.empty and "ret_pct" in trdf.columns:
            neg = pd.to_numeric(trdf["ret_pct"], errors="coerce")
            row["realized_trade_count"] = int(len(neg))
            row["realized_trade_win_pct"] = float((neg > 0).mean() * 100.0)
            row["realized_abs_negative_loss_sum_pct"] = float(-neg[neg < 0].sum())
            tail = neg[neg <= -15.0]
            row["realized_tail15_loss_share_pct"] = safe_pct(float(-tail.sum()), float(-neg[neg < 0].sum())) if len(tail) else 0.0
        else:
            row["realized_trade_count"] = 0
            row["realized_trade_win_pct"] = np.nan
            row["realized_abs_negative_loss_sum_pct"] = np.nan
            row["realized_tail15_loss_share_pct"] = np.nan

        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_metrics.csv", index=False)
    monthly = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    monthly.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_monthly.csv", index=False)

    # Step 4 — fixed-policy transaction-cost stress.
    stress_rows = []
    for cost in STRESS_COSTS:
        cost_runs = br.run_engine_backtest(
            md_b, md_u, fx, start, oos_end, cost, lookups, C.CAPITAL_TL
        )
        stress_base = metrics_from_daily(cost_runs["BASE"][1])
        for name, (_state, daily) in cost_runs.items():
            cm = metrics_from_daily(daily)
            stress_rows.append({
                "cost_rt_pct": cost,
                "policy": name,
                "cagr_pct": cm["cagr_pct"],
                "cagr_delta_vs_base_pp": cm["cagr_pct"] - stress_base["cagr_pct"],
                "maxdd_pct": cm["maxdd_pct"],
                "pf": cm["profit_factor"],
                "worst_12m_pct": cm["worst_12m_pct"],
            })
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_stress.csv", index=False)

    report = build_report(
        years=years,
        dataset=ds,
        train=train,
        hold=hold,
        model_metrics=model_metrics,
        qtab=qtab,
        matrix=matrix,
        mkt=mkt,
        usage=usage,
        metrics=metrics,
        stress=stress,
    )
    (out_dir / "gmq_asymmetric_tail_risk_allocation_raporu.md").write_text(report, encoding="utf-8")

    print("=== GMQ Asymmetric Tail-Risk Allocation V1 ===")
    print(f"all={len(ds):,} fit={len(train):,} oos={len(hold):,}")
    cols = ["policy", "cagr_pct", "maxdd_pct", "profit_factor", "cagr_delta_pp", "maxdd_delta_pp"]
    print(metrics[cols].to_string(index=False))
    print(f"Outputs: {out_dir.resolve()}")


def build_report(
    years: int,
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    hold: pd.DataFrame,
    model_metrics: pd.DataFrame,
    qtab: pd.DataFrame,
    matrix: pd.DataFrame,
    mkt: pd.DataFrame,
    usage: pd.DataFrame,
    metrics: pd.DataFrame,
    stress: pd.DataFrame,
) -> str:
    now = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    danger = hold[hold["danger_quadrant"]]
    danger_tm = trade_metrics(danger)
    hold_tm = trade_metrics(hold)

    # Automatic, conservative verdict. This is only a research flag.
    candidates = metrics[metrics["policy"] != "BASE"].copy()
    stress_grp = stress[stress["policy"] != "BASE"].copy()
    winners = candidates[candidates["cagr_delta_pp"] >= -0.25]
    dd_better = candidates[candidates["maxdd_delta_pp"] > 0]
    stress_positive = (
        stress_grp.groupby("policy")["cagr_delta_vs_base_pp"].apply(lambda s: (s >= -0.25).all())
        if not stress_grp.empty else pd.Series(dtype=bool)
    )
    verdict = "CONDITIONAL PASS — further validation"
    if winners.empty or dd_better.empty:
        verdict = "REJECT — asymmetric sizing did not improve the economic objective enough"
    elif not bool(stress_positive.all() if len(stress_positive) else False):
        verdict = "REJECT — cost-stress robustness is insufficient"

    lines = [
        "# Global Momentum Quant — Asymmetric Tail-Risk Allocation V1",
        "",
        f"_Üretim: {now} · years={years} · primary RT cost={PRIMARY_COST_RT:.2f}%_",
        "",
        "## 1. Araştırma hipotezi",
        "Önceki geniş tail-risk sizing yaklaşımı yüksek riskli ama yüksek alpha taşıyan kazananları da küçültüyordu. Bu test bunun tersine yalnızca **LOW ALPHA + HIGH TAIL RISK** uyumsuzluk bölgesini küçültür.",
        "",
        "### Önceden sabitlenen kural",
        f"- Low-alpha cutoff: **{ALPHA_LOW_CUTOFF:.2f}** market percentile",
        f"- High-risk cutoff: **{RISK_HIGH_CUTOFF:.2f}** market percentile",
        "- Müdahale bölgesi: `alpha_pctile < 0.40 AND risk_pctile >= 0.60`",
        "- Yüksek alpha + yüksek risk: **tam pozisyon korunur**",
        "- Yüksek alpha + düşük risk: **tam pozisyon korunur**",
        "- Düşük alpha + düşük risk: **tam pozisyon korunur**",
        "- Yalnızca düşük alpha + yüksek risk bölgesi küçültülür.",
        "",
        "| Policy | Danger multiplier |",
        "|---|---:|",
        "| BASE | 1.00 |",
        "| ASYM_GENTLE | 0.85 |",
        "| ASYM_BALANCED | 0.70 |",
        "| ASYM_STRONG | 0.50 |",
        "",
        "## 2. Veri / leakage",
        f"- Tüm trade satırı: **{len(dataset):,}**",
        f"- FIT: **{len(train):,}**",
        f"- OOS: **{len(hold):,}**",
        f"- FIT/OOS cutoff: **{SPLIT_DATE.date()}**",
        "- FIT işlemleri: signal_date ve exit_date cutoff öncesi.",
        "- OOS model parametreleri yeniden optimize edilmez.",
        "- Alpha/risk percentile referansı market bazında yalnızca TRAIN dağılımından üretilir.",
        "- Production dosyaları değiştirilmez; BUY scaling yalnızca araştırma wrapper'ındadır.",
        "",
        "## 3. Model doğrulaması",
        model_metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 4. Kritik trade-level kontrol: tehlike bölgesi gerçekten tehlikeli mi?",
        "",
        f"OOS toplamı: **{hold_tm['n']:,} trade**, win={hold_tm['win_pct']:.2f}%, mean={hold_tm['mean_ret_pct']:.2f}%.",
        f"LOW_ALPHA_HIGH_RISK: **{danger_tm.get('n',0):,} trade** ({safe_pct(danger_tm.get('n',0), len(hold)):.2f}%), win={danger_tm.get('win_pct',np.nan):.2f}%, mean={danger_tm.get('mean_ret_pct',np.nan):.2f}%, tail15={danger_tm.get('tail15_rate_pct',np.nan):.2f}%.",
        "",
        "Bu bölge OOS pozitif getirinin de büyük bölümünü taşıyorsa güçlü bir haircut ekonomik olarak yanlış olabilir. Bu yüzden yalnızca loss capture değil opportunity-cost de raporlanır.",
        "",
        qtab.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. 5×5 alpha-risk haritası",
        "",
        matrix.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. Market kırılımı",
        mkt.to_markdown(index=False, floatfmt=".4f") if not mkt.empty else "—",
        "",
        "## 7. Policy kullanım denetimi",
        usage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 8. Primary OOS portföy sonucu",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 9. Aylık bootstrap",
    ]

    for _, r in metrics[metrics["policy"] != "BASE"].iterrows():
        lines.append(
            f"- **{r['policy']}**: aylık delta={r['bootstrap_monthly_delta_pp']:.4f} pp; CI5={r['bootstrap_ci5_pp']:.4f}; CI95={r['bootstrap_ci95_pp']:.4f}; P(delta≤0)={r['bootstrap_p_delta_le0']:.4f}; pozitif ay={r['bootstrap_positive_month_pct']:.2f}%."
        )

    lines += [
        "",
        "## 10. Maliyet stresi",
        stress.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 11. Karar",
        f"**{verdict}**",
        "",
        "Kabul için yalnızca MaxDD düşüşü yeterli değildir. CAGR kaybı, PF, worst-12M, paired monthly bootstrap ve maliyet stresinin birlikte değerlendirilmesi gerekir.",
        "",
        "### Canlıya alma kuralı",
        "- OOS CAGR kaybı belirginse: RED.",
        "- MaxDD iyileşirken CAGR kaybı sınırlıysa: ancak bootstrap + cost stress destekliyorsa devam.",
        "- High-alpha/high-risk bölgesi açıkça kazanç üretiyorsa: broad risk haircut yapılmaz.",
        "- Bu V1 sonucu production koduna otomatik olarak taşınmaz.",
        "",
        "**CANLI DEĞİŞİKLİK YOK — RESEARCH ONLY**",
    ]
    return "\n".join(lines) + "\n"


def run_synthetic(out_dir: Path) -> None:
    # Standalone smoke test: validates the quadrant rule and artifact plumbing
    # without importing any production repository module.
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(42)
    n = 3000
    dates = pd.bdate_range("2016-01-04", periods=n)
    market = np.where(np.arange(n) % 2 == 0, "bist", "us")
    score = rng.normal(55, 18, n)
    vol = np.abs(rng.normal(24, 6, n))
    breadth = np.clip(rng.normal(0.52, 0.16, n), 0, 1)
    f = pd.DataFrame({
        "market": market,
        "ticker": [f"T{i % 80:03d}" for i in range(n)],
        "signal_date": dates,
        "entry_date": dates + pd.Timedelta(days=1),
        "exit_date": dates + pd.Timedelta(days=30),
    })
    f["score_pctile"] = score
    f["ret_5d_pct"] = score / 7 + rng.normal(0, 4, n)
    f["ret_21d_pct"] = score / 3 + rng.normal(0, 6, n)
    f["ret_63d_pct"] = score / 2 + rng.normal(0, 8, n)
    f["vol20_ann_pct"] = vol
    f["atr14_pct"] = vol / 4
    f["market_breadth_5d"] = breadth + rng.normal(0, 0.04, n)
    f["market_breadth_21d"] = breadth
    f["market_median_1d_pct"] = rng.normal(0, 1, n)
    f["market_median_21d_pct"] = rng.normal(0, 4, n)
    f["market_median_63d_pct"] = rng.normal(0, 8, n)
    f["market_dispersion_1d_pct"] = np.abs(rng.normal(2, 0.8, n))
    f["stock_vs_market_21d_pp"] = f["ret_21d_pct"] - rng.normal(0, 5, n)
    f["momentum_extension_z"] = rng.normal(0, 1, n)
    latent_alpha = 0.10 * (score - 50) - 0.15 * (vol - 24) + 8 * (breadth - 0.5)
    tail_push = np.maximum(vol - 28, 0) * 1.1 + np.maximum(0.40 - breadth, 0) * 22
    f["ret_pct"] = latent_alpha - tail_push + rng.normal(0, 7, n)

    feature_cols = [
        "score_pctile", "ret_5d_pct", "ret_21d_pct", "ret_63d_pct",
        "vol20_ann_pct", "atr14_pct", "market_breadth_5d", "market_breadth_21d",
        "market_median_1d_pct", "market_median_21d_pct", "market_median_63d_pct",
        "market_dispersion_1d_pct", "stock_vs_market_21d_pp", "momentum_extension_z",
    ]
    split = pd.Timestamp("2022-01-01")
    tr = f[f["signal_date"] < split].copy()
    ho = f[f["signal_date"] >= split].copy()
    pre = lambda est: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", est)])
    alpha = pre(Ridge(alpha=10.0)).fit(tr[feature_cols], tr["ret_pct"].clip(-25, 25))
    ho["pred_alpha_pct"] = alpha.predict(ho[feature_cols])
    for target, thr in (("tail_5", -5), ("tail_10", -10), ("tail_15", -15)):
        m = pre(LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000, random_state=42))
        m.fit(tr[feature_cols], (tr["ret_pct"] <= thr).astype(int))
        ho[f"p_{target}"] = m.predict_proba(ho[feature_cols])[:, 1]
    ho["tail_risk_score"] = 0.25 * ho["p_tail_5"] + 0.35 * ho["p_tail_10"] + 0.40 * ho["p_tail_15"]

    # Market-specific empirical percentiles from TRAIN only.
    parts = []
    for mk in sorted(ho["market"].unique()):
        a = tr.loc[tr["market"] == mk, "ret_pct"]
        # For smoke only, use the same model outputs on a train reference.
        tr_a = alpha.predict(tr.loc[tr["market"] == mk, feature_cols])
        tr_r = np.zeros(len(tr_a)) + np.nan
        for target, thr in (("tail_5", -5), ("tail_10", -10), ("tail_15", -15)):
            m = pre(LogisticRegression(C=0.25, class_weight="balanced", max_iter=2000, random_state=42))
            m.fit(tr[feature_cols], (tr["ret_pct"] <= thr).astype(int))
            p = m.predict_proba(tr.loc[tr["market"] == mk, feature_cols])[:, 1]
            tr_r = np.nan_to_num(tr_r, nan=0.0) + p * ({"tail_5":0.25,"tail_10":0.35,"tail_15":0.40}[target])
        g = ho[ho["market"] == mk].copy()
        g["alpha_pctile_mkt"] = [(np.sum(tr_a <= v) - 0.5 * np.sum(tr_a == v)) / len(tr_a) for v in g["pred_alpha_pct"]]
        g["risk_pctile_mkt"] = [(np.sum(tr_r <= v) - 0.5 * np.sum(tr_r == v)) / len(tr_r) for v in g["tail_risk_score"]]
        parts.append(g)
    hp = pd.concat(parts, ignore_index=True)
    hp = add_asymmetric_features(hp, 0.85)

    hp.to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_predictions.csv", index=False)
    quadrant_table(hp).to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_quadrants.csv", index=False)
    alpha_risk_matrix(hp).to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_alpha_risk_matrix.csv", index=False)
    policy_usage(hp).to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_policy_usage.csv", index=False)
    market_table(hp).to_csv(out_dir / "gmq_asymmetric_tail_risk_allocation_market.csv", index=False)
    (out_dir / "gmq_asymmetric_tail_risk_allocation_model_metrics.csv").write_text("synthetic_smoke,ok\n", encoding="utf-8")
    (out_dir / "gmq_asymmetric_tail_risk_allocation_metrics.csv").write_text("synthetic_smoke,ok\n", encoding="utf-8")
    (out_dir / "gmq_asymmetric_tail_risk_allocation_stress.csv").write_text("synthetic_smoke,not_a_strategy_backtest\n", encoding="utf-8")
    (out_dir / "gmq_asymmetric_tail_risk_allocation_monthly.csv").write_text("synthetic_smoke,not_a_strategy_backtest\n", encoding="utf-8")
    (out_dir / "gmq_asymmetric_tail_risk_allocation_trades.csv").write_text("synthetic_smoke,not_a_strategy_backtest\n", encoding="utf-8")
    (out_dir / "gmq_asymmetric_tail_risk_allocation_raporu.md").write_text(
        "# GMQ — Asymmetric Tail-Risk Allocation V1 (Synthetic Smoke)\n\nSMOKE PASS. Quadrant classification and artifact generation succeeded. This is not a trading result.\n\nRule: alpha percentile < 0.40 AND risk percentile >= 0.60 => danger quadrant.\n\n**RESEARCH ONLY — NO PRODUCTION CHANGE.**\n",
        encoding="utf-8",
    )
    print("SMOKE OK — synthetic only")

def main() -> None:
    ap = argparse.ArgumentParser(description="GMQ Asymmetric Tail-Risk Allocation V1 research")
    ap.add_argument("--years", type=int, default=14, help="historical years to download")
    ap.add_argument("--start", default="2022-01-01", help="OOS portfolio start")
    ap.add_argument("--out", default="research/results_asymmetric_tail_risk_v1")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start).normalize()

    if args.synthetic:
        run_synthetic(out_dir)
        return

    run_research(max(int(args.years), 3), start, out_dir)


if __name__ == "__main__":
    main()
