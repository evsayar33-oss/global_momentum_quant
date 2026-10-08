#!/usr/bin/env python3
"""
Global Momentum Quant — Momentum Entry Timing / Chase Avoidance V3

RESEARCH ONLY.

V3 objective
------------
V2 showed that the broad `chase_risk >= 0.20` group is economically mixed:
healthy momentum continuation and dangerous late/chase entries are being
combined. V3 keeps the V2 candidate universe, then classifies those SAME
candidate events into:

    HEALTHY_CHASE   = chase event with strong continuation/regime support
    DANGEROUS_CHASE = chase event with weak continuation/regime support

Only DANGEROUS_CHASE is allowed to receive a sizing haircut in the primary
policy. HEALTHY_CHASE receives no penalty.

Point-in-time / leakage rule
----------------------------
- `continuation_quality` and `regime_support` use signal-date features only.
- `up_gap_atr` is only active for a positive entry gap and is known at entry.
- The classification rule is fixed before OOS evaluation; no OOS optimizer.
- No realized return/path/exit information is used to form the classification.

V3 classification rule (fixed, pre-declared)
---------------------------------------------
Candidate universe:
    chase_risk >= 0.20

Danger score:
    60% * quality_deficit
  + 25% * regime_weakness
  + 15% * up_gap_stress

where:
    quality_deficit = 1 - continuation_quality
    regime_weakness = 1 - regime_support

Dangerous chase:
    candidate AND danger_score >= 0.55

Healthy chase:
    candidate AND NOT dangerous

Hard danger override:
    candidate AND continuation_quality < 0.45
              AND regime_support < 0.45

The hard override is a fixed safety rule, not learned from OOS returns.

Policies
--------
BASE                    : no intervention
DANGEROUS_SOFT          : dangerous only -> 0.80x  [PRIMARY]
DANGEROUS_BALANCED      : dangerous only -> 0.65x
DANGEROUS_STRONG        : dangerous only -> 0.50x
ALL_CHASE_SOFT          : all V2 chase candidates -> 0.80x (control)
HEALTHY_PENALTY_CONTROL : healthy chase only -> 0.80x (negative-control)
DANGEROUS_SKIP          : dangerous only -> 0.00x

The control policies help determine whether the classifier is actually
isolating the economically weaker subset rather than merely shrinking any
chase signal.

Production protection
---------------------
Production engine.py, portfolio.py, config.py, signals.py, state, orders,
NAV and trade files are never modified. The research only changes BUY amount
inside an in-memory wrapper.

Outputs
-------
- gmq_momentum_entry_timing_chase_avoidance_v3_raporu.md
- gmq_momentum_entry_timing_chase_avoidance_v3_metrics.csv
- gmq_momentum_entry_timing_chase_avoidance_v3_folds.csv
- gmq_momentum_entry_timing_chase_avoidance_v3_monthly.csv
- gmq_momentum_entry_timing_chase_avoidance_v3_stress.csv
- gmq_momentum_entry_timing_chase_avoidance_v3_diagnostics.csv
- gmq_momentum_entry_timing_chase_avoidance_v3_classification.csv
- gmq_momentum_entry_timing_chase_avoidance_v3_policy_usage.csv
- gmq_momentum_entry_timing_chase_avoidance_v3_predictions.csv
- gmq_momentum_entry_timing_chase_avoidance_v3_trades.csv

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
DEFAULT_YEARS = 14
DEFAULT_START = "2018-01-01"
PRIMARY_POLICY = "DANGEROUS_SOFT"
FOLD_MONTHS = 6
ROBUST_WINDOW = 504
ROBUST_MIN_PERIODS = 60

QUALITY_COMPONENTS = {
    "ret_5d_pct": 0.15,
    "ret_21d_pct": 0.25,
    "ret_63d_pct": 0.15,
    "stock_vs_market_21d_pp": 0.20,
    "market_breadth_21d": 0.15,
    "score_pctile": 0.10,
}

REGIME_COMPONENTS = {
    "market_breadth_21d": 0.60,
    "market_median_21d_pct": 0.40,
}

POLICIES = {
    "BASE": {"mode": "none", "multiplier": 1.00},
    "DANGEROUS_SOFT": {"mode": "dangerous", "multiplier": 0.80},
    "DANGEROUS_BALANCED": {"mode": "dangerous", "multiplier": 0.65},
    "DANGEROUS_STRONG": {"mode": "dangerous", "multiplier": 0.50},
    "ALL_CHASE_SOFT": {"mode": "candidate", "multiplier": 0.80},
    "HEALTHY_PENALTY_CONTROL": {"mode": "healthy", "multiplier": 0.80},
    "DANGEROUS_SKIP": {"mode": "dangerous", "multiplier": 0.00},
}

REQUIRED_COLUMNS = [
    "market", "ticker", "signal_date", "entry_date", "ret_pct",
    "score_pctile", "ret_5d_pct", "ret_21d_pct", "ret_63d_pct",
    "market_breadth_21d", "market_median_21d_pct",
    "stock_vs_market_21d_pp", "gap_pct", "gap_atr",
]


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


def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sigmoid(x: pd.Series) -> pd.Series:
    z = safe_num(x).clip(-6.0, 6.0)
    return 1.0 / (1.0 + np.exp(-z))


def robust_prior_score(
    values: pd.Series,
    window: int = ROBUST_WINDOW,
    min_periods: int = ROBUST_MIN_PERIODS,
) -> pd.Series:
    s = safe_num(values)
    prior = s.shift(1)
    med = prior.rolling(window=window, min_periods=min_periods).median()
    q25 = prior.rolling(window=window, min_periods=min_periods).quantile(0.25)
    q75 = prior.rolling(window=window, min_periods=min_periods).quantile(0.75)
    scale = ((q75 - q25) / 1.349).replace(0.0, np.nan)
    z = (s - med) / scale
    return sigmoid(z).fillna(0.50)


def add_adaptive_features(ds: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in ds.columns]
    if missing:
        raise ValueError(f"Eksik V3 feature: {missing}")

    x = ds.copy()
    x["signal_date"] = normalize_date_series(x["signal_date"])
    x["entry_date"] = normalize_date_series(x["entry_date"])
    for c in REQUIRED_COLUMNS:
        if c not in {"market", "ticker", "signal_date", "entry_date"}:
            x[c] = safe_num(x[c])

    x = x.sort_values(["market", "signal_date", "ticker"]).reset_index(drop=True)

    quality_parts = []
    for col, weight in QUALITY_COMPONENTS.items():
        comp = pd.Series(np.nan, index=x.index, dtype=float)
        for _market, g in x.groupby("market", sort=False):
            comp.loc[g.index] = robust_prior_score(
                g[col], window=ROBUST_WINDOW, min_periods=ROBUST_MIN_PERIODS
            ).to_numpy()
        x[f"quality_{col}"] = comp
        quality_parts.append(weight * comp)
    x["continuation_quality"] = pd.concat(quality_parts, axis=1).sum(axis=1)

    regime_parts = []
    for col, weight in REGIME_COMPONENTS.items():
        comp = pd.Series(np.nan, index=x.index, dtype=float)
        for _market, g in x.groupby("market", sort=False):
            comp.loc[g.index] = robust_prior_score(
                g[col], window=ROBUST_WINDOW, min_periods=ROBUST_MIN_PERIODS
            ).to_numpy()
        x[f"regime_{col}"] = comp
        regime_parts.append(weight * comp)
    x["regime_support"] = pd.concat(regime_parts, axis=1).sum(axis=1)
    x["fragile_regime"] = x["regime_support"] < 0.40

    # Directional entry chase: positive gap only.
    x["up_gap_atr"] = np.where(x["gap_pct"] > 0.0, x["gap_atr"], 0.0)
    x["down_gap_atr"] = np.where(x["gap_pct"] < 0.0, x["gap_atr"], 0.0)
    x["up_gap_stress"] = ((x["up_gap_atr"] - 1.0) / 1.5).clip(0.0, 1.0)
    x["quality_deficit"] = (1.0 - x["continuation_quality"]).clip(0.0, 1.0)
    x["fragile_amplifier"] = np.where(x["fragile_regime"], 1.25, 1.00)
    x["chase_risk"] = (
        x["up_gap_stress"] * x["quality_deficit"] * x["fragile_amplifier"]
    ).clip(0.0, 1.0)

    # V3: preserve the V2 candidate universe and classify it into healthy vs dangerous.
    x["chase_candidate"] = x["chase_risk"] >= 0.20
    x["regime_weakness"] = (1.0 - x["regime_support"]).clip(0.0, 1.0)
    x["danger_score"] = (
        0.60 * x["quality_deficit"]
        + 0.25 * x["regime_weakness"]
        + 0.15 * x["up_gap_stress"]
    ).clip(0.0, 1.0)
    x["hard_danger_override"] = (
        x["chase_candidate"]
        & (x["continuation_quality"] < 0.45)
        & (x["regime_support"] < 0.45)
    )
    x["dangerous_chase"] = (
        x["chase_candidate"]
        & ((x["danger_score"] >= 0.55) | x["hard_danger_override"])
    )
    x["healthy_chase"] = x["chase_candidate"] & ~x["dangerous_chase"]
    x["chase_class"] = np.select(
        [x["dangerous_chase"], x["healthy_chase"]],
        ["DANGEROUS_CHASE", "HEALTHY_CHASE"],
        default="NON_CHASE",
    )

    # Readable fixed diagnostics. No OOS outcome is used here.
    x["flag_up_gap_125"] = x["up_gap_atr"] > 1.25
    x["flag_up_gap_150"] = x["up_gap_atr"] > 1.50
    x["flag_lowq_045"] = x["continuation_quality"] < 0.45
    x["flag_lowq_055"] = x["continuation_quality"] < 0.55
    x["flag_fragile"] = x["fragile_regime"]
    x["flag_chase_risk_020"] = x["chase_risk"] >= 0.20
    x["flag_chase_risk_030"] = x["chase_risk"] >= 0.30
    return x


def policy_condition(x: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "none":
        return pd.Series(False, index=x.index)
    if mode == "dangerous":
        return x["dangerous_chase"].fillna(False)
    if mode == "candidate":
        return x["chase_candidate"].fillna(False)
    if mode == "healthy":
        return x["healthy_chase"].fillna(False)
    raise KeyError(mode)


def apply_policy(x: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in POLICIES:
        raise KeyError(name)
    spec = POLICIES[name]
    z = x.copy()
    cond = policy_condition(z, spec["mode"])
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


def metrics_from_daily(daily: pd.Series) -> Dict[str, float]:
    s = daily.dropna().astype(float)
    if len(s) < 2:
        return {}
    r = s.pct_change().dropna()
    years = max((s.index[-1] - s.index[0]).days / 365.25, 1 / 365.25)
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0)
    peak = s.cummax()
    dd = s / peak - 1.0
    pos = r[r > 0]
    neg = -r[r < 0]
    pf = float(pos.sum() / neg.sum()) if len(neg) and neg.sum() > 0 else float("nan")
    win = float((r > 0).mean()) if len(r) else float("nan")
    roll = (s / s.shift(252) - 1.0).dropna()
    worst12 = float(roll.min()) if len(roll) else float("nan")
    return {
        "start_nav": float(s.iloc[0]),
        "end_nav": float(s.iloc[-1]),
        "cagr_pct": cagr * 100.0,
        "maxdd_pct": float(dd.min() * 100.0),
        "profit_factor": pf,
        "daily_win_pct": win * 100.0,
        "worst_12m_pct": worst12 * 100.0,
    }


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


def fold_metrics(runs: Dict[str, Tuple[dict, pd.Series]], calendar: pd.DataFrame, metrics_fn) -> pd.DataFrame:
    rows = []
    base_daily = runs["BASE"][1]
    for row in calendar.to_dict("records"):
        t0 = pd.Timestamp(row["oos_start"])
        t1 = pd.Timestamp(row["oos_end"])
        b = base_daily[(base_daily.index >= t0) & (base_daily.index < t1)]
        if len(b) < 2:
            continue
        bm = metrics_fn(b)
        b_ret = float(b.iloc[-1] / b.iloc[0] - 1)
        for name, (_st, daily) in runs.items():
            s = daily[(daily.index >= t0) & (daily.index < t1)]
            if len(s) < 2:
                continue
            m = metrics_fn(s)
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


def bootstrap_class_gap(pred: pd.DataFrame, n_boot: int = 10000, seed: int = 20261008) -> dict:
    d = pred.loc[pred["dangerous_chase"], "ret_pct"].dropna().to_numpy(float)
    h = pred.loc[pred["healthy_chase"], "ret_pct"].dropna().to_numpy(float)
    if len(d) < 10 or len(h) < 10:
        return {"danger_mean": np.nan, "healthy_mean": np.nan, "delta_pp": np.nan, "ci5_pp": np.nan, "ci95_pp": np.nan, "p_danger_ge_healthy": np.nan}
    rng = np.random.default_rng(seed)
    danger_idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    healthy_idx = rng.integers(0, len(h), size=(n_boot, len(h)))
    deltas = d[danger_idx].mean(axis=1) - h[healthy_idx].mean(axis=1)
    return {
        "danger_mean": float(d.mean()),
        "healthy_mean": float(h.mean()),
        "delta_pp": float(d.mean() - h.mean()),
        "ci5_pp": float(np.quantile(deltas, 0.05)),
        "ci95_pp": float(np.quantile(deltas, 0.95)),
        "p_danger_ge_healthy": float((deltas >= 0).mean()),
    }


def policy_usage(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in POLICIES:
        z = apply_policy(pred, name)
        cond = z["timing_condition"]
        g = z.loc[cond, "ret_pct"]
        rows.append({
            "policy": name,
            "mode": POLICIES[name]["mode"],
            "multiplier_when_triggered": POLICIES[name]["multiplier"],
            "n": len(z),
            "condition_n": int(cond.sum()),
            "condition_pct": float(cond.mean() * 100.0),
            "mean_multiplier": float(z["sizing_multiplier"].mean()),
            "mean_trigger_ret_pct": float(g.mean()) if len(g) else np.nan,
            "win_trigger_pct": float((g > 0).mean() * 100.0) if len(g) else np.nan,
            "tail10_trigger_pct": float((g <= -10).mean() * 100.0) if len(g) else np.nan,
            "tail15_trigger_pct": float((g <= -15).mean() * 100.0) if len(g) else np.nan,
        })
    return pd.DataFrame(rows)


def classification_table(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "NON_CHASE": pred["chase_class"] == "NON_CHASE",
        "HEALTHY_CHASE": pred["chase_class"] == "HEALTHY_CHASE",
        "DANGEROUS_CHASE": pred["chase_class"] == "DANGEROUS_CHASE",
        "ALL_CHASE": pred["chase_candidate"],
    }
    for name, cond in groups.items():
        g = pred.loc[cond].copy()
        rows.append({
            "class": name,
            "n": len(g),
            "share_pct": float(len(g) / max(len(pred), 1) * 100.0),
            "mean_danger_score": float(g["danger_score"].mean()) if len(g) else np.nan,
            "mean_quality": float(g["continuation_quality"].mean()) if len(g) else np.nan,
            "mean_regime_support": float(g["regime_support"].mean()) if len(g) else np.nan,
            "mean_up_gap_atr": float(g["up_gap_atr"].mean()) if len(g) else np.nan,
            "win_rate_pct": float((g["ret_pct"] > 0).mean() * 100.0) if len(g) else np.nan,
            "mean_ret_pct": float(g["ret_pct"].mean()) if len(g) else np.nan,
            "median_ret_pct": float(g["ret_pct"].median()) if len(g) else np.nan,
            "tail10_rate_pct": float((g["ret_pct"] <= -10).mean() * 100.0) if len(g) else np.nan,
            "tail15_rate_pct": float((g["ret_pct"] <= -15).mean() * 100.0) if len(g) else np.nan,
            "bist_mean_ret_pct": float(g.loc[g.market == "bist", "ret_pct"].mean()) if (g.market == "bist").any() else np.nan,
            "us_mean_ret_pct": float(g.loc[g.market == "us", "ret_pct"].mean()) if (g.market == "us").any() else np.nan,
        })
    return pd.DataFrame(rows)


def diagnostics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = {
        "all_up_gap": pred["up_gap_atr"] > 0,
        "up_gap_125": pred["up_gap_atr"] > 1.25,
        "up_gap_150": pred["up_gap_atr"] > 1.50,
        "chase_risk_020": pred["chase_risk"] >= 0.20,
        "dangerous_chase": pred["dangerous_chase"],
        "healthy_chase": pred["healthy_chase"],
        "down_gap_150_control": pred["down_gap_atr"] > 1.50,
        "hard_danger_override": pred["hard_danger_override"],
    }
    for name, cond in scenarios.items():
        g = pred.loc[cond].copy()
        rows.append({
            "scenario": name,
            "n": len(g),
            "share_pct": float(len(g) / max(len(pred), 1) * 100.0),
            "win_rate_pct": float((g["ret_pct"] > 0).mean() * 100.0) if len(g) else np.nan,
            "mean_ret_pct": float(g["ret_pct"].mean()) if len(g) else np.nan,
            "median_ret_pct": float(g["ret_pct"].median()) if len(g) else np.nan,
            "tail10_rate_pct": float((g["ret_pct"] <= -10).mean() * 100.0) if len(g) else np.nan,
            "tail15_rate_pct": float((g["ret_pct"] <= -15).mean() * 100.0) if len(g) else np.nan,
            "bist_mean_ret_pct": float(g.loc[g.market == "bist", "ret_pct"].mean()) if (g.market == "bist").any() else np.nan,
            "us_mean_ret_pct": float(g.loc[g.market == "us", "ret_pct"].mean()) if (g.market == "us").any() else np.nan,
        })
    return pd.DataFrame(rows)


def build_feature_dataset(state, md_by_mk, W_by_mk) -> pd.DataFrame:
    from gmq_early_adverse_predictor import build_trade_dataset, extract_trades, normalize_trades

    raw = extract_trades(state)
    if raw.empty:
        raise RuntimeError("BASE trade log boş.")
    trades = normalize_trades(raw, W_by_mk)
    ds = build_trade_dataset(trades, md_by_mk, W_by_mk)
    if ds.empty:
        raise RuntimeError("Feature dataset boş.")
    ds["signal_date"] = normalize_date_series(ds["signal_date"])
    ds["entry_date"] = normalize_date_series(ds["entry_date"])
    ds["exit_date"] = normalize_date_series(ds["exit_date"])
    ds["ret_pct"] = pd.to_numeric(ds["ret_pct"], errors="coerce")
    ds = ds.dropna(subset=["signal_date", "ret_pct"]).copy()
    return ds.sort_values(["signal_date", "market", "ticker"]).reset_index(drop=True)


def run_engine_backtest(
    md_b,
    md_u,
    fx,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    cost_rt: float,
    lookups_by_policy: Dict[str, Dict[Tuple[str, str, str], float]] | None,
    capital: float,
):
    import portfolio as P
    import engine as E
    import config as C

    old_fc_b, old_fc_u = md_b.fc, md_u.fc
    md_b.fc = cost_rt / 200.0
    md_u.fc = cost_rt / 200.0
    old_step = E.step_market

    P.set_hist_rates(None)
    E.step_market = old_step
    rb, _ = P.self_financed_returns(md_b)
    ru_usd, _ = P.self_financed_returns(md_u)
    ru = P._to_tl_returns(ru_usd, "us", fx)
    P.VOLSCALE["bist"] = P.make_vol_scale(rb)
    P.VOLSCALE["us"] = P.make_vol_scale(ru_usd)

    def run_single(policy_name: str, lookup: Dict[Tuple[str, str, str], float] | None):
        state = P.new_state(capital, P.fx_at(fx, start))

        def patched_step(md, i, st, ctx):
            ev = old_step(md, i, st, ctx)
            if not lookup or st.get("paused"):
                return ev
            d = pd.Timestamp(md.dates[i]).strftime("%Y-%m-%d")
            changed = []
            for o in st.get("pending", []):
                if o.get("side") != "buy":
                    continue
                key = (md.mk, str(o.get("t")), d)
                m = lookup.get(key)
                if m is None or not np.isfinite(m) or m >= 0.999999:
                    continue
                old_amt = float(o.get("amount", 0.0))
                new_amt = old_amt * float(m)
                o["amount"] = new_amt
                o["full_research"] = old_amt
                changed.append({
                    "ticker": str(o.get("t")),
                    "signal_date": d,
                    "multiplier": float(m),
                    "amount_before": old_amt,
                    "amount_after": new_amt,
                    "saved_cash": old_amt - new_amt,
                    "tranche": o.get("tranche"),
                })
            if changed:
                ev.setdefault("notes", []).append(
                    f"Momentum V3 sizing: {policy_name} · {len(changed)} BUY emri ölçeklendi."
                )
                ev["momentum_v3_sizing"] = changed
                ev["shortfall"] = 0.0
            return ev

        E.step_market = patched_step
        try:
            def shadow(mk, d):
                s = rb if mk == "bist" else ru
                return s[s.index < pd.Timestamp(d)].tail(P.C.RP_WINDOW + 5)

            days = sorted(
                d for d in set(md_b.dates + md_u.dates)
                if start <= pd.Timestamp(d) and (end is None or pd.Timestamp(d) <= end)
            )
            for d in days:
                if d in md_b.didx:
                    P.process_day(state, "bist", md_b, md_b.didx[d], fx, shadow)
                if d in md_u.didx:
                    P.process_day(state, "us", md_u, md_u.didx[d], fx, shadow)
            nav = pd.DataFrame(state["pf"]["nav_tl"], columns=["date", "mk", "total_tl", "fx"])
            nav["date"] = pd.to_datetime(nav["date"])
            daily = nav.groupby("date")["total_tl"].last().sort_index()
            return state, daily
        finally:
            E.step_market = old_step

    out = {}
    if lookups_by_policy is None:
        out["BASE"] = run_single("BASE", None)
    else:
        for name, lookup in lookups_by_policy.items():
            out[name] = run_single(name, lookup)
    md_b.fc, md_u.fc = old_fc_b, old_fc_u
    return out


def write_report(
    out_dir: Path,
    pred: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    stress: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    classification: pd.DataFrame,
    usage: pd.DataFrame,
    boot: dict,
    class_boot: dict,
) -> None:
    primary = metrics.loc[metrics.policy == PRIMARY_POLICY].iloc[0]
    p_folds = folds.loc[folds.policy == PRIMARY_POLICY]
    positive_folds = float((p_folds["period_return_delta_pp"] > 0).mean() * 100.0) if len(p_folds) else np.nan
    stress_p = stress.loc[stress.policy == PRIMARY_POLICY]
    stress_positive = int((stress_p["cagr_delta_vs_base_pp"] > 0).sum()) if len(stress_p) else 0
    trigger_n = int(usage.loc[usage.policy == PRIMARY_POLICY, "condition_n"].iloc[0]) if not usage.loc[usage.policy == PRIMARY_POLICY].empty else 0
    danger_n = int(pred["dangerous_chase"].sum())
    healthy_n = int(pred["healthy_chase"].sum())

    # Promotion is deliberately strict. A good classifier must first separate
    # dangerous and healthy chase observations; only then does portfolio impact matter.
    class_separation_ok = bool(
        danger_n >= 10
        and healthy_n >= 10
        and np.isfinite(class_boot["ci95_pp"])
        and class_boot["ci95_pp"] < 0
        and np.isfinite(class_boot["p_danger_ge_healthy"])
        and class_boot["p_danger_ge_healthy"] < 0.10
    )
    portfolio_ok = bool(
        primary.get("cagr_delta_vs_base_pp", np.nan) >= 0
        and primary.get("maxdd_delta_vs_base_pp", np.nan) >= 0
        and primary.get("pf_delta_vs_base", np.nan) >= 0
        and np.isfinite(boot["ci5_pp"]) and boot["ci5_pp"] > 0
        and np.isfinite(positive_folds) and positive_folds >= 60
        and stress_positive >= 4
        and trigger_n >= 10
    )
    passed = class_separation_ok and portfolio_ok
    verdict = "PROMISING — NEEDS FINAL HOLDOUT / PARITY" if passed else "REJECT FOR PROMOTION"

    lines = [
        "# Global Momentum Quant — Momentum Entry Timing / Chase Avoidance V3",
        "",
        "**RESEARCH ONLY — PRODUCTION DOSYALARINA DOKUNULMADI**",
        "",
        "## 1. Amaç",
        "V2'nin yaklaşık 75 işlemden oluşan chase-risk evrenini yok etmek yerine iki ekonomik sınıfa ayırmak: **HEALTHY_CHASE** ve **DANGEROUS_CHASE**. Primary politika yalnızca dangerous sınıfa boyut keser.",
        "",
        "## 2. Sabit point-in-time sınıflandırma",
        "- Candidate universe: `chase_risk >= 0.20` (V2 ile aynı).",
        "- Danger score = 60% quality deficit + 25% regime weakness + 15% up-gap stress.",
        "- Dangerous: `danger_score >= 0.55`.",
        "- Hard override: candidate + continuation_quality < 0.45 + regime_support < 0.45.",
        "- Healthy: kalan candidate işlemler.",
        "- Hiçbir realized return/path/exit değişkeni sınıflandırmada kullanılmaz.",
        "",
        "## 3. Classification results",
        classification.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 4. Dangerous vs healthy separation bootstrap",
        f"Dangerous n: **{danger_n}** · Healthy n: **{healthy_n}**",
        f"Dangerous mean return: **{class_boot['danger_mean']:.4f}%**",
        f"Healthy mean return: **{class_boot['healthy_mean']:.4f}%**",
        f"Dangerous - Healthy: **{class_boot['delta_pp']:.4f} pp**",
        f"Bootstrap CI 5–95: **[{class_boot['ci5_pp']:.4f}, {class_boot['ci95_pp']:.4f}] pp**",
        f"P(Dangerous mean >= Healthy mean): **{class_boot['p_danger_ge_healthy']:.4f}**",
        "",
        "## 5. Policy usage",
        usage.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. Diagnostics",
        diagnostics_df.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 7. Portfolio metrics",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 8. Six-month robustness",
        folds.to_markdown(index=False, floatfmt=".4f") if not folds.empty else "—",
        "",
        "## 9. Matched-month bootstrap — primary",
        f"Months: **{boot['n_months']}**",
        f"Mean monthly delta: **{boot['mean_pp']:.4f} pp**",
        f"Bootstrap CI 5–95: **[{boot['ci5_pp']:.4f}, {boot['ci95_pp']:.4f}] pp**",
        f"P(mean delta <= 0): **{boot['p_le0']:.4f}**",
        f"Positive months: **{boot['positive_pct']:.2f}%**",
        "",
        "## 10. Cost stress",
        stress.to_markdown(index=False, floatfmt=".4f"),
        f"Positive stress scenarios for primary: **{stress_positive}/{len(stress_p)}**",
        "",
        "## 11. Decision gate",
        "1) Dangerous ve healthy sınıfları istatistiksel olarak ayrışmalı; 2) primary dangerous-only sizing portföy ölçümlerinde zarar vermemeli; 3) six-month ve cost-stress testleri birlikte pozitif kalmalı.",
        f"Classification separation gate: **{'PASS' if class_separation_ok else 'FAIL'}**",
        f"Portfolio robustness gate: **{'PASS' if portfolio_ok else 'FAIL'}**",
        "",
        f"# FINAL VERDICT\n**{verdict}**",
        "",
        "Not: Bu araştırmada trade-level win rate değişmez; yalnızca BUY amount sizing değişir.",
        "Production engine.py / portfolio.py / config.py / signals.py ve state/NAV/order dosyaları değiştirilmez.",
    ]
    (out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_raporu.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def synthetic_smoke(out_dir: Path) -> None:
    rng = np.random.default_rng(20261008)
    n = 9000
    dates = pd.bdate_range("2016-01-04", periods=700)
    signal_dates = np.tile(dates.to_numpy(), int(np.ceil(n / len(dates))))[:n]
    market = np.where(np.arange(n) % 2 == 0, "bist", "us")
    ticker = [f"T{i % 150:03d}" for i in range(n)]
    ret5 = rng.normal(1.5, 5.0, n)
    ret21 = ret5 + rng.normal(2.5, 8.0, n)
    ret63 = ret21 + rng.normal(4.0, 11.0, n)
    rel = rng.normal(0.5, 7.0, n)
    breadth = np.clip(rng.normal(0.55, 0.16, n), 0.0, 1.0)
    score = np.clip(rng.normal(0.62, 0.20, n), 0.0, 1.0)
    gap_pct = rng.normal(0.2, 4.0, n)
    gap_atr = np.abs(gap_pct) / np.maximum(rng.normal(2.0, 0.7, n), 0.5)
    market_median = rng.normal(1.0, 3.0, n)
    quality_latent = 0.20 * ret5 + 0.35 * ret21 + 0.15 * ret63 + 0.20 * rel + 0.10 * (breadth - 0.5) * 20
    ret = 2.0 + 0.18 * quality_latent + rng.normal(0, 7.0, n)
    ret -= np.where((gap_pct > 0) & (gap_atr > 1.25) & (quality_latent < 0) & (breadth < 0.5), 3.5, 0.0)

    ds = pd.DataFrame({
        "market": market,
        "ticker": ticker,
        "signal_date": signal_dates,
        "entry_date": signal_dates + pd.Timedelta(days=1),
        "ret_pct": ret,
        "score_pctile": score,
        "ret_5d_pct": ret5,
        "ret_21d_pct": ret21,
        "ret_63d_pct": ret63,
        "market_breadth_21d": breadth,
        "market_median_21d_pct": market_median,
        "stock_vs_market_21d_pp": rel,
        "gap_pct": gap_pct,
        "gap_atr": gap_atr,
    })

    p = add_adaptive_features(ds)
    usage = policy_usage(p)
    cls = classification_table(p)
    diag = diagnostics(p)
    for name in POLICIES:
        z = apply_policy(p, name)
        assert len(z) == len(p)
        assert ((z["sizing_multiplier"] >= 0) & (z["sizing_multiplier"] <= 1)).all()
    assert (p.loc[p.gap_pct < 0, "up_gap_atr"] == 0).all()
    assert not ((p["dangerous_chase"]) & (p["healthy_chase"])).any()
    assert (p.loc[p["chase_candidate"], "chase_class"] != "NON_CHASE").all()

    p.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_predictions.csv", index=False)
    usage.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_policy_usage.csv", index=False)
    cls.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_classification.csv", index=False)
    diag.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_diagnostics.csv", index=False)
    (out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_metrics.csv").write_text("synthetic_smoke,ok\n", encoding="utf-8")
    print("SMOKE OK — V2 candidate universe / healthy-vs-dangerous classification / policy path")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--out", default="research/results_momentum_entry_timing_chase_avoidance_v3")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start).normalize()

    if args.synthetic:
        synthetic_smoke(out_dir)
        return

    helper = load_research_helper()

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
    base_state = helper.run_engine_backtest(
        md_b, md_u, fx, start_full, None, PRIMARY_COST_RT, None, C.CAPITAL_TL
    )["BASE"][0]
    ds = build_feature_dataset(base_state, {"bist": md_b, "us": md_u}, W)
    ds = ds.dropna(subset=["signal_date", "ret_pct"]).copy()
    ds = ds[ds["signal_date"] >= start].copy()
    if ds.empty:
        raise RuntimeError("Başlangıç tarihinden sonra trade yok.")

    pred = add_adaptive_features(ds)
    usage = policy_usage(pred)
    cls = classification_table(pred)
    diag = diagnostics(pred)
    usage.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_policy_usage.csv", index=False)
    cls.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_classification.csv", index=False)
    diag.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_diagnostics.csv", index=False)

    lookups = {}
    policy_parts = []
    for name in POLICIES:
        z = apply_policy(pred, name)
        lookups[name] = make_lookup(z)
        policy_parts.append(z)
    pred.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_predictions.csv", index=False)
    pd.concat(policy_parts, ignore_index=True).to_csv(
        out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_trades.csv", index=False
    )

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
            "worst12m_delta_vs_base_pp": m["worst_12m_pct"] - bm["worst_12m_pct"],
        })
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_metrics.csv", index=False)

    calendar = six_month_calendar(last_date, start)
    folds = fold_metrics(runs, calendar, metrics_from_daily)
    folds.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_folds.csv", index=False)

    primary_daily = runs[PRIMARY_POLICY][1]
    mdlt = matched_monthly(base_daily, primary_daily)
    mdlt.reset_index().rename(columns={"index": "month"}).to_csv(
        out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_monthly.csv", index=False
    )
    boot = bootstrap(mdlt["delta_pp"])

    stress_rows = []
    for cost in STRESS_COSTS:
        sr = run_engine_backtest(
            md_b, md_u, fx, start, last_date, float(cost),
            {"BASE": None, PRIMARY_POLICY: lookups[PRIMARY_POLICY]}, C.CAPITAL_TL
        )
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
    stress.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v3_stress.csv", index=False)

    class_boot = bootstrap_class_gap(pred)
    write_report(out_dir, pred, metrics, folds, stress, diag, cls, usage, boot, class_boot)

    print("=== GMQ Momentum Entry Timing / Chase Avoidance V3 ===")
    print(metrics[["policy", "cagr_pct", "maxdd_pct", "profit_factor", "cagr_delta_vs_base_pp"]].to_string(index=False))
    print("=== Classification ===")
    print(cls[["class", "n", "mean_ret_pct", "win_rate_pct", "tail10_rate_pct", "tail15_rate_pct"]].to_string(index=False))
    print(f"Dangerous vs healthy mean delta: {class_boot['delta_pp']:.4f} pp")
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
