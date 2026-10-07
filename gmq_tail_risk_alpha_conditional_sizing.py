#!/usr/bin/env python3
"""
Global Momentum Quant — Tail-Risk × Alpha Conditional Sizing Research V1

RESEARCH ONLY.

Purpose
-------
Test whether the tail-risk signal becomes economically useful when it is
conditioned on a separate, signal-date alpha forecast, instead of being used
as a blunt risk-only position-size haircut.

Critical design rules
---------------------
1) Production files are NOT modified.
2) The research backtest imports the repository's live engine.py,
   portfolio.py, data.py and config.py.
3) BASE and conditional-sizing runs use the same production mechanics:
   4 tranches, 5-day stagger, 21-day refresh, costs, risk parity, cash,
   catastrophe stop, and existing order path.
4) The first pass is a strict historical holdout:
      FIT: trades whose signal_date AND exit_date are both before 2022-01-01
      OOS : signal_date >= 2022-01-01
   No holdout observations are used to fit or tune models.
5) Feature inputs are signal-date only. No early-path/future fields are used
   by either alpha or tail models.
6) Risk score = 25% P(tail <= -5%) + 35% P(tail <= -10%) +
   40% P(tail <= -15%). These weights are fixed before the OOS run.
7) Alpha model = robustly clipped signal-date return forecast via fixed Ridge.
8) Alpha and risk are ranked PER MARKET using training-only empirical CDFs.
9) Sizing is conditional only when tail-risk percentile exceeds alpha percentile:
      multiplier = max(floor, 1 - strength * max(risk_pct - alpha_pct, 0))
   Three pre-declared policy strengths are tested. There is NO optimizer.
10) Only BUY order amounts are scaled inside the research wrapper. Sells,
    stops, signal ranking, tranche timing and portfolio risk-parity remain
    untouched.

Outputs
-------
- gmq_tail_risk_alpha_conditional_sizing_raporu.md
- gmq_tail_risk_alpha_conditional_metrics.csv
- gmq_tail_risk_alpha_conditional_monthly.csv
- gmq_tail_risk_alpha_conditional_trades.csv
- gmq_tail_risk_alpha_conditional_predictions.csv
- gmq_tail_risk_alpha_conditional_model_metrics.csv
- gmq_tail_risk_alpha_conditional_policy_usage.csv
- gmq_tail_risk_alpha_conditional_stress.csv
- gmq_tail_risk_alpha_conditional_alpha_bins.csv
- gmq_tail_risk_alpha_conditional_risk_bins.csv

Run on GitHub Actions or locally from repository root.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SPLIT_DATE = pd.Timestamp("2022-01-01")
DEFAULT_YEARS = 14
PRIMARY_COST_RT = 0.35
STRESS_COSTS = (0.35, 0.50, 0.75, 1.00, 1.25)

STRICT_FEATURES = [
    "score_pctile",
    "ret_5d_pct",
    "ret_21d_pct",
    "ret_63d_pct",
    "vol20_ann_pct",
    "atr14_pct",
    "market_breadth_5d",
    "market_breadth_21d",
    "market_median_1d_pct",
    "market_median_21d_pct",
    "market_median_63d_pct",
    "market_dispersion_1d_pct",
    "stock_vs_market_21d_pp",
    "momentum_extension_z",
]

TARGETS = {
    "tail_5": lambda d: (d["ret_pct"] <= -5.0).astype(int),
    "tail_10": lambda d: (d["ret_pct"] <= -10.0).astype(int),
    "tail_15": lambda d: (d["ret_pct"] <= -15.0).astype(int),
}

# Fixed policy family. Not optimized on OOS.
POLICIES = {
    "BASE": {"strength": 0.00, "floor": 1.00},
    "CONDITIONAL_GENTLE": {"strength": 0.35, "floor": 0.65},
    "CONDITIONAL_BALANCED": {"strength": 0.60, "floor": 0.40},
    "CONDITIONAL_STRONG": {"strength": 0.80, "floor": 0.25},
}


def make_classifier() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


def make_alpha_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )


def safe_auc(y: pd.Series, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan")


def safe_pr_auc(y: pd.Series, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if y.nunique() == 2 else float("nan")


def normalize_date_series(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, errors="coerce")
    try:
        x = x.dt.tz_localize(None)
    except TypeError:
        pass
    return x.dt.normalize()


def empirical_percentile(train_values: pd.Series, value: float) -> float:
    a = pd.to_numeric(train_values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 20 or not np.isfinite(value):
        return float("nan")
    # Mid-rank CDF, bounded to [0,1].
    return float((np.sum(a <= value) - 0.5 * np.sum(a == value)) / len(a))


def fit_models(train: pd.DataFrame) -> Tuple[Dict[str, Pipeline], Pipeline, pd.DataFrame]:
    missing = [c for c in STRICT_FEATURES if c not in train.columns]
    if missing:
        raise ValueError(f"Eksik strict feature: {missing}")

    models: Dict[str, Pipeline] = {}
    rows: List[dict] = []
    X = train[STRICT_FEATURES]

    alpha = make_alpha_model()
    # Fixed clipping to prevent catastrophe-stop outliers from dominating the
    # alpha forecast. The clipping rule is set before OOS evaluation.
    y_alpha = train["ret_pct"].clip(-25.0, 25.0).to_numpy(dtype=float)
    alpha.fit(X, y_alpha)
    pred_alpha = alpha.predict(X)
    rows.append(
        {
            "model": "alpha_ridge",
            "train_n": len(train),
            "target": "clipped_ret_pct",
            "r2_train": float(r2_score(y_alpha, pred_alpha)),
            "mae_train": float(mean_absolute_error(y_alpha, pred_alpha)),
            "coef_note": "Ridge(alpha=10), target clipped to [-25,25]%",
        }
    )

    for target, fn in TARGETS.items():
        y = fn(train).astype(int)
        if y.nunique() < 2:
            raise RuntimeError(f"{target}: training target tek sınıf")
        m = make_classifier()
        m.fit(X, y)
        p = m.predict_proba(X)[:, 1]
        rows.append(
            {
                "model": f"risk_{target}",
                "train_n": len(train),
                "target": target,
                "r2_train": np.nan,
                "mae_train": np.nan,
                "auc_train": safe_auc(y, p),
                "pr_auc_train": safe_pr_auc(y, p),
                "coef_note": "LogisticRegression(C=0.25, balanced)",
            }
        )
        models[target] = m

    return models, alpha, pd.DataFrame(rows)


def predict_models(
    df: pd.DataFrame,
    risk_models: Dict[str, Pipeline],
    alpha_model: Pipeline,
) -> pd.DataFrame:
    out = df.copy()
    X = out[STRICT_FEATURES]
    out["pred_alpha_pct"] = alpha_model.predict(X)
    for target, model in risk_models.items():
        out[f"p_{target}"] = model.predict_proba(X)[:, 1]
    out["tail_risk_score"] = (
        0.25 * out["p_tail_5"]
        + 0.35 * out["p_tail_10"]
        + 0.40 * out["p_tail_15"]
    )
    return out


def add_market_percentiles(train_pred: pd.DataFrame, hold_pred: pd.DataFrame) -> pd.DataFrame:
    train_pred = train_pred.copy()
    hold_pred = hold_pred.copy()
    train_pred["sample"] = "train"
    hold_pred["sample"] = "holdout"

    rows = []
    for market in sorted(set(train_pred["market"].astype(str)) | set(hold_pred["market"].astype(str))):
        tr = train_pred[train_pred["market"].astype(str) == market]
        for sample_name, src in (("train", train_pred), ("holdout", hold_pred)):
            g = src[src["market"].astype(str) == market].copy()
            if g.empty:
                continue
            g["alpha_pctile_mkt"] = [
                empirical_percentile(tr["pred_alpha_pct"], v) for v in g["pred_alpha_pct"]
            ]
            g["risk_pctile_mkt"] = [
                empirical_percentile(tr["tail_risk_score"], v) for v in g["tail_risk_score"]
            ]
            rows.append(g)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def attach_multiplier(df: pd.DataFrame, strength: float, floor: float) -> pd.DataFrame:
    x = df.copy()
    gap = (x["risk_pctile_mkt"] - x["alpha_pctile_mkt"]).clip(lower=0.0)
    x["sizing_multiplier"] = np.clip(1.0 - strength * gap, floor, 1.0)
    return x


def make_lookup(predictions: pd.DataFrame) -> Dict[Tuple[str, str, str], float]:
    out: Dict[Tuple[str, str, str], float] = {}
    for r in predictions.to_dict("records"):
        d = pd.Timestamp(r["signal_date"]).strftime("%Y-%m-%d")
        key = (str(r["market"]), str(r["ticker"]), d)
        out[key] = float(r["sizing_multiplier"])
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
    # Worst rolling 252 trading-day period. Uses actual available dates but
    # reports as a one-year proxy, consistent with research-only diagnostics.
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


def monthly_delta(base: pd.Series, cand: pd.Series) -> pd.DataFrame:
    b = base.resample("ME").last().pct_change()
    c = cand.resample("ME").last().pct_change()
    df = pd.concat([b.rename("base_ret"), c.rename("cand_ret")], axis=1).dropna()
    df["delta_pp"] = (df["cand_ret"] - df["base_ret"]) * 100.0
    return df


def bootstrap_ci(values: pd.Series, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 10:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(x)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=n, replace=True).mean()
    return float(np.mean(x)), float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def build_feature_dataset(state, md_by_mk, W_by_mk) -> pd.DataFrame:
    # Reuse the already validated, production-aligned feature builder from the
    # prior Early-Adverse research module. We explicitly use only STRICT_FEATURES
    # downstream, never its future/path labels.
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

    # Shadow series for risk parity are always BASE mechanics; the conditional
    # sizing intervention is applied only to the actual BUY path below.
    old_fc_b, old_fc_u = md_b.fc, md_u.fc
    md_b.fc = cost_rt / 200.0
    md_u.fc = cost_rt / 200.0

    old_step = E.step_market

    # Shadow series must remain the unmodified BASE path for every policy.
    # Compute them before installing the research BUY-sizing wrapper.
    P.set_hist_rates(None)
    E.step_market = old_step
    rb, _ = P.self_financed_returns(md_b)
    ru_usd, _ = P.self_financed_returns(md_u)
    ru = P._to_tl_returns(ru_usd, "us", fx)
    P.VOLSCALE["bist"] = P.make_vol_scale(rb)
    P.VOLSCALE["us"] = P.make_vol_scale(ru_usd)

    def run_single(policy_name: str, lookup: Dict[Tuple[str, str, str], float] | None):
        state = P.new_state(capital, P.fx_at(fx, start))

        # Patch only the research process boundary. Production source code is
        # not edited. The original function creates the exact orders; we then
        # scale newly-created BUY amounts and reserve the difference as cash.
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
                    f"Tail×Alpha sizing: {policy_name} · {len(changed)} BUY emri ölçeklendi."
                )
                ev["tail_alpha_sizing"] = changed
                # The original step may have reported a shortfall based on the
                # unsized target. Since the reduced amount is intentional cash,
                # do not trigger a compensating inter-market transfer.
                ev["shortfall"] = 0.0
            return ev

        E.step_market = patched_step
        try:
            def shadow(mk, d):
                s = rb if mk == "bist" else ru
                return s[s.index < pd.Timestamp(d)].tail(P.C.RP_WINDOW + 5)

            days = sorted(d for d in set(md_b.dates + md_u.dates) if start <= pd.Timestamp(d) and (end is None or pd.Timestamp(d) <= end))
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


def make_bin_table(pred: pd.DataFrame, value_col: str, ret_col: str = "ret_pct") -> pd.DataFrame:
    x = pred.copy()
    x["bin"] = pd.qcut(x[value_col], q=5, labels=False, duplicates="drop")
    rows = []
    for b, g in x.groupby("bin", observed=True):
        rows.append(
            {
                "bin": int(b),
                "n": len(g),
                "mean_signal": float(g[value_col].mean()),
                "win_rate_pct": float((g[ret_col] > 0).mean() * 100.0),
                "mean_ret_pct": float(g[ret_col].mean()),
                "tail10_rate_pct": float((g[ret_col] <= -10).mean() * 100.0),
                "tail15_rate_pct": float((g[ret_col] <= -15).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("bin")


def summary_report(
    out_dir: Path,
    dataset_stats: Dict[str, object],
    model_metrics: pd.DataFrame,
    alpha_bins: pd.DataFrame,
    risk_bins: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    stress: pd.DataFrame,
    usage: pd.DataFrame,
):
    lines = [
        "# Global Momentum Quant — Tail-Risk × Alpha Conditional Sizing V1",
        "",
        f"_Üretim: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')} · primary RT cost: {PRIMARY_COST_RT:.2f}%_",
        "",
        "## 1. Amaç",
        "Tail-risk bilgisinin tek başına pozisyonu küçültmek yerine, sinyal gününde tahmin edilen alpha ile koşullandırıldığında gerçekten ekonomik bir katkı sağlayıp sağlamadığı test edildi.",
        "",
        "## 2. Veri ve leakage kontrolü",
        f"- Toplam üretim trade'i: **{dataset_stats.get('all_n', 0):,}**",
        f"- Model fit satırı: **{dataset_stats.get('train_n', 0):,}**",
        f"- OOS satırı: **{dataset_stats.get('holdout_n', 0):,}**",
        f"- FIT cutoff: **{SPLIT_DATE.date()}**",
        "- FIT'e alınan işlemlerde hem signal_date hem exit_date cutoff öncesindedir.",
        "- Alpha/risk feature'ları yalnızca signal_date t verisinden alınmıştır.",
        "- Production engine.py / portfolio.py / config.py / signals.py değiştirilmemiştir; sizing yalnızca araştırma wrapper'ında uygulanmıştır.",
        "",
        "## 3. Modeller",
        "- Alpha: Ridge(alpha=10), ret_pct hedefi [-25%, +25%] ile sabit clipped.",
        "- Tail-risk: 3 ayrı LogisticRegression(C=0.25, balanced) modeli.",
        "- Kombine risk: 0.25×tail_5 + 0.35×tail_10 + 0.40×tail_15.",
        "- Alpha/risk percentile referansı market bazında ve yalnızca FIT dağılımından türetilmiştir.",
        "",
        "## 4. Model doğrulaması",
        model_metrics.to_markdown(index=False, floatfmt=".4f") if not model_metrics.empty else "—",
        "",
        "## 5. Alpha OOS quintile davranışı",
        alpha_bins.to_markdown(index=False, floatfmt=".4f") if not alpha_bins.empty else "—",
        "",
        "## 6. Tail-risk OOS quintile davranışı",
        risk_bins.to_markdown(index=False, floatfmt=".4f") if not risk_bins.empty else "—",
        "",
        "## 7. Primary OOS portföy karşılaştırması",
        policy_metrics.to_markdown(index=False, floatfmt=".4f") if not policy_metrics.empty else "—",
        "",
        "## 8. Maliyet stresi",
        stress.to_markdown(index=False, floatfmt=".4f") if not stress.empty else "—",
        "",
        "## 9. Policy kullanım denetimi",
        usage.to_markdown(index=False, floatfmt=".4f") if not usage.empty else "—",
        "",
        "## 10. Karar",
        "PRIMARY OOS'da BASE'e karşı CAGR, MaxDD, PF, worst-12M ve bootstrap aylık fark birlikte değerlendirilmelidir.",
        "Bir aday yalnızca MaxDD düşürdüğü için kabul edilmez; CAGR kaybı ve tail-loss katkısı birlikte ölçülür.",
        "Maliyet stresinde yön bozuluyorsa ve/veya policy kullanımı belirli bir market/ticker alt grubuna yoğunlaşıyorsa canlı entegrasyon reddedilir.",
        "",
        "**CANLI DEĞİŞİKLİK YOK — RESEARCH ONLY**",
    ]
    (out_dir / "gmq_tail_risk_alpha_conditional_sizing_raporu.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def synthetic_dataset(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    d = pd.date_range("2014-01-01", periods=n, freq="B")
    market = np.where(np.arange(n) % 2 == 0, "bist", "us")
    score = rng.normal(50, 15, n)
    vol = np.abs(rng.normal(22, 7, n))
    breadth = rng.uniform(0.2, 0.8, n)
    alpha_latent = 0.08 * (score - 50) - 0.04 * (vol - 22) + 4 * (breadth - 0.5)
    ret = alpha_latent + rng.normal(0, 7, n)
    # Inject a tail mechanism where high vol + poor breadth drives the left tail.
    tail_push = np.maximum(vol - 27, 0) * 0.9 + np.maximum(0.45 - breadth, 0) * 18
    ret = ret - tail_push
    out = pd.DataFrame({
        "market": market,
        "ticker": [f"T{i%60:03d}" for i in range(n)],
        "signal_date": d,
        "entry_date": d + pd.Timedelta(days=1),
        "exit_date": d + pd.Timedelta(days=30),
        "ret_pct": ret,
    })
    for c in STRICT_FEATURES:
        out[c] = rng.normal(0, 1, n)
    out["score_pctile"] = score
    out["vol20_ann_pct"] = vol
    out["market_breadth_21d"] = breadth
    out["ret_5d_pct"] = score / 6 + rng.normal(0, 4, n)
    out["ret_21d_pct"] = score / 3 + rng.normal(0, 6, n)
    out["ret_63d_pct"] = score / 2 + rng.normal(0, 8, n)
    out["atr14_pct"] = vol / 4
    out["market_breadth_5d"] = breadth + rng.normal(0, 0.05, n)
    out["stock_vs_market_21d_pp"] = out["ret_21d_pct"] - (breadth - 0.5) * 20
    out["momentum_extension_z"] = (out["ret_5d_pct"] / (out["vol20_ann_pct"] + 1e-6))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--out", default="research/results_tail_risk_alpha_conditional")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--cost", type=float, default=PRIMARY_COST_RT)
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        # Smoke path validates model and reporting only; it does not represent
        # a strategy result and never touches production modules.
        ds = synthetic_dataset()
        ds["signal_date"] = normalize_date_series(ds["signal_date"])
        ds["entry_date"] = normalize_date_series(ds["entry_date"])
        ds["exit_date"] = normalize_date_series(ds["exit_date"])
        fit = ds[ds.signal_date < SPLIT_DATE].copy()
        oos = ds[ds.signal_date >= SPLIT_DATE].copy()
        models, alpha, mm = fit_models(fit)
        trp = predict_models(fit, models, alpha)
        hop = predict_models(oos, models, alpha)
        pred = add_market_percentiles(trp, hop)
        alpha_bins = make_bin_table(pred[pred["sample"] == "holdout"], "pred_alpha_pct")
        risk_bins = make_bin_table(pred[pred["sample"] == "holdout"], "tail_risk_score")
        pred.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_predictions.csv", index=False)
        mm.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_model_metrics.csv", index=False)
        alpha_bins.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_alpha_bins.csv", index=False)
        risk_bins.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_risk_bins.csv", index=False)
        (out_dir / "gmq_tail_risk_alpha_conditional_metrics.csv").write_text("synthetic_smoke,ok\n", encoding="utf-8")
        print("SMOKE OK — synthetic only")
        return

    # Production-aligned historical dataset construction.
    import data as DA
    import engine as E
    import portfolio as P
    import config as C

    years = max(int(args.years), 3)
    panels = {
        "bist": DA.get_panel("bist", years=years, force_full=True),
        "us": DA.get_panel("us", years=years, force_full=True),
    }
    if panels["bist"].empty or panels["us"].empty:
        raise RuntimeError("BIST veya US paneli boş.")
    fx = DA.get_fx(years)
    if fx.empty:
        raise RuntimeError("USD/TRY boş.")
    W = {mk: DA.to_wide(panel) for mk, panel in panels.items()}
    md_b = E.MarketData("bist", W["bist"], spec=C.MARKETS["bist"]["spec"])
    md_u = E.MarketData("us", W["us"], spec=C.MARKETS["us"]["spec"])

    warm = max(260, P.C.RP_WINDOW + 5)
    start_full = max(pd.Timestamp(md_b.dates[warm]), pd.Timestamp(md_u.dates[warm]))
    # Build a full BASE trade history once. The model is fit only on realized
    # pre-cutoff trades whose exit is also pre-cutoff.
    base_runs = run_engine_backtest(md_b, md_u, fx, start_full, None, PRIMARY_COST_RT, None, C.CAPITAL_TL)
    base_state, base_daily = base_runs["BASE"]
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

    # OOS prediction artifact contains no future/path features beyond the actual
    # return label needed for offline evaluation. Sizing map uses signal keys only.
    pred.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_predictions.csv", index=False)
    model_metrics.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_model_metrics.csv", index=False)

    hold_p = pred[pred["sample"] == "holdout"].copy()
    alpha_bins = make_bin_table(hold_p, "pred_alpha_pct")
    risk_bins = make_bin_table(hold_p, "tail_risk_score")
    alpha_bins.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_alpha_bins.csv", index=False)
    risk_bins.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_risk_bins.csv", index=False)

    lookups = {}
    usage_rows = []
    for name, spec in POLICIES.items():
        hp = attach_multiplier(hold_p, spec["strength"], spec["floor"])
        lookups[name] = make_lookup(hp)
        usage_rows.append(
            {
                "policy": name,
                "strength": spec["strength"],
                "floor": spec["floor"],
                "oos_n": len(hp),
                "mean_multiplier": float(hp["sizing_multiplier"].mean()),
                "median_multiplier": float(hp["sizing_multiplier"].median()),
                "pct_below_1": float((hp["sizing_multiplier"] < 0.999999).mean() * 100),
                "pct_floor": float((hp["sizing_multiplier"] <= spec["floor"] + 1e-12).mean() * 100),
            }
        )
    usage = pd.DataFrame(usage_rows)
    usage.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_policy_usage.csv", index=False)

    hold_pred = hold_p.copy()
    hold_pred["policy"] = "BASE"
    hold_pred["sizing_multiplier"] = 1.0
    policy_trade_rows = [hold_pred]
    for name, spec in POLICIES.items():
        if name == "BASE":
            continue
        x = hold_p.copy()
        x["policy"] = name
        x["sizing_multiplier"] = attach_multiplier(x, spec["strength"], spec["floor"])["sizing_multiplier"]
        policy_trade_rows.append(x)
    policy_trades = pd.concat(policy_trade_rows, ignore_index=True)
    policy_trades.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_trades.csv", index=False)

    # OOS BASE vs candidate runs. start at cutoff so all policies share exactly
    # the same OOS production path and capitalization.
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

    policy_rows = []
    base_daily_oos = runs["BASE"][1]
    base_m = metrics_from_daily(base_daily_oos)
    for name, (state, daily) in runs.items():
        m = metrics_from_daily(daily)
        row = {"policy": name, **m}
        if name != "BASE":
            row["cagr_delta_pp"] = m["cagr_pct"] - base_m["cagr_pct"]
            row["maxdd_delta_pp"] = m["maxdd_pct"] - base_m["maxdd_pct"]
            md = monthly_delta(base_daily_oos, daily)
            mean_delta, ci5, ci95 = bootstrap_ci(md["delta_pp"], seed=20261007)
            row["bootstrap_monthly_delta_pp"] = mean_delta
            row["bootstrap_ci5_pp"] = ci5
            row["bootstrap_ci95_pp"] = ci95
            row["bootstrap_p_delta_le0"] = float((
                np.random.default_rng(20261008).choice(
                    md["delta_pp"].to_numpy(), size=(2000, len(md)), replace=True
                ).mean(axis=1) <= 0
            ).mean()) if len(md) >= 10 else np.nan
        else:
            row["cagr_delta_pp"] = 0.0
            row["maxdd_delta_pp"] = 0.0
            row["bootstrap_monthly_delta_pp"] = 0.0
            row["bootstrap_ci5_pp"] = 0.0
            row["bootstrap_ci95_pp"] = 0.0
            row["bootstrap_p_delta_le0"] = 0.0
        row["trade_count"] = sum(len(v.get("markets",{}).get(mk,{}).get("trades",[])) for mk in ("bist","us") for v in [state])
        # Tail-loss contribution at trade level, as a diagnostics-only ratio.
        all_tr = []
        for mk in ("bist", "us"):
            all_tr.extend(state.get("markets", {}).get(mk, {}).get("trades", []) or [])
        trdf = pd.DataFrame(all_tr)
        if not trdf.empty:
            neg = trdf.loc[trdf["ret_pct"] < 0, "ret_pct"]
            tail15 = trdf.loc[trdf["ret_pct"] <= -15, "ret_pct"]
            row["abs_negative_loss_sum_pct"] = float(-neg.sum()) if len(neg) else 0.0
            row["tail15_loss_share_pct"] = float((-tail15.sum()) / (-neg.sum()) * 100.0) if len(tail15) and len(neg) and neg.sum() < 0 else np.nan
        else:
            row["abs_negative_loss_sum_pct"] = np.nan
            row["tail15_loss_share_pct"] = np.nan
        policy_rows.append(row)
    policy_metrics = pd.DataFrame(policy_rows)
    policy_metrics.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_metrics.csv", index=False)

    # Cost stress uses FIXED policies and FIXED model parameters/predictions.
    stress_rows = []
    for cost in STRESS_COSTS:
        cost_runs = run_engine_backtest(md_b, md_u, fx, SPLIT_DATE, hold_end, cost, lookups, C.CAPITAL_TL)
        b_daily = cost_runs["BASE"][1]
        bm = metrics_from_daily(b_daily)
        for name, (_st, daily) in cost_runs.items():
            cm = metrics_from_daily(daily)
            stress_rows.append(
                {
                    "cost_rt_pct": cost,
                    "policy": name,
                    "cagr_pct": cm["cagr_pct"],
                    "cagr_delta_vs_base_pp": cm["cagr_pct"] - bm["cagr_pct"],
                    "maxdd_pct": cm["maxdd_pct"],
                    "pf": cm["profit_factor"],
                    "worst_12m_pct": cm["worst_12m_pct"],
                }
            )
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_stress.csv", index=False)

    model_metrics.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_model_metrics.csv", index=False)
    usage.to_csv(out_dir / "gmq_tail_risk_alpha_conditional_policy_usage.csv", index=False)

    dataset_stats = {
        "all_n": len(ds),
        "train_n": len(train),
        "holdout_n": len(hold),
    }
    summary_report(out_dir, dataset_stats, model_metrics, alpha_bins, risk_bins, policy_metrics, stress, usage)

    print("=== GMQ Tail-Risk × Alpha Conditional Sizing V1 ===")
    print(f"all={len(ds):,} fit={len(train):,} oos={len(hold):,}")
    print(policy_metrics[["policy","cagr_pct","maxdd_pct","profit_factor","cagr_delta_pp","maxdd_delta_pp"]].to_string(index=False))
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
