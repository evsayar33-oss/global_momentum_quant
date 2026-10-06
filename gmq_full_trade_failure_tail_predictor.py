#!/usr/bin/env python3
"""
Global Momentum Quant — Full Trade Failure / Tail Loss Predictor
Research-only. Never modifies production engine/config/state.

Purpose:
    Test whether signal-date (and, separately, entry-open-known) information
    can directly predict full-trade failure and tail losses.

Targets:
    bad_trade   : ret_pct <= 0%
    tail_5      : ret_pct <= -5%
    tail_10     : ret_pct <= -10%
    tail_15     : ret_pct <= -15%
    blind_spot  : ret_pct <= 0% AND early_adverse == False

Feature sets:
    strict_signal : features known by the signal close; excludes entry gap.
    entry_open     : adds gap_pct and gap_atr, which are only available once
                     the next session opens and therefore are NOT strict
                     signal-close features.

The research split is fixed at 2022-01-01:
    TRAIN   < 2022-01-01
    HOLDOUT >= 2022-01-01

Model:
    StandardScaler + median imputation + LogisticRegression(C=0.25,
    class_weight='balanced', max_iter=2000).

Thresholds are NOT optimized on holdout. Hard-filter diagnostics use
train-derived upper-tail risk quantiles (75/80/90/95%) and apply them to
holdout unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPLIT_DATE = pd.Timestamp("2022-01-01")
RT_COST_PCT = 0.35

COMMON_FEATURES = [
    "score_pctile",
    "ret_5d_pct",
    "ret_21d_pct",
    "ret_63d_pct",
    "vol20_ann_pct",
    "atr14_pct",
    "gap_pct",
    "gap_atr",
    "market_breadth_5d",
    "market_breadth_21d",
    "market_median_1d_pct",
    "market_median_21d_pct",
    "market_median_63d_pct",
    "market_dispersion_1d_pct",
    "stock_vs_market_21d_pp",
    "momentum_extension_z",
]

FEATURE_SETS: Dict[str, List[str]] = {
    "strict_signal": [f for f in COMMON_FEATURES if f not in {"gap_pct", "gap_atr"}],
    "entry_open": list(COMMON_FEATURES),
}

TARGETS = {
    "bad_trade": lambda d: (d["ret_pct"] <= 0.0).astype(int),
    "tail_5": lambda d: (d["ret_pct"] <= -5.0).astype(int),
    "tail_10": lambda d: (d["ret_pct"] <= -10.0).astype(int),
    "tail_15": lambda d: (d["ret_pct"] <= -15.0).astype(int),
    "blind_spot": lambda d: ((d["ret_pct"] <= 0.0) & (~d["early_adverse"].astype(bool))).astype(int),
}

THRESHOLD_QUANTILES = (0.75, 0.80, 0.90, 0.95)


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
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


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.4f}"


def safe_auc(y: pd.Series, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan")


def safe_pr_auc(y: pd.Series, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if y.nunique() == 2 else float("nan")


def target_rows(df: pd.DataFrame, target: str) -> pd.Series:
    return TARGETS[target](df).astype(int)


def fit_predict(
    df: pd.DataFrame,
    feature_set: str,
    target: str,
) -> Tuple[pd.DataFrame, Pipeline, Dict[str, float]]:
    features = FEATURE_SETS[feature_set]
    work = df.dropna(subset=["signal_date", "ret_pct"]).copy()
    work["y"] = target_rows(work, target)
    train_mask = work["signal_date"] < SPLIT_DATE
    hold_mask = ~train_mask

    train = work.loc[train_mask].copy()
    hold = work.loc[hold_mask].copy()
    if train.empty or hold.empty:
        raise RuntimeError(f"{feature_set}/{target}: TRAIN or HOLDOUT empty")
    if train["y"].nunique() < 2 or hold["y"].nunique() < 2:
        raise RuntimeError(f"{feature_set}/{target}: target has <2 classes")

    model = make_model()
    model.fit(train[features], train["y"])
    train_p = model.predict_proba(train[features])[:, 1]
    hold_p = model.predict_proba(hold[features])[:, 1]

    train_out = train[["market", "ticker", "signal_date", "entry_date", "exit_date", "ret_pct", "early_adverse", "period"]].copy()
    train_out["sample"] = "train"
    train_out["target"] = target
    train_out["feature_set"] = feature_set
    train_out["y"] = train["y"].to_numpy()
    train_out["pred_risk"] = train_p

    hold_out = hold[["market", "ticker", "signal_date", "entry_date", "exit_date", "ret_pct", "early_adverse", "period"]].copy()
    hold_out["sample"] = "holdout"
    hold_out["target"] = target
    hold_out["feature_set"] = feature_set
    hold_out["y"] = hold["y"].to_numpy()
    hold_out["pred_risk"] = hold_p

    preds = pd.concat([train_out, hold_out], ignore_index=True)

    metrics = {
        "train_n": float(len(train)),
        "holdout_n": float(len(hold)),
        "train_rate_pct": float(train["y"].mean() * 100.0),
        "holdout_rate_pct": float(hold["y"].mean() * 100.0),
        "train_auc": safe_auc(train["y"], train_p),
        "holdout_auc": safe_auc(hold["y"], hold_p),
        "train_pr_auc": safe_pr_auc(train["y"], train_p),
        "holdout_pr_auc": safe_pr_auc(hold["y"], hold_p),
        "holdout_brier": float(brier_score_loss(hold["y"], hold_p)),
    }
    return preds, model, metrics


def decile_table(hold: pd.DataFrame) -> pd.DataFrame:
    out = hold.copy()
    out["risk_decile"] = pd.qcut(out["pred_risk"], q=10, labels=False, duplicates="drop")
    rows = []
    for dec, g in out.groupby("risk_decile", observed=True):
        losses = g.loc[g["ret_pct"] <= 0.0, "ret_pct"]
        rows.append(
            {
                "risk_decile": int(dec),
                "n": int(len(g)),
                "pred_risk_mean": float(g["pred_risk"].mean()),
                "target_rate_pct": float(g["y"].mean() * 100.0),
                "win_rate_pct": float((g["ret_pct"] > 0.0).mean() * 100.0),
                "mean_ret_pct": float(g["ret_pct"].mean()),
                "median_ret_pct": float(g["ret_pct"].median()),
                "mean_loss_pct": float(losses.mean()) if len(losses) else float("nan"),
                "tail5_rate_pct": float((g["ret_pct"] <= -5.0).mean() * 100.0),
                "tail10_rate_pct": float((g["ret_pct"] <= -10.0).mean() * 100.0),
                "tail15_rate_pct": float((g["ret_pct"] <= -15.0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("risk_decile")


def filter_table(train: pd.DataFrame, hold: pd.DataFrame, baseline_mean: float) -> pd.DataFrame:
    rows = []
    total_n = len(hold)
    for q in THRESHOLD_QUANTILES:
        threshold = float(train["pred_risk"].quantile(q))
        skipped = hold[hold["pred_risk"] >= threshold].copy()
        kept = hold[hold["pred_risk"] < threshold].copy()
        if kept.empty:
            continue
        neg_total = hold.loc[hold["y"] == 1, "ret_pct"]
        neg_skip = skipped.loc[skipped["y"] == 1, "ret_pct"]
        neg_loss_total = float(-neg_total.sum()) if len(neg_total) else 0.0
        neg_loss_skip = float(-neg_skip.sum()) if len(neg_skip) else 0.0
        rows.append(
            {
                "train_quantile": q,
                "train_threshold": threshold,
                "holdout_skipped_pct": float(len(skipped) / total_n * 100.0),
                "holdout_kept_n": int(len(kept)),
                "holdout_skipped_n": int(len(skipped)),
                "kept_win_pct": float((kept["ret_pct"] > 0.0).mean() * 100.0),
                "skipped_win_pct": float((skipped["ret_pct"] > 0.0).mean() * 100.0) if len(skipped) else float("nan"),
                "kept_mean_ret_pct": float(kept["ret_pct"].mean()),
                "skipped_mean_ret_pct": float(skipped["ret_pct"].mean()) if len(skipped) else float("nan"),
                "kept_mean_after_0p35_cost_pct": float(kept["ret_pct"].mean() - RT_COST_PCT),
                "selection_delta_if_skipped_return_0_pp": float((len(kept) / total_n) * kept["ret_pct"].mean() - baseline_mean),
                "target_capture_pct": float(skipped["y"].sum() / hold["y"].sum() * 100.0) if hold["y"].sum() else float("nan"),
                "target_rate_kept_pct": float(kept["y"].mean() * 100.0),
                "target_rate_skipped_pct": float(skipped["y"].mean() * 100.0) if len(skipped) else float("nan"),
                "negative_loss_capture_pct": float(neg_loss_skip / neg_loss_total * 100.0) if neg_loss_total > 0 else float("nan"),
                "temporary_adverse_skipped_pct": float((skipped["early_adverse"] & (skipped["ret_pct"] > 0.0)).mean() * 100.0) if len(skipped) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def market_table(hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, g in hold.groupby("market"):
        rows.append(
            {
                "market": market,
                "n": int(len(g)),
                "target_rate_pct": float(g["y"].mean() * 100.0),
                "win_rate_pct": float((g["ret_pct"] > 0.0).mean() * 100.0),
                "mean_ret_pct": float(g["ret_pct"].mean()),
                "median_ret_pct": float(g["ret_pct"].median()),
                "mean_pred_risk": float(g["pred_risk"].mean()),
                "auc": safe_auc(g["y"], g["pred_risk"]),
                "pr_auc": safe_pr_auc(g["y"], g["pred_risk"]),
                "tail5_rate_pct": float((g["ret_pct"] <= -5.0).mean() * 100.0),
                "tail10_rate_pct": float((g["ret_pct"] <= -10.0).mean() * 100.0),
                "tail15_rate_pct": float((g["ret_pct"] <= -15.0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("market")


def period_table(hold: pd.DataFrame) -> pd.DataFrame:
    period_order = ["2013-2017", "2018-2022", "2023-2026"]
    rows = []
    for period, g in hold.groupby("period"):
        rows.append(
            {
                "period": period,
                "n": int(len(g)),
                "target_rate_pct": float(g["y"].mean() * 100.0),
                "win_rate_pct": float((g["ret_pct"] > 0.0).mean() * 100.0),
                "mean_ret_pct": float(g["ret_pct"].mean()),
                "median_ret_pct": float(g["ret_pct"].median()),
                "mean_pred_risk": float(g["pred_risk"].mean()),
                "auc": safe_auc(g["y"], g["pred_risk"]),
                "tail5_rate_pct": float((g["ret_pct"] <= -5.0).mean() * 100.0),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_order"] = out["period"].map({p: i for i, p in enumerate(period_order)}).fillna(99)
        out = out.sort_values(["_order", "period"]).drop(columns="_order")
    return out


def coefficient_table(model: Pipeline, features: Iterable[str], target: str, feature_set: str) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    arr = np.asarray(clf.coef_[0], dtype=float)
    out = pd.DataFrame(
        {
            "feature_set": feature_set,
            "target": target,
            "feature": list(features),
            "standardized_logit_coef": arr,
            "abs_coef": np.abs(arr),
        }
    )
    return out.sort_values("abs_coef", ascending=False)


def blind_spot_diagnostics(hold: pd.DataFrame) -> pd.DataFrame:
    bad = hold[hold["ret_pct"] <= 0.0]
    blind = bad[~bad["early_adverse"].astype(bool)]
    early = bad[bad["early_adverse"].astype(bool)]
    total_abs_loss = float(-bad["ret_pct"].sum()) if len(bad) else 0.0
    rows = [
        {
            "group": "all_negative_trades",
            "n": int(len(bad)),
            "mean_ret_pct": float(bad["ret_pct"].mean()) if len(bad) else float("nan"),
            "abs_loss_pct": total_abs_loss,
            "share_of_negative_loss_pct": 100.0 if total_abs_loss else float("nan"),
            "tail5_rate_pct": float((bad["ret_pct"] <= -5.0).mean() * 100.0) if len(bad) else float("nan"),
        },
        {
            "group": "blind_spot_negative_no_early_adverse",
            "n": int(len(blind)),
            "mean_ret_pct": float(blind["ret_pct"].mean()) if len(blind) else float("nan"),
            "abs_loss_pct": float(-blind["ret_pct"].sum()) if len(blind) else 0.0,
            "share_of_negative_loss_pct": float((-blind["ret_pct"].sum()) / total_abs_loss * 100.0) if total_abs_loss else float("nan"),
            "tail5_rate_pct": float((blind["ret_pct"] <= -5.0).mean() * 100.0) if len(blind) else float("nan"),
        },
        {
            "group": "negative_with_early_adverse",
            "n": int(len(early)),
            "mean_ret_pct": float(early["ret_pct"].mean()) if len(early) else float("nan"),
            "abs_loss_pct": float(-early["ret_pct"].sum()) if len(early) else 0.0,
            "share_of_negative_loss_pct": float((-early["ret_pct"].sum()) / total_abs_loss * 100.0) if total_abs_loss else float("nan"),
            "tail5_rate_pct": float((early["ret_pct"] <= -5.0).mean() * 100.0) if len(early) else float("nan"),
        },
    ]
    return pd.DataFrame(rows)


def summary_md(
    source: Path,
    dataset: pd.DataFrame,
    metric_rows: List[Dict[str, object]],
    blind: pd.DataFrame,
    out_dir: Path,
) -> None:
    mdf = pd.DataFrame(metric_rows)
    lines = [
        "# Global Momentum Quant — Full Trade Failure / Tail Loss Predictor",
        "",
        f"_Üretim: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')} · kaynak: `{source.name}` · RT benchmark: {RT_COST_PCT:.2f}%_",
        "",
        "## 1. Araştırma amacı",
        "Bu çalışma BASE sistemine canlı değişiklik yapmadan, sinyal/entry öncesi bilginin doğrudan tam işlem başarısızlığını ve kuyruk kaybını öngörüp öngöremediğini test eder.",
        "Önceki Early-Adverse yaklaşımının kör noktası olan, `early_adverse=False` iken sonradan negatif kapanan işlemler ayrıca `blind_spot` hedefiyle ayrıştırılır.",
        "",
        "## 2. Sabit tasarım",
        f"- Toplam işlem satırı: **{len(dataset):,}**",
        f"- TRAIN: `< {SPLIT_DATE.date()}` · HOLDOUT: `>= {SPLIT_DATE.date()}`",
        "- Model: `SimpleImputer(median) + StandardScaler + LogisticRegression(C=0.25, class_weight=balanced)`",
        "- Holdout eşikleri: yalnızca TRAIN tahmin dağılımından türetilen %75 / %80 / %90 / %95 üst risk quantile eşikleri",
        "- Ticker/market kimliği modele verilmez; market/ticker yalnızca kırılım analizi için kullanılır.",
        "- `entry_open` feature setindeki `gap_pct` ve `gap_atr`, yalnızca ertesi açılışta bilinen değişkenlerdir; strict signal-close modeli bunları kullanmaz.",
        "- Canlı `config.py`, `engine.py`, `portfolio.py`, `signals.py`, state/order/NAV/trade dosyaları değiştirilmez.",
        "",
        "## 3. Hedefler",
        "- `bad_trade`: `ret_pct <= 0%`",
        "- `tail_5`: `ret_pct <= -5%`",
        "- `tail_10`: `ret_pct <= -10%`",
        "- `tail_15`: `ret_pct <= -15%`",
        "- `blind_spot`: `ret_pct <= 0% AND early_adverse == False` (tanısal araştırma hedefidir; tek başına canlı filtre değildir)",
        "",
        "## 4. Model doğrulama özeti",
        "",
    ]
    if not mdf.empty:
        show = mdf.copy()
        cols = ["feature_set", "target", "train_n", "holdout_n", "train_rate_pct", "holdout_rate_pct", "train_auc", "holdout_auc", "train_pr_auc", "holdout_pr_auc", "holdout_brier"]
        show = show[cols]
        lines.append(show.to_markdown(index=False, floatfmt=".4f"))
    lines += [
        "",
        "## 5. Blind-spot loss decomposition (HOLDOUT)",
        "",
        blind.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 6. Karar kapısı",
        "",
        "Bu araştırma ancak HOLDOUT'ta risk sıralaması belirgin, dönem/market yönü tutarlı ve risk filtresi sıfır-getiri varsayımı altında dahi ekonomik olarak pozitif sinyal verirse bir sonraki aşamada tam portföy A/B testine adaydır.",
        "",
        "Özellikle `bad_trade` ve `tail_*` hedeflerinin birlikte değerlendirilmesi gerekir. Yalnızca AUC artışı canlı entegrasyon gerekçesi değildir.",
        "",
        "### Canlı entegrasyon kararı",
        "**NO LIVE CHANGE — RESEARCH ONLY**",
        "",
        "## 7. Üretilen dosyalar",
        f"- `{out_dir.name}/gmq_full_trade_failure_tail_metrics.csv`",
        f"- `{out_dir.name}/gmq_full_trade_failure_tail_deciles.csv`",
        f"- `{out_dir.name}/gmq_full_trade_failure_tail_filter_scenarios.csv`",
        f"- `{out_dir.name}/gmq_full_trade_failure_tail_market_breakdown.csv`",
        f"- `{out_dir.name}/gmq_full_trade_failure_tail_period_breakdown.csv`",
        f"- `{out_dir.name}/gmq_full_trade_failure_tail_coefficients.csv`",
        f"- `{out_dir.name}/gmq_full_trade_failure_tail_holdout_predictions.csv`",
        f"- `{out_dir.name}/gmq_full_trade_failure_tail_blind_spot.csv`",
    ]
    (out_dir / "gmq_full_trade_failure_tail_raporu.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="research/input/gmq_early_adverse_predictor_trades.csv")
    ap.add_argument("--out", default="research/results_full_trade_failure_tail")
    ap.add_argument("--synthetic", action="store_true", help="CI/smoke için küçük sentetik veri oluşturur")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        rng = np.random.default_rng(42)
        n = 2200
        dates = pd.date_range("2014-01-01", periods=n, freq="B")
        x = rng.normal(size=n)
        df = pd.DataFrame(
            {
                "market": np.where(np.arange(n) % 2 == 0, "bist", "us"),
                "ticker": [f"T{i%20:02d}" for i in range(n)],
                "signal_date": dates,
                "entry_date": dates + pd.Timedelta(days=1),
                "exit_date": dates + pd.Timedelta(days=30),
                "ret_pct": 8.0 * x + rng.normal(0, 8, n) - 1.0,
                "early_adverse": rng.random(n) < 0.41,
                "period": np.where(dates < "2018-01-01", "2013-2017", np.where(dates < "2023-01-01", "2018-2022", "2023-2026")),
            }
        )
        for f in COMMON_FEATURES:
            df[f] = rng.normal(size=n)
        df["score_pctile"] = 50 + 20 * x + rng.normal(0, 10, n)
        source = Path("synthetic")
    else:
        source = Path(args.input)
        if not source.exists():
            raise FileNotFoundError(
                f"Input bulunamadı: {source}. `gmq_early_adverse_predictor_trades.csv` dosyasını "
                "research/input/ altına koyun veya --input ile yol verin."
            )
        df = pd.read_csv(source)

    required = {"signal_date", "ret_pct", "early_adverse", "market", "ticker"} | set(COMMON_FEATURES)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.tz_localize(None)
    df["ret_pct"] = pd.to_numeric(df["ret_pct"], errors="coerce")
    df = df.dropna(subset=["signal_date", "ret_pct"]).sort_values("signal_date").reset_index(drop=True)

    metric_rows: List[Dict[str, object]] = []
    all_deciles = []
    all_filters = []
    all_market = []
    all_period = []
    all_coef = []
    all_holdout = []

    baseline_holdout = df.loc[df["signal_date"] >= SPLIT_DATE, "ret_pct"]
    baseline_mean = float(baseline_holdout.mean())

    for feature_set, features in FEATURE_SETS.items():
        for target in TARGETS:
            preds, model, metrics = fit_predict(df, feature_set, target)
            hold = preds.loc[preds["sample"] == "holdout"].copy()
            train = preds.loc[preds["sample"] == "train"].copy()

            row = {"feature_set": feature_set, "target": target, **metrics}
            metric_rows.append(row)

            dec = decile_table(hold)
            dec.insert(0, "feature_set", feature_set)
            dec.insert(1, "target", target)
            all_deciles.append(dec)

            fil = filter_table(train, hold, baseline_mean)
            fil.insert(0, "feature_set", feature_set)
            fil.insert(1, "target", target)
            all_filters.append(fil)

            mk = market_table(hold)
            mk.insert(0, "feature_set", feature_set)
            mk.insert(1, "target", target)
            all_market.append(mk)

            pr = period_table(hold)
            pr.insert(0, "feature_set", feature_set)
            pr.insert(1, "target", target)
            all_period.append(pr)

            cf = coefficient_table(model, features, target, feature_set)
            all_coef.append(cf)

            hp = hold.copy()
            # Keep all signal-date inputs out of this prediction artifact on purpose.
            all_holdout.append(hp)

    metrics_df = pd.DataFrame(metric_rows)
    deciles_df = pd.concat(all_deciles, ignore_index=True)
    filters_df = pd.concat(all_filters, ignore_index=True)
    market_df = pd.concat(all_market, ignore_index=True)
    period_df = pd.concat(all_period, ignore_index=True)
    coef_df = pd.concat(all_coef, ignore_index=True)
    holdout_df = pd.concat(all_holdout, ignore_index=True)

    # Blind-spot report is computed from the strict signal bad-trade model's holdout,
    # because blind-spot is a diagnostic of the prior failure mechanism.
    strict_bad = holdout_df[(holdout_df["feature_set"] == "strict_signal") & (holdout_df["target"] == "bad_trade")].copy()
    blind_df = blind_spot_diagnostics(strict_bad)

    metrics_df.to_csv(out_dir / "gmq_full_trade_failure_tail_metrics.csv", index=False)
    deciles_df.to_csv(out_dir / "gmq_full_trade_failure_tail_deciles.csv", index=False)
    filters_df.to_csv(out_dir / "gmq_full_trade_failure_tail_filter_scenarios.csv", index=False)
    market_df.to_csv(out_dir / "gmq_full_trade_failure_tail_market_breakdown.csv", index=False)
    period_df.to_csv(out_dir / "gmq_full_trade_failure_tail_period_breakdown.csv", index=False)
    coef_df.to_csv(out_dir / "gmq_full_trade_failure_tail_coefficients.csv", index=False)
    holdout_df.to_csv(out_dir / "gmq_full_trade_failure_tail_holdout_predictions.csv", index=False)
    blind_df.to_csv(out_dir / "gmq_full_trade_failure_tail_blind_spot.csv", index=False)

    summary_md(source, df, metric_rows, blind_df, out_dir)

    # Human-readable console summary.
    print("=== GMQ Full Trade Failure / Tail Loss Predictor ===")
    print(f"rows={len(df):,} train={(df['signal_date'] < SPLIT_DATE).sum():,} holdout={(df['signal_date'] >= SPLIT_DATE).sum():,}")
    for _, r in metrics_df.iterrows():
        print(
            f"{r['feature_set']:13s} {r['target']:10s} "
            f"AUC train={pct(r['train_auc'])} holdout={pct(r['holdout_auc'])} "
            f"PR train={pct(r['train_pr_auc'])} holdout={pct(r['holdout_pr_auc'])} "
            f"Brier={pct(r['holdout_brier'])}"
        )
    print(f"Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
