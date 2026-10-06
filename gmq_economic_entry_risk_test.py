#!/usr/bin/env python3
"""
Global Momentum Quant — Economic Entry Risk / Adverse Severity Test

Amaç:
    Önceki testin gösterdiği "early adverse" olayını üç ekonomik path sınıfına
    ayırmak:
        1) healthy
        2) temporary_adverse  = early adverse + final trade > 0
        3) harmful_adverse    = early adverse + final trade <= 0

Ana soru:
    Sinyal tarihindeki EX-ANTE feature'lar, temporary adverse ile harmful
    adverse'i birbirinden ayırabiliyor mu?

Metodoloji:
    - Kullanılan veri, önceki BASE Failure/Early-Adverse testinden üretilen
      işlem+feature CSV'sidir.
    - Outcome sütunları model feature'ı olarak KESİNLİKLE kullanılmaz.
    - Train/Holdout ayrımı signal_date/period üzerinden zamansaldır.
    - Model yalnızca TRAIN'de fit edilir.
    - Q75/Q90 eşikleri yalnızca TRAIN tahmin dağılımından alınır.
    - Holdout tamamen sonradan değerlendirilir.
    - Canlı engine/config/state/order/NAV değiştirilmez.
    - Bu çalışma tam portföy A/B backtest değildir; ekonomik seçim teşhisidir.

Çıktılar:
    gmq_economic_entry_risk_raporu.md
    gmq_economic_entry_risk_trades.csv
    gmq_economic_entry_risk_holdout.csv
    gmq_economic_entry_risk_deciles.csv
    gmq_economic_entry_risk_filter_scenarios.csv
    gmq_economic_entry_risk_confusion_matrix.csv
    gmq_economic_entry_risk_market_breakdown.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
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

HOLDOUT_START = pd.Timestamp("2022-01-01")
COST_RT = 0.35
SEED = 20261006

# Fixed, non-optimized clipping bounds carried from the preceding model.
CLIP = {
    "score_pctile": (0.0, 100.0),
    "ret_5d_pct": (-50.0, 50.0),
    "ret_21d_pct": (-100.0, 200.0),
    "ret_63d_pct": (-150.0, 500.0),
    "vol20_ann_pct": (0.0, 300.0),
    "atr14_pct": (0.0, 30.0),
    "gap_pct": (-30.0, 30.0),
    "gap_atr": (0.0, 10.0),
    "market_breadth_5d": (0.0, 1.0),
    "market_breadth_21d": (0.0, 1.0),
    "market_median_1d_pct": (-20.0, 20.0),
    "market_median_21d_pct": (-50.0, 100.0),
    "market_median_63d_pct": (-80.0, 300.0),
    "market_dispersion_1d_pct": (0.0, 20.0),
    "stock_vs_market_21d_pp": (-150.0, 250.0),
    "momentum_extension_z": (-10.0, 10.0),
}


def clip_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df[FEATURES].copy()
    for c in FEATURES:
        lo, hi = CLIP[c]
        x[c] = pd.to_numeric(x[c], errors="coerce").clip(lo, hi)
    return x


def make_labels(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["signal_date"] = pd.to_datetime(x["signal_date"], errors="coerce")
    x["period"] = np.where(x["signal_date"] >= HOLDOUT_START, "holdout", "train")
    x["early_adverse"] = x["early_adverse"].astype(bool)

    # Core mutually exclusive path classes.
    x["path_class"] = np.select(
        [
            ~x["early_adverse"],
            x["early_adverse"] & (x["ret_pct"] > 0),
            x["early_adverse"] & (x["ret_pct"] <= 0),
        ],
        ["healthy", "temporary_adverse", "harmful_adverse"],
        default="unknown",
    )

    x["harmful_adverse"] = x["path_class"].eq("harmful_adverse")
    x["temporary_adverse"] = x["path_class"].eq("temporary_adverse")
    x["tail_harmful"] = x["harmful_adverse"] & (x["ret_pct"] <= -5.0)
    x["ret_negative"] = x["ret_pct"] <= 0
    return x


def fit_multiclass(train: pd.DataFrame, holdout: pd.DataFrame):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Do not use any outcome columns, pred_risk, or market as a predictor.
    xtr = clip_features(train)
    xho = clip_features(holdout)

    # Availability is measured from TRAIN only to avoid holdout-driven feature
    # selection.
    usable = [c for c in FEATURES if xtr[c].notna().mean() >= 0.80]
    xtr = xtr[usable]
    xho = xho[usable]

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )

    ytr = train["path_class"]
    if ytr.nunique() < 3:
        raise RuntimeError(f"Train path_class 3 sınıfı içermiyor: {ytr.unique().tolist()}")

    model.fit(xtr, ytr)
    p_tr = pd.DataFrame(model.predict_proba(xtr), columns=model.classes_, index=train.index)
    p_ho = pd.DataFrame(model.predict_proba(xho), columns=model.classes_, index=holdout.index)

    return model, usable, p_tr, p_ho


def one_vs_rest_metrics(y_true: pd.Series, p: pd.Series) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    y = y_true.astype(int).to_numpy()
    s = p.to_numpy(dtype=float)
    return {
        "n": len(y),
        "prevalence_pct": float(y.mean() * 100.0),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
        "logloss": float(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6), labels=[0, 1])),
    }


def confusion_df(y_true: pd.Series, pred_class: pd.Series) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix

    order = ["healthy", "temporary_adverse", "harmful_adverse"]
    cm = confusion_matrix(y_true, pred_class, labels=order)
    return pd.DataFrame(cm, index=[f"actual_{x}" for x in order], columns=[f"pred_{x}" for x in order])


def fixed_thresholds(train_risk: pd.Series) -> dict:
    a = train_risk.to_numpy(dtype=float)
    return {
        "Q75": float(np.nanquantile(a, 0.75)),
        "Q90": float(np.nanquantile(a, 0.90)),
    }


def economic_stats(all_df: pd.DataFrame, threshold: float) -> dict:
    x = all_df.copy()
    x["skip"] = x["p_harmful"] >= threshold
    kept = x[~x["skip"]]
    skipped = x[x["skip"]]

    all_n = len(x)
    harmful_total = int(x["harmful_adverse"].sum())
    harmful_skipped = int(skipped["harmful_adverse"].sum())
    harmful_loss = float(-x.loc[x["harmful_adverse"] & (x["ret_pct"] < 0), "ret_pct"].sum())
    harmful_loss_skipped = float(-skipped.loc[skipped["harmful_adverse"] & (skipped["ret_pct"] < 0), "ret_pct"].sum())

    skipped_positive = int((skipped["ret_pct"] > 0).sum())
    temporary_skipped = int(skipped["temporary_adverse"].sum())

    def mean(z):
        return float(z["ret_pct"].mean()) if len(z) else np.nan

    # "zero-replacement" is a trade-level diagnostic only:
    # if skipped trades earn 0 instead, expected change vs keeping all is
    # -sum(skipped returns)/N. It is NOT a portfolio backtest.
    selection_delta_pp = float(-skipped["ret_pct"].sum() / all_n) if all_n else np.nan

    return {
        "threshold": float(threshold),
        "skipped_pct": float(len(skipped) / all_n * 100.0) if all_n else np.nan,
        "kept_n": len(kept),
        "skipped_n": len(skipped),
        "kept_win_pct": float((kept["ret_pct"] > 0).mean() * 100.0) if len(kept) else np.nan,
        "skipped_win_pct": float((skipped["ret_pct"] > 0).mean() * 100.0) if len(skipped) else np.nan,
        "kept_mean_ret_pct": mean(kept),
        "skipped_mean_ret_pct": mean(skipped),
        "kept_mean_after_0p35_cost_pct": mean(kept) - COST_RT if len(kept) else np.nan,
        "skipped_positive_pct": float(skipped_positive / len(skipped) * 100.0) if len(skipped) else np.nan,
        "temporary_skipped_pct": float(temporary_skipped / len(skipped) * 100.0) if len(skipped) else np.nan,
        "harmful_total_n": harmful_total,
        "harmful_skipped_n": harmful_skipped,
        "harmful_capture_pct": float(harmful_skipped / harmful_total * 100.0) if harmful_total else np.nan,
        "harmful_loss_total_pp": harmful_loss,
        "harmful_loss_skipped_pp": harmful_loss_skipped,
        "harmful_loss_capture_pct": float(harmful_loss_skipped / harmful_loss * 100.0) if harmful_loss > 0 else np.nan,
        "kept_harmful_pct": float(kept["harmful_adverse"].mean() * 100.0) if len(kept) else np.nan,
        "skipped_harmful_pct": float(skipped["harmful_adverse"].mean() * 100.0) if len(skipped) else np.nan,
        "selection_delta_if_skipped_return_0_pp": selection_delta_pp,
    }


def deciles_from_train(train: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    edges = np.nanquantile(train["p_harmful"], np.linspace(0, 1, 11))
    edges = np.maximum.accumulate(edges)
    edges[-1] = np.inf

    # Collapse duplicated edges safely.
    unique_edges = np.unique(edges)
    if len(unique_edges) < 3:
        holdout = holdout.copy()
        holdout["risk_bin"] = "single_bin"
    else:
        bins = [-np.inf] + list(unique_edges[1:-1]) + [np.inf]
        holdout = holdout.copy()
        holdout["risk_bin"] = pd.cut(
            holdout["p_harmful"],
            bins=bins,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        )

    rows = []
    for b, g in holdout.groupby("risk_bin", observed=True):
        rows.append(
            {
                "risk_bin": str(b),
                "n": len(g),
                "pred_harmful_mean": float(g["p_harmful"].mean()),
                "harmful_rate_pct": float(g["harmful_adverse"].mean() * 100.0),
                "temporary_rate_pct": float(g["temporary_adverse"].mean() * 100.0),
                "win_rate_pct": float((g["ret_pct"] > 0).mean() * 100.0),
                "mean_ret_pct": float(g["ret_pct"].mean()),
                "mean_loss_pct": float(g.loc[g["ret_pct"] <= 0, "ret_pct"].mean()) if (g["ret_pct"] <= 0).any() else np.nan,
                "tail_harmful_rate_pct": float(g["tail_harmful"].mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def market_breakdown(holdout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for market, g in holdout.groupby("market"):
        row = {"market": market, "n": len(g)}
        try:
            row["harmful_auc"] = one_vs_rest_metrics(g["harmful_adverse"], g["p_harmful"])["roc_auc"]
            row["harmful_pr_auc"] = one_vs_rest_metrics(g["harmful_adverse"], g["p_harmful"])["pr_auc"]
        except Exception:
            row["harmful_auc"] = np.nan
            row["harmful_pr_auc"] = np.nan
        row["harmful_rate_pct"] = float(g["harmful_adverse"].mean() * 100.0)
        row["temporary_rate_pct"] = float(g["temporary_adverse"].mean() * 100.0)
        row["win_rate_pct"] = float((g["ret_pct"] > 0).mean() * 100.0)
        row["mean_ret_pct"] = float(g["ret_pct"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def report_text(df, holdout, train_metrics, holdout_metrics, filt, deciles, market, coefs):
    all_mean = float(df["ret_pct"].mean())
    all_win = float((df["ret_pct"] > 0).mean() * 100.0)
    harmful = df[df["harmful_adverse"]]
    temp = df[df["temporary_adverse"]]
    oracle_avoid = float(-harmful.loc[harmful["ret_pct"] < 0, "ret_pct"].sum())

    lines = [
        "# Global Momentum Quant — Economic Entry Risk / Adverse Severity",
        f"_Üretim: {pd.Timestamp.now():%Y-%m-%d %H:%M} · source: prior Early-Adverse trade dataset · RT benchmark: {COST_RT:.2f}%_",
        "",
        "## 1. Amaç",
        "Önceki Early-Adverse Predictor'ın yakaladığı olayları `healthy`, `temporary_adverse` ve `harmful_adverse` olarak ayırıp, sinyal günündeki bilgilerin ekonomik açıdan zararlı adverse path'i öngörüp öngöremediği test edildi.",
        "Bu çalışma canlı sistemi değiştirmez ve tam portföy A/B backtesti değildir.",
        "",
        "## 2. Path dağılımı",
        f"- Toplam işlem: **{len(df)}**",
        f"- Healthy: **{(df.path_class=='healthy').mean()*100:.2f}%**",
        f"- Temporary adverse: **{(df.path_class=='temporary_adverse').mean()*100:.2f}%**",
        f"- Harmful adverse: **{(df.path_class=='harmful_adverse').mean()*100:.2f}%**",
        f"- Tail harmful (<= -5%): **{df.tail_harmful.mean()*100:.2f}%**",
        "",
        f"- BASE tüm işlemler win rate: **{all_win:.2f}%**",
        f"- BASE tüm işlemler ortalama getiri: **{all_mean:.2f}%**",
        f"- Temporary adverse ortalama getiri: **{temp.ret_pct.mean():.2f}%**" if len(temp) else "- Temporary adverse ortalama getiri: —",
        f"- Harmful adverse ortalama getiri: **{harmful.ret_pct.mean():.2f}%**" if len(harmful) else "- Harmful adverse ortalama getiri: —",
        f"- Oracle olarak harmful işlemlerin tamamı önceden atlatılabilse teorik trade-level negatif getiri kaçınması: **{oracle_avoid:.2f} puan**",
        "",
        "## 3. Zararlı adverse predictor doğrulaması",
        f"TRAIN harmful AUC: **{train_metrics['roc_auc']:.4f}**",
        f"HOLDOUT harmful AUC: **{holdout_metrics['roc_auc']:.4f}**",
        f"TRAIN harmful PR-AUC: **{train_metrics['pr_auc']:.4f}**",
        f"HOLDOUT harmful PR-AUC: **{holdout_metrics['pr_auc']:.4f}**",
        f"HOLDOUT Brier: **{holdout_metrics['brier']:.4f}**",
        "",
        "## 4. Holdout risk filtresi",
        "",
        filt.to_markdown(index=False, floatfmt=".4f") if not filt.empty else "—",
        "",
        "Yorum: `selection_delta_if_skipped_return_0_pp` pozitifse, yalnızca trade-level ve sıfır getirili nakit varsayımı altında seçimin ekonomik yönde iyileşme işareti vardır. Bu portföy CAGR testi değildir.",
        "",
        "## 5. Holdout risk binleri",
        "",
        deciles.to_markdown(index=False, floatfmt=".4f") if not deciles.empty else "—",
        "",
        "## 6. BIST / US ayrı doğrulama",
        "",
        market.to_markdown(index=False, floatfmt=".4f") if not market.empty else "—",
        "",
        "## 7. Model katsayıları",
        "",
        coefs.to_markdown(index=False, floatfmt=".4f") if not coefs.empty else "—",
        "",
        "## 8. Karar çerçevesi",
        "Canlı Entry Risk Quality filtresi ancak aynı anda şu kanıtlar oluşursa düşünülebilir:",
        "1. Holdout harmful AUC anlamlı ve train'e göre makul ölçüde korunuyor.",
        "2. Risk yükseldikçe holdout harmful rate belirgin biçimde artıyor.",
        "3. Q75/Q90 ile atlanan grubun ortalama getirisi açık biçimde daha kötü.",
        "4. Pozitif temporary-adverse işlemlerin aşırı büyük kısmı filtreyle atılmıyor.",
        "5. Harmful loss capture ekonomik olarak anlamlı.",
        "6. BIST ve US sonuçları aynı yönde.",
        "7. Daha sonra tam üretim portföyünde bağımsız A/B + 0.35/0.50/0.75/1.00/1.25 maliyet stresi + holdout doğrulaması yapılmalı.",
        "",
        "**Bu test pozitif çıksa bile doğrudan canlı entegrasyon yapılmaz.**",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="research_inputs/gmq_early_adverse_predictor_trades.csv")
    ap.add_argument("--out-dir", default="gmq_economic_entry_risk_output")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = set(FEATURES + ["market","ticker","signal_date","entry_date","ret_pct","early_adverse"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Gerekli sütunlar eksik: {missing}")

    df = make_labels(df)
    train = df[df["period"] == "train"].copy()
    holdout = df[df["period"] == "holdout"].copy()
    if len(train) < 100 or len(holdout) < 100:
        raise RuntimeError(f"Train/Holdout çok küçük: {len(train)}/{len(holdout)}")

    model, used_features, p_tr, p_ho = fit_multiclass(train, holdout)

    # Attach class probabilities.
    for c in ["healthy", "temporary_adverse", "harmful_adverse"]:
        train[f"p_{c}"] = p_tr[c] if c in p_tr.columns else 0.0
        holdout[f"p_{c}"] = p_ho[c] if c in p_ho.columns else 0.0

    train["p_harmful"] = train["p_harmful_adverse"]
    holdout["p_harmful"] = holdout["p_harmful_adverse"]

    pred_tr = model.predict(clip_features(train)[used_features])
    pred_ho = model.predict(clip_features(holdout)[used_features])
    cm = confusion_df(train["path_class"], pred_tr)
    cm_hold = confusion_df(holdout["path_class"], pred_ho)

    tr_metrics = one_vs_rest_metrics(train["harmful_adverse"], train["p_harmful"])
    ho_metrics = one_vs_rest_metrics(holdout["harmful_adverse"], holdout["p_harmful"])

    thresholds = fixed_thresholds(train["p_harmful"])
    filt_rows = []
    for name, th in thresholds.items():
        r = economic_stats(holdout, th)
        r["scenario"] = name
        filt_rows.append(r)
    filt = pd.DataFrame(filt_rows)

    deciles = deciles_from_train(train, holdout)
    market = market_breakdown(holdout)

    # Coefficients, preserving fixed feature list/order.
    coef = model.named_steps["logit"].coef_
    classes = list(model.named_steps["logit"].classes_)
    coef_rows = []
    for i, cls in enumerate(classes):
        for f, v in zip(used_features, coef[i]):
            coef_rows.append({
                "class": cls,
                "feature": f,
                "coefficient_z": float(v),
                "abs_coefficient": abs(float(v)),
            })
    coefs = pd.DataFrame(coef_rows).sort_values(
        ["class","abs_coefficient"], ascending=[True,False]
    )

    # Save trade-level output; no outcome leakage into model features.
    keep_cols = [
        "market","ticker","signal_date","entry_date","exit_date","ret_pct",
        "early_adverse","early_mae_pct","early_mfe_pct","early_mae_atr",
        "path_class","harmful_adverse","temporary_adverse","tail_harmful",
        "p_healthy","p_temporary_adverse","p_harmful_adverse","p_harmful","period",
    ]
    for c in keep_cols:
        if c not in df.columns and c not in holdout.columns:
            # handled below for combined output
            pass
    all_out = pd.concat([train, holdout], ignore_index=True)
    export_cols = [c for c in keep_cols if c in all_out.columns]
    all_out[export_cols].to_csv(out_dir/"gmq_economic_entry_risk_trades.csv", index=False)
    holdout[export_cols].to_csv(out_dir/"gmq_economic_entry_risk_holdout.csv", index=False)
    deciles.to_csv(out_dir/"gmq_economic_entry_risk_deciles.csv", index=False)
    filt.to_csv(out_dir/"gmq_economic_entry_risk_filter_scenarios.csv", index=False)

    cm_long = cm.copy()
    cm_long["sample"] = "train"
    cmh_long = cm_hold.copy()
    cmh_long["sample"] = "holdout"
    pd.concat([cm_long.reset_index(names="actual_class"), cmh_long.reset_index(names="actual_class")], ignore_index=True).to_csv(
        out_dir/"gmq_economic_entry_risk_confusion_matrix.csv", index=False
    )
    market.to_csv(out_dir/"gmq_economic_entry_risk_market_breakdown.csv", index=False)

    report = report_text(
        df, holdout, tr_metrics, ho_metrics, filt, deciles, market, coefs
    )
    (out_dir/"gmq_economic_entry_risk_raporu.md").write_text(report, encoding="utf-8")

    print("=== ECONOMIC ENTRY RISK / ADVERSE SEVERITY ===")
    print(f"Trades={len(df)} | Train={len(train)} | Holdout={len(holdout)}")
    print(f"Healthy={((df.path_class=='healthy').mean()*100):.2f}% | Temporary={((df.path_class=='temporary_adverse').mean()*100):.2f}% | Harmful={((df.path_class=='harmful_adverse').mean()*100):.2f}%")
    print(f"Train harmful AUC={tr_metrics['roc_auc']:.4f} | Holdout harmful AUC={ho_metrics['roc_auc']:.4f}")
    print(f"Train harmful PR-AUC={tr_metrics['pr_auc']:.4f} | Holdout harmful PR-AUC={ho_metrics['pr_auc']:.4f}")
    print(f"Artifacts={out_dir}")
    print("Live config/state/order/NAV/trades were not modified.")


if __name__ == "__main__":
    main()
