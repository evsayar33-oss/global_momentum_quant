#!/usr/bin/env python3
"""
Global Momentum Quant — Early-Adverse Predictor / Entry Risk Quality

Amaç
----
BASE stratejisini değiştirmeden, sinyal günündeki (t) bilgilerden ertesi
gün girişten sonraki ilk birkaç seansta "early adverse" path oluşacağını
öngörmenin mümkün olup olmadığını test eder.

KRİTİK HİZALAMA / LEAKAGE KURALI
--------------------------------
- Predictor feature'ları yalnızca SIGNAL DATE (t) verilerinden üretilir.
- Early-adverse outcome yalnızca ENTRY DATE (t+1) sonrası ilk 4 gözlemden
  hesaplanır.
- Model eşikleri/model fit yalnızca TRAIN döneminde yapılır.
- HOLDOUT (2022-01-01 sonrası varsayılan) hiçbir şekilde model fit/tuning
  için kullanılmaz.
- BASE üretim mekanikleri engine.MarketData + portfolio.process_day ile
  korunur.
- Bu çalışma canlı config/state/order/NAV dosyalarını değiştirmez.

Çıktılar
--------
1) gmq_early_adverse_predictor_raporu.md
2) gmq_early_adverse_predictor_trades.csv
3) gmq_early_adverse_predictor_model.csv
4) gmq_early_adverse_predictor_holdout.csv
5) gmq_early_adverse_predictor_buckets.csv
6) gmq_early_adverse_predictor_filter_scenarios.csv
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_YEARS = 14
DEFAULT_COST_RT = 0.35
WARMUP_DAYS = 260
ATR_WINDOW = 14
VOL_WINDOW = 20
HOLDOUT_START = pd.Timestamp("2022-01-01")

EARLY_BARS = 4
EARLY_MAE_ATR_THRESHOLD = -1.0

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

# Fixed, research-stage transforms. These are not optimizer-selected.
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


def sf(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def norm_date(x):
    if x is None:
        return pd.NaT
    try:
        return pd.Timestamp(x).tz_localize(None).normalize()
    except Exception:
        try:
            return pd.Timestamp(x).normalize()
        except Exception:
            return pd.NaT


def before(idx: pd.DatetimeIndex, d):
    d = norm_date(d)
    if pd.isna(d):
        return pd.NaT
    p = idx.searchsorted(d, side="left")
    return idx[p - 1] if p > 0 else pd.NaT


def rolling_atr(high, low, close, w=ATR_WINDOW):
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w).mean()


def score_frame(md, idx, cols):
    arr = np.asarray(md.score, dtype=float)
    if arr.shape != (len(idx), len(cols)):
        raise RuntimeError(f"MarketData score boyutu uyumsuz: {arr.shape} != {(len(idx), len(cols))}")
    return pd.DataFrame(arr, index=idx, columns=cols)


def universe_frame(md, idx, cols):
    arr = np.asarray(md.U, dtype=bool)
    if arr.shape != (len(idx), len(cols)):
        raise RuntimeError(f"MarketData U boyutu uyumsuz: {arr.shape} != {(len(idx), len(cols))}")
    return pd.DataFrame(arr, index=idx, columns=cols)


def score_pctile(score_row, ticker):
    x = score_row.replace([np.inf, -np.inf], np.nan).dropna()
    if ticker not in x.index or len(x) < 5:
        return np.nan
    return float(x.rank(pct=True, method="average")[ticker] * 100.0)


def market_features(W):
    c = W["c"]
    r = c.pct_change(fill_method=None)
    med1 = r.median(axis=1, skipna=True)
    br1 = (r > 0).mean(axis=1, skipna=True)
    disp = r.std(axis=1, skipna=True)
    r5 = c / c.shift(5) - 1.0
    r21 = c / c.shift(21) - 1.0
    r63 = c / c.shift(63) - 1.0
    return {
        "median_1d": med1,
        "breadth_5d": (r5 > 0).mean(axis=1, skipna=True),
        "breadth_21d": (r21 > 0).mean(axis=1, skipna=True),
        "median_5d": r5.median(axis=1, skipna=True),
        "median_21d": r21.median(axis=1, skipna=True),
        "median_63d": r63.median(axis=1, skipna=True),
        "dispersion_1d": disp,
    }


def extract_trades(state):
    rows = []
    for mk in ("bist", "us"):
        for tr in state.get("markets", {}).get(mk, {}).get("trades", []) or []:
            q = dict(tr)
            q["market"] = mk
            rows.append(q)
    return pd.DataFrame(rows)


def normalize_trades(raw, W_by_mk):
    if raw.empty:
        return pd.DataFrame()
    out = []
    for r in raw.to_dict("records"):
        mk = str(r.get("market", "")).lower()
        if mk not in W_by_mk:
            continue
        idx = pd.DatetimeIndex(W_by_mk[mk]["c"].index)
        ticker = str(r.get("ticker", r.get("symbol", ""))).strip()
        ed = norm_date(r.get("entry_date"))
        xd = norm_date(r.get("exit_date"))
        if not ticker or pd.isna(ed) or pd.isna(xd):
            continue
        sd = norm_date(r.get("signal_date"))
        if pd.isna(sd):
            sd = before(idx, ed)
        entry_px = sf(r.get("entry_px", r.get("entry_price")))
        exit_px = sf(r.get("exit_px", r.get("exit_price")))
        ret_pct = sf(r.get("ret_pct"))
        if not np.isfinite(ret_pct) and np.isfinite(entry_px) and entry_px > 0 and np.isfinite(exit_px):
            ret_pct = (exit_px / entry_px - 1.0) * 100.0
        out.append({
            "market": mk, "ticker": ticker, "signal_date": sd,
            "entry_date": ed, "exit_date": xd, "entry_px": entry_px,
            "exit_px": exit_px, "ret_pct": ret_pct,
        })
    return pd.DataFrame(out)


def early_path(W, ticker, entry_date, entry_px):
    c, h, l = W["c"], W["h"], W["l"]
    idx = pd.DatetimeIndex(c.index)
    if ticker not in c.columns or not np.isfinite(entry_px) or entry_px <= 0:
        return {}
    p = idx.searchsorted(norm_date(entry_date), side="left")
    if p >= len(idx):
        return {}
    q = min(p + EARLY_BARS, len(idx))
    hs = pd.to_numeric(h[ticker].iloc[p:q], errors="coerce")
    ls = pd.to_numeric(l[ticker].iloc[p:q], errors="coerce")
    cs = pd.to_numeric(c[ticker].iloc[p:q], errors="coerce")
    if len(hs) == 0:
        return {}
    mae = (ls / entry_px - 1.0).min(skipna=True) * 100.0
    mfe = (hs / entry_px - 1.0).max(skipna=True) * 100.0
    c3 = (cs.iloc[min(2, len(cs)-1)] / entry_px - 1.0) * 100.0
    return {"early_mae_pct": sf(mae), "early_mfe_pct": sf(mfe),
            "early_close_3bar_pct": sf(c3), "early_bars_used": len(cs)}


def build_trade_dataset(trades, md_by_mk, W_by_mk):
    if trades.empty:
        return pd.DataFrame()
    caches = {}
    for mk, md in md_by_mk.items():
        idx = pd.DatetimeIndex(md.dates)
        cols = pd.Index(md.tickers)
        Wa = {k: W_by_mk[mk][k].reindex(index=idx, columns=cols) for k in ("o","h","l","c","v")}
        score = score_frame(md, idx, cols)
        mf = market_features(Wa)
        caches[mk] = {"idx": idx, "score": score, "mf": mf, "W": Wa}

    rows = []
    for r in trades.to_dict("records"):
        mk, t = r["market"], r["ticker"]
        if mk not in caches or t not in caches[mk]["W"]["c"].columns:
            continue
        z = caches[mk]
        idx, W, score, mf = z["idx"], z["W"], z["score"], z["mf"]
        sd = norm_date(r["signal_date"])
        ed = norm_date(r["entry_date"])
        if pd.isna(sd) or pd.isna(ed) or sd not in idx:
            continue

        c, h, l = W["c"][t], W["h"][t], W["l"][t]
        ret1 = c.pct_change(fill_method=None)
        ret5 = c / c.shift(5) - 1.0
        ret21 = c / c.shift(21) - 1.0
        ret63 = c / c.shift(63) - 1.0
        vol20 = ret1.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * math.sqrt(252.0)
        atr = rolling_atr(h, l, c)

        def at(s, d):
            try:
                return sf(s.loc[d])
            except Exception:
                return np.nan

        sig_close = at(c, sd)
        prev_close = at(c, before(idx, ed))
        entry_px = sf(r.get("entry_px"))
        if not np.isfinite(entry_px) and t in W["o"].columns:
            entry_px = at(W["o"][t], ed)

        gap_pct = (entry_px / prev_close - 1.0) * 100.0 if np.isfinite(entry_px) and np.isfinite(prev_close) and prev_close > 0 else np.nan
        atr_sig = at(atr, sd)
        atr_pct = atr_sig / sig_close * 100.0 if np.isfinite(atr_sig) and np.isfinite(sig_close) and sig_close > 0 else np.nan
        gap_atr = abs(gap_pct) / atr_pct if np.isfinite(gap_pct) and np.isfinite(atr_pct) and atr_pct > 0 else np.nan
        v20 = at(vol20, sd) * 100.0
        r5 = at(ret5, sd) * 100.0
        ext = r5 / (v20 / math.sqrt(252.0) * math.sqrt(5.0)) if np.isfinite(v20) and v20 > 0 and np.isfinite(r5) else np.nan

        sc = score_pctile(score.loc[sd], t)
        m21 = sf(mf["median_21d"].get(sd, np.nan)) * 100.0
        stock21 = at(ret21, sd) * 100.0

        path = early_path(W, t, ed, entry_px)
        early_mae_atr = path.get("early_mae_pct", np.nan) / atr_pct if np.isfinite(path.get("early_mae_pct", np.nan)) and np.isfinite(atr_pct) and atr_pct > 0 else np.nan
        y = bool(np.isfinite(early_mae_atr) and early_mae_atr <= EARLY_MAE_ATR_THRESHOLD)

        rows.append({
            **r,
            "score_pctile": sc,
            "ret_5d_pct": r5,
            "ret_21d_pct": at(ret21, sd) * 100.0,
            "ret_63d_pct": at(ret63, sd) * 100.0,
            "vol20_ann_pct": v20,
            "atr14_pct": atr_pct,
            "gap_pct": gap_pct,
            "gap_atr": gap_atr,
            "market_breadth_5d": sf(mf["breadth_5d"].get(sd, np.nan)),
            "market_breadth_21d": sf(mf["breadth_21d"].get(sd, np.nan)),
            "market_median_1d_pct": sf(mf["median_1d"].get(sd, np.nan)) * 100.0,
            "market_median_21d_pct": m21,
            "market_median_63d_pct": sf(mf["median_63d"].get(sd, np.nan)) * 100.0,
            "market_dispersion_1d_pct": sf(mf["dispersion_1d"].get(sd, np.nan)) * 100.0,
            "stock_vs_market_21d_pp": stock21 - m21 if np.isfinite(stock21) and np.isfinite(m21) else np.nan,
            "momentum_extension_z": ext,
            **path,
            "early_mae_atr": early_mae_atr,
            "early_adverse": y,
            "period": "holdout" if norm_date(r["signal_date"]) >= HOLDOUT_START else "train",
        })
    return pd.DataFrame(rows)


def zscore_fit_transform(train, other, cols):
    mu = train[cols].mean()
    sd = train[cols].std(ddof=0).replace(0, np.nan)
    a = (train[cols] - mu) / sd
    b = (other[cols] - mu) / sd
    return a.replace([np.inf, -np.inf], np.nan), b.replace([np.inf, -np.inf], np.nan), mu, sd


def clip_df(df, cols):
    x = df[cols].copy()
    for c in cols:
        lo, hi = CLIP[c]
        x[c] = pd.to_numeric(x[c], errors="coerce").clip(lo, hi)
    return x


def metrics_binary(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]
    out = {"n": int(len(y))}
    if len(np.unique(y)) < 2 or len(y) < 20:
        out.update({"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "logloss": np.nan})
        return out
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["pr_auc"] = float(average_precision_score(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    out["logloss"] = float(log_loss(y, np.clip(p, 1e-6, 1-1e-6)))
    return out


def fixed_risk_thresholds(train_prob):
    q75 = float(np.nanquantile(train_prob, 0.75))
    q90 = float(np.nanquantile(train_prob, 0.90))
    return {"Q75": q75, "Q90": q90}


def economic_filter_stats(df, prob_col, threshold, cost_rt):
    x = df.copy()
    x["skip"] = x[prob_col] >= threshold
    kept = x[~x["skip"]].copy()
    skipped = x[x["skip"]].copy()
    def agg(z):
        if len(z) == 0:
            return {"n": 0, "win_pct": np.nan, "mean_ret_pct": np.nan, "loss_mean_pct": np.nan,
                    "early_adverse_pct": np.nan, "gross_sum_pct": 0.0}
        losses = z.loc[z.ret_pct < 0, "ret_pct"]
        return {
            "n": int(len(z)),
            "win_pct": float((z.ret_pct > 0).mean() * 100.0),
            "mean_ret_pct": float(z.ret_pct.mean()),
            "loss_mean_pct": float(losses.mean()) if len(losses) else np.nan,
            "early_adverse_pct": float(z.early_adverse.mean() * 100.0),
            "gross_sum_pct": float(z.ret_pct.sum()),
        }
    k, s = agg(kept), agg(skipped)
    # Approximate per-position cost impact only for comparison; this is NOT a
    # full portfolio re-simulation and is therefore diagnostic, not production.
    k["net_mean_after_rt_cost_pct"] = k["mean_ret_pct"] - cost_rt
    return {
        "threshold": threshold,
        "skipped_pct": float(x["skip"].mean() * 100.0),
        "kept_n": k["n"], "kept_win_pct": k["win_pct"], "kept_mean_ret_pct": k["mean_ret_pct"],
        "kept_loss_mean_pct": k["loss_mean_pct"], "kept_early_adverse_pct": k["early_adverse_pct"],
        "kept_net_mean_after_rt_cost_pct": k["net_mean_after_rt_cost_pct"],
        "skipped_n": s["n"], "skipped_win_pct": s["win_pct"], "skipped_mean_ret_pct": s["mean_ret_pct"],
        "skipped_early_adverse_pct": s["early_adverse_pct"],
    }


def build_model(train, holdout):
    # Feature eligibility and scaling are learned from TRAIN only.
    cols = list(FEATURES)
    xtr_raw = clip_df(train, cols)
    xho_raw = clip_df(holdout, cols)

    eligible = xtr_raw.notna().mean(axis=0) >= 0.80
    cols = [c for c in cols if bool(eligible.get(c, False))]
    if len(cols) < 4:
        raise RuntimeError(f"Yeterli feature kalmadı: {len(cols)}")

    mu = xtr_raw[cols].median()
    sd = xtr_raw[cols].std(ddof=0)
    # Constant training features carry no predictive information and would
    # create NaN after z-scoring; remove them before model fitting.
    variable = sd.replace(0.0, np.nan).notna()
    cols = [c for c in cols if bool(variable.get(c, False))]
    if len(cols) < 4:
        raise RuntimeError(f"Yeterli değişken feature kalmadı: {len(cols)}")

    mu = xtr_raw[cols].median()
    sd = xtr_raw[cols].std(ddof=0).replace(0.0, np.nan)
    a = (xtr_raw[cols] - mu) / sd
    b = (xho_raw[cols] - mu) / sd

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("logit", LogisticRegression(
            C=0.25, class_weight="balanced", max_iter=2000,
            solver="lbfgs", random_state=20261005
        )),
    ])
    ytr = train["early_adverse"].astype(int).to_numpy()
    model.fit(a, ytr)
    p_tr = model.predict_proba(a)[:, 1]
    p_ho = model.predict_proba(b)[:, 1]

    out = train[["market","ticker","signal_date","entry_date","ret_pct","early_adverse"]].copy()
    out["pred_risk"] = p_tr
    out["sample"] = "train"

    out2 = holdout[["market","ticker","signal_date","entry_date","ret_pct","early_adverse"]].copy()
    out2["pred_risk"] = p_ho
    out2["sample"] = "holdout"
    all_out = pd.concat([out, out2], ignore_index=True)

    coef = model.named_steps["logit"].coef_[0]
    coef_df = pd.DataFrame({"feature": cols, "coefficient_z": coef, "abs_coefficient": np.abs(coef)})
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return model, cols, all_out, coef_df


def fixed_univariate_tests(train, holdout):
    rows = []
    for c in FEATURES:
        tr = train[[c,"early_adverse","ret_pct"]].dropna()
        ho = holdout[[c,"early_adverse","ret_pct"]].dropna()
        if len(tr) < 50 or len(ho) < 50:
            continue
        from sklearn.metrics import roc_auc_score, average_precision_score
        # Direction is chosen only on TRAIN; HOLDOUT only evaluates that direction.
        auc = roc_auc_score(tr.early_adverse.astype(int), tr[c])
        direction = 1.0 if auc >= 0.5 else -1.0
        trp = (tr[c] * direction).to_numpy()
        hop = (ho[c] * direction).to_numpy()
        rows.append({
            "feature": c,
            "train_auc": float(max(auc, 1-auc)),
            "holdout_auc": float(roc_auc_score(ho.early_adverse.astype(int), hop)),
            "train_pr_auc": float(average_precision_score(tr.early_adverse.astype(int), (trp-trp.min())/(trp.max()-trp.min()+1e-9))),
            "holdout_pr_auc": float(average_precision_score(ho.early_adverse.astype(int), (hop-hop.min())/(hop.max()-hop.min()+1e-9))),
            "direction": "higher_risk" if direction > 0 else "lower_risk",
            "train_n": len(tr), "holdout_n": len(ho),
        })
    return pd.DataFrame(rows).sort_values("holdout_auc", ascending=False) if rows else pd.DataFrame()


def report_text(years, cost, df, model_metrics, uni, filt, coefs, data_stats):
    losses = df[df.ret_pct < 0]
    base_early = df.early_adverse.mean() * 100.0
    ho = df[df.period=="holdout"]
    lines = [
        "# Global Momentum Quant — Early-Adverse Predictor / Entry Risk Quality",
        f"_Üretim: {pd.Timestamp.now():%Y-%m-%d %H:%M} · veri: {years} yıl + warm-up · RT cost: {cost:.2f}%_",
        "",
        "## 1. Araştırma amacı",
        "Sinyal günündeki bilgilerden, ertesi gün girişten sonraki ilk 4 gözlemde erken adverse hareket oluşup oluşmayacağının öngörülebilirliği test edildi.",
        "Bu çalışma canlı sistemi değiştirmez ve bir optimizer değildir.",
        "",
        "## 2. BASE temel istatistik",
        f"- İşlem: **{len(df)}**",
        f"- Early-adverse oranı: **{base_early:.2f}%**",
        f"- Win rate: **{(df.ret_pct > 0).mean()*100:.2f}%**",
        f"- Ortalama işlem: **{df.ret_pct.mean():.2f}%**",
        f"- Ortalama kayıp: **{losses.ret_pct.mean():.2f}%**",
        f"- Train: **{(df.period=='train').sum()}** | Holdout: **{(df.period=='holdout').sum()}**",
        "",
        "## 3. Model doğrulama",
        "",
        f"TRAIN ROC-AUC: **{model_metrics.get('train_roc_auc', np.nan):.4f}**",
        f"HOLDOUT ROC-AUC: **{model_metrics.get('holdout_roc_auc', np.nan):.4f}**",
        f"TRAIN PR-AUC: **{model_metrics.get('train_pr_auc', np.nan):.4f}**",
        f"HOLDOUT PR-AUC: **{model_metrics.get('holdout_pr_auc', np.nan):.4f}**",
        f"HOLDOUT Brier: **{model_metrics.get('holdout_brier', np.nan):.4f}**",
        "",
        "Karar ölçütü: holdout performansı belirgin değilse predictor canlı filtre olarak reddedilir.",
        "",
        "## 4. Holdout risk dağılımı",
        "",
    ]
    if not filt.empty:
        lines += [filt.to_markdown(index=False, floatfmt=".4f"), ""]
    lines += [
        "## 5. Tek-değişkenli ön tarama",
        "",
        uni.head(16).to_markdown(index=False, floatfmt=".4f") if not uni.empty else "—",
        "",
        "Bu sıralama yalnızca araştırma hipotezlerini daraltır; tek başına model seçimi değildir.",
        "",
        "## 6. Lojistik model katsayıları",
        "",
        coefs.head(16).to_markdown(index=False, floatfmt=".4f") if not coefs.empty else "—",
        "",
        "## 7. Mimari / veri notları",
        f"- BIST tickers: `{data_stats['bist_tickers']}`",
        f"- US tickers: `{data_stats['us_tickers']}`",
        f"- BIST rows: `{data_stats['bist_rows']}`",
        f"- US rows: `{data_stats['us_rows']}`",
        f"- FX: `{data_stats['fx_first']}` → `{data_stats['fx_last']}`",
        "- Feature'lar yalnızca signal_date t verisidir.",
        "- Outcome yalnızca entry_date sonrası ilk 4 gözlemdir.",
        "- Model fit/treshold seçimi train ile sınırlıdır.",
        "- Canlı state/order/NAV/trades/config değiştirilmez.",
        "",
        "## 8. Araştırma kararı çerçevesi",
        "Aşağıdaki koşullar sağlanmadan canlı Entry Risk Quality filtresi eklenmemelidir:",
        "1. Holdout ROC-AUC/PR-AUC train'e göre anlamlı biçimde korunmalı.",
        "2. Q75/Q90 risk filtresi holdout'ta early-adverse oranını gerçekten düşürmeli.",
        "3. Beklenen işlem getirisi ve maliyet sonrası ortalama getiri bozulmamalı.",
        "4. 0.50/0.75/1.00/1.25% maliyet streslerinde sonuç yönü korunmalı.",
        "5. BIST ve US ayrı incelendiğinde predictor sadece tek bir kol üzerinde çalışıyor olmamalı.",
        "",
        "**Bu test tek başına canlı değişiklik önerisi üretmez.**",
    ]
    return "\n".join(lines) + "\n"


def run_project(years, cost_rt, out_dir):
    import data as DA
    import engine as E
    import portfolio as P
    import config as C

    out_dir.mkdir(parents=True, exist_ok=True)
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

    bist_spec = C.CANDIDATES["bist"][next(iter(C.CANDIDATES["bist"]))]
    us_spec = C.CANDIDATES["us"][next(iter(C.CANDIDATES["us"]))]
    md_b = E.MarketData("bist", W["bist"], spec=bist_spec)
    md_u = E.MarketData("us", W["us"], spec=us_spec)

    start_i = max(WARMUP_DAYS, 0)
    start = max(pd.Timestamp(md_b.dates[start_i]), pd.Timestamp(md_u.dates[start_i]))

    # Keep benchmark cost at 0.35% RT by default; modify only in-memory.
    old_fc = (getattr(md_b, "fc", None), getattr(md_u, "fc", None))
    md_b.fc = cost_rt / 200.0
    md_u.fc = cost_rt / 200.0

    irx = pd.Series(dtype=float)
    try:
        yf = DA._yf()
        raw = yf.download("^IRX", period=f"{years}y", interval="1d", auto_adjust=False, progress=False)["Close"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:,0]
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        irx = raw.dropna()
    except Exception as exc:
        print(f"⚠️ ^IRX alınamadı, portfolio fallback: {exc}")

    P.set_hist_rates(irx)
    rb,_ = P.self_financed_returns(md_b)
    ru_usd,_ = P.self_financed_returns(md_u)
    ru = P._to_tl_returns(ru_usd, "us", fx)

    def shadow(mk,d):
        s = rb if mk == "bist" else ru
        return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)

    state = P.new_state(C.CAPITAL_TL, P.fx_at(fx, start))
    days = sorted(d for d in set(md_b.dates + md_u.dates) if start <= pd.Timestamp(d))
    for d in days:
        if d in md_b.didx:
            P.process_day(state, "bist", md_b, md_b.didx[d], fx, shadow)
        if d in md_u.didx:
            P.process_day(state, "us", md_u, md_u.didx[d], fx, shadow)

    raw_tr = extract_trades(state)
    trades = normalize_trades(raw_tr, W)
    if trades.empty:
        raise RuntimeError("BASE trade log boş.")
    trades = build_trade_dataset(trades, {"bist":md_b,"us":md_u}, W)
    if trades.empty:
        raise RuntimeError("Feature dataset boş.")

    train = trades[trades["period"] == "train"].copy()
    hold = trades[trades["period"] == "holdout"].copy()
    if train.empty or hold.empty:
        raise RuntimeError("Train/Holdout ayrımı boş.")

    model, used_features, preds, coefs = build_model(train, hold)
    pm_tr = preds[preds["sample"] == "train"].copy()
    pm_ho = preds[preds["sample"] == "holdout"].copy()

    mt = metrics_binary(pm_tr["early_adverse"], pm_tr["pred_risk"])
    mh = metrics_binary(pm_ho["early_adverse"], pm_ho["pred_risk"])

    uni = fixed_univariate_tests(train, hold)

    thresholds = fixed_risk_thresholds(pm_tr["pred_risk"].to_numpy())
    filt_rows = []
    for name, th in thresholds.items():
        r = economic_filter_stats(pm_ho, "pred_risk", th, cost_rt)
        r["scenario"] = name
        filt_rows.append(r)
    filt = pd.DataFrame(filt_rows)

    stress_rows = []
    for cst in (0.35,0.50,0.75,1.00,1.25):
        # same HOLDOUT classification; only diagnostic economic net mean changes
        for name, th in thresholds.items():
            r = economic_filter_stats(pm_ho, "pred_risk", th, cst)
            r["cost_rt_pct"] = cst
            r["scenario"] = name
            stress_rows.append(r)
    stress = pd.DataFrame(stress_rows)

    # Annotate full trade dataset with out-of-sample risk only where applicable.
    trades_out = trades.copy()
    trades_out["pred_risk"] = np.nan
    idx_tr = pm_tr.set_index(["market","ticker","signal_date","entry_date"]).pred_risk
    idx_ho = pm_ho.set_index(["market","ticker","signal_date","entry_date"]).pred_risk
    for idxs, val in idx_tr.items():
        mask = pd.Series(True, index=trades_out.index)
        for k, v in zip(["market","ticker","signal_date","entry_date"], idxs):
            mask &= trades_out[k].astype(str) == str(v)
        trades_out.loc[mask,"pred_risk"] = float(val)
    for idxs, val in idx_ho.items():
        mask = pd.Series(True, index=trades_out.index)
        for k, v in zip(["market","ticker","signal_date","entry_date"], idxs):
            mask &= trades_out[k].astype(str) == str(v)
        trades_out.loc[mask,"pred_risk"] = float(val)

    # Restore in-memory cost attributes.
    md_b.fc, md_u.fc = old_fc

    data_stats = {
        "bist_tickers": int(panels["bist"]["ticker"].nunique()),
        "us_tickers": int(panels["us"]["ticker"].nunique()),
        "bist_rows": int(len(panels["bist"])),
        "us_rows": int(len(panels["us"])),
        "fx_first": str(fx.index.min().date()),
        "fx_last": str(fx.index.max().date()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    report = report_text(
        years, cost_rt, trades, {
            "train_roc_auc": mt["roc_auc"], "holdout_roc_auc": mh["roc_auc"],
            "train_pr_auc": mt["pr_auc"], "holdout_pr_auc": mh["pr_auc"],
            "holdout_brier": mh["brier"],
        }, uni, filt, coefs, data_stats
    )
    (out_dir/"gmq_early_adverse_predictor_raporu.md").write_text(report, encoding="utf-8")
    trades_out.to_csv(out_dir/"gmq_early_adverse_predictor_trades.csv", index=False)
    pd.concat([
        pm_tr.assign(sample="train"),
        pm_ho.assign(sample="holdout")
    ], ignore_index=True).to_csv(out_dir/"gmq_early_adverse_predictor_model.csv", index=False)
    pm_ho.to_csv(out_dir/"gmq_early_adverse_predictor_holdout.csv", index=False)
    uni.to_csv(out_dir/"gmq_early_adverse_predictor_buckets.csv", index=False)
    stress.to_csv(out_dir/"gmq_early_adverse_predictor_filter_scenarios.csv", index=False)

    print("=== EARLY-ADVERSE PREDICTOR ===")
    print(f"Trades={len(trades)} | EarlyAdverse={trades["early_adverse"].mean()*100:.2f}%")
    print(f"Train ROC-AUC={mt['roc_auc']:.4f} | Holdout ROC-AUC={mh['roc_auc']:.4f}")
    print(f"Train PR-AUC={mt['pr_auc']:.4f} | Holdout PR-AUC={mh['pr_auc']:.4f}")
    print(f"Artifacts: {out_dir}")
    print("Live config/state/order/NAV/trades were not modified.")


def synthetic(out_dir):
    # Smoke test only. Does not emulate production portfolio performance.
    rng = np.random.default_rng(20261005)
    dates = pd.bdate_range("2012-01-01", periods=3300)
    tks = [f"T{i:03d}" for i in range(80)]
    ret = rng.normal(0.0002, 0.018, (len(dates),len(tks)))
    ret[::31,:10] -= 0.025
    c = 100*np.exp(np.cumsum(ret,axis=0))
    o = c*np.exp(rng.normal(0,0.002,c.shape))
    h = np.maximum(o,c)*(1+rng.random(c.shape)*0.01)
    l = np.minimum(o,c)*(1-rng.random(c.shape)*0.01)
    v = rng.lognormal(12,0.2,c.shape)
    W = {"o":pd.DataFrame(o,index=dates,columns=tks),
         "h":pd.DataFrame(h,index=dates,columns=tks),
         "l":pd.DataFrame(l,index=dates,columns=tks),
         "c":pd.DataFrame(c,index=dates,columns=tks),
         "v":pd.DataFrame(v,index=dates,columns=tks)}
    class MD:
        pass
    md = MD()
    md.dates = list(dates)
    md.tickers = tks
    md.score = rng.random((len(dates), len(tks)))
    md.U = np.ones((len(dates), len(tks)), dtype=bool)
    trades=[]
    for i in range(280,len(dates)-25,21):
        for t in tks[:10]:
            sd=dates[i]; ed=dates[i+1]; xd=dates[i+21]; ep=float(o[ i+1, tks.index(t)]); xp=float(c[i+21,tks.index(t)])
            trades.append({"market":"bist","ticker":t,"signal_date":sd,"entry_date":ed,"exit_date":xd,"entry_px":ep,"exit_px":xp,"ret_pct":(xp/ep-1)*100})
    tr=normalize_trades(pd.DataFrame(trades),{"bist":W})
    ds=build_trade_dataset(tr,{"bist":md},{"bist":W,"us":W})
    train=ds[ds.period=="train"].copy(); hold=ds[ds.period=="holdout"].copy()
    # Synthetic dates extend through 2024+, so both sets exist.
    _,_,preds,coefs=build_model(train,hold)
    assert len(preds)>100
    out_dir.mkdir(parents=True,exist_ok=True)
    (out_dir/"SMOKE_PASS.txt").write_text("Early-Adverse Predictor synthetic smoke test PASS\n")
    print("Synthetic smoke test PASS")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--years",type=int,default=DEFAULT_YEARS)
    ap.add_argument("--cost",type=float,default=DEFAULT_COST_RT)
    ap.add_argument("--out-dir",default="gmq_early_adverse_predictor_output")
    ap.add_argument("--synthetic",action="store_true")
    a=ap.parse_args()
    if a.cost < 0:
        raise ValueError("Maliyet negatif olamaz.")
    if a.years < 8 and not a.synthetic:
        raise ValueError("Gerçek test için en az 8 yıl gerekir.")
    if a.synthetic:
        synthetic(Path(a.out_dir))
    else:
        run_project(a.years,a.cost,Path(a.out_dir))


if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        print(f"❌ HATA: {e}",file=sys.stderr)
        raise SystemExit(1)
