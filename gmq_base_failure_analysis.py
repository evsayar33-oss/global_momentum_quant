#!/usr/bin/env python3
"""
Global Momentum Quant — BASE Failure Analysis

Amaç
----
Mevcut BASE stratejisini DEĞİŞTİRMEDEN, gerçek üretim motoru üzerinden
BASE kayıplarının neden tekrar ettiğini teşhis etmek.

Üretim mekanikleri korunur:
- data.get_panel / data.to_wide
- engine.MarketData
- portfolio.process_day
- mevcut risk parity / 4 tranche / 5 günlük stagger / 21 günlük refresh
- mevcut maliyet / nakit faizi / felaket stopu

Bu araştırma canlı state/order/NAV/trade dosyalarını yazmaz.

Çıktılar
--------
1) gmq_base_failure_analysis_raporu.md
2) gmq_base_failure_analysis_trades.csv
3) gmq_base_failure_analysis_baskets.csv
4) gmq_base_failure_analysis_patterns.csv
5) gmq_base_failure_analysis_buckets.csv

Temel prensip
-------------
Buradaki eşikler bir optimizer değildir. Teşhis etiketleri önceden sabitlenmiştir.
Amaç "hangi koşullarda BASE daha sık kaybediyor?" sorusunu cevaplamaktır.
Hiçbir bulgu otomatik olarak canlı sisteme uygulanmaz.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORT_FILE = ROOT / "gmq_base_failure_analysis_raporu.md"
TRADES_FILE = ROOT / "gmq_base_failure_analysis_trades.csv"
BASKETS_FILE = ROOT / "gmq_base_failure_analysis_baskets.csv"
PATTERNS_FILE = ROOT / "gmq_base_failure_analysis_patterns.csv"
BUCKETS_FILE = ROOT / "gmq_base_failure_analysis_buckets.csv"

DEFAULT_YEARS = 14
DEFAULT_COST_RT = 0.35
WARMUP_DAYS = 260
ATR_WINDOW = 14
VOL_WINDOW = 20
MARKET_WINDOW = 63

# Diagnostic-only thresholds. Do not tune these on the backtest.
LATE_EXTENSION_Z = 1.50
GAP_ATR_THRESHOLD = 1.00
RANK_DECAY_PP = -15.0
STRESS_BREADTH = 0.35
EARLY_MAE_ATR = -1.00


@dataclass
class Scenario:
    bist_md: object
    us_md: object
    daily: pd.Series
    state: dict
    trades: pd.DataFrame


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _safe_float(x, default=np.nan):
    try:
        y = float(x)
        return y if np.isfinite(y) else default
    except Exception:
        return default


def _first_present(row: dict, keys: tuple[str, ...], default=np.nan):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _normalize_date(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    try:
        return pd.Timestamp(x).tz_localize(None).normalize()
    except TypeError:
        try:
            return pd.Timestamp(x).normalize()
        except Exception:
            return pd.NaT
    except Exception:
        return pd.NaT


def _date_before(index: pd.DatetimeIndex, d: pd.Timestamp):
    if pd.isna(d) or len(index) < 2:
        return pd.NaT
    pos = index.searchsorted(d, side="left")
    if pos <= 0:
        return pd.NaT
    return index[pos - 1]


def _date_after(index: pd.DatetimeIndex, d: pd.Timestamp, n: int = 1):
    if pd.isna(d):
        return pd.NaT
    pos = index.searchsorted(d, side="left")
    pos2 = pos + n
    if pos2 >= len(index):
        return pd.NaT
    return index[pos2]


def _rolling_atr(high: pd.Series, low: pd.Series, close: pd.Series, window=ATR_WINDOW):
    prev = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev).abs(),
        (low - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def _market_features(W: dict[str, pd.DataFrame]):
    c = W["c"].copy()
    r = c.pct_change(fill_method=None)
    cs_med_1d = r.median(axis=1, skipna=True)
    cs_breadth_1d = (r > 0).mean(axis=1, skipna=True)
    cs_disp_1d = r.std(axis=1, skipna=True)

    ret5 = c / c.shift(5) - 1.0
    ret21 = c / c.shift(21) - 1.0
    ret63 = c / c.shift(63) - 1.0

    return {
        "median_1d": cs_med_1d,
        "breadth_1d": cs_breadth_1d,
        "dispersion_1d": cs_disp_1d,
        "breadth_5d": (ret5 > 0).mean(axis=1, skipna=True),
        "breadth_21d": (ret21 > 0).mean(axis=1, skipna=True),
        "median_5d": ret5.median(axis=1, skipna=True),
        "median_21d": ret21.median(axis=1, skipna=True),
        "median_63d": ret63.median(axis=1, skipna=True),
        "dispersion_rolling": cs_disp_1d.rolling(MARKET_WINDOW, min_periods=20).median(),
    }


def _score_frame(md, index: pd.DatetimeIndex, columns: pd.Index):
    arr = np.asarray(md.score, dtype=float)
    if arr.shape != (len(index), len(columns)):
        raise RuntimeError(f"MarketData score boyutu beklenenden farklı: {arr.shape}")
    return pd.DataFrame(arr, index=index, columns=columns)


def _universe_frame(md, index: pd.DatetimeIndex, columns: pd.Index):
    arr = np.asarray(md.U, dtype=bool)
    if arr.shape != (len(index), len(columns)):
        raise RuntimeError(f"MarketData U boyutu beklenenden farklı: {arr.shape}")
    return pd.DataFrame(arr, index=index, columns=columns)


def _score_percentile(score_row: pd.Series, ticker: str):
    if ticker not in score_row.index:
        return np.nan
    x = score_row.replace([np.inf, -np.inf], np.nan).dropna()
    if ticker not in x.index or len(x) < 5:
        return np.nan
    return float(x.rank(pct=True, method="average").get(ticker, np.nan) * 100.0)


def _extract_market_trades(state: dict):
    rows = []
    for market in ("bist", "us"):
        mstate = state.get("markets", {}).get(market, {})
        for tr in mstate.get("trades", []) or []:
            row = dict(tr)
            row["market"] = market
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _normalize_trades(raw: pd.DataFrame, market_data: dict, W: dict[str, dict[str, pd.DataFrame]]):
    if raw.empty:
        return raw

    out_rows = []
    for src in raw.to_dict("records"):
        market = str(src.get("market", "")).lower()
        if market not in market_data:
            continue
        Wm = W[market]
        idx = pd.DatetimeIndex(Wm["c"].index)
        ticker = str(_first_present(src, ("ticker", "symbol", "asset", "code"), "")).strip()
        entry_date = _normalize_date(_first_present(src, ("entry_date", "open_date", "start_date")))
        exit_date = _normalize_date(_first_present(src, ("exit_date", "close_date", "end_date")))
        if pd.isna(entry_date) or pd.isna(exit_date) or not ticker:
            continue

        signal_date = _normalize_date(_first_present(src, ("signal_date", "signal_dt")))
        if pd.isna(signal_date):
            signal_date = _date_before(idx, entry_date)

        entry_px = _safe_float(_first_present(src, ("entry_px", "entry_price", "entry")))
        exit_px = _safe_float(_first_present(src, ("exit_px", "exit_price", "exit")))
        ret_pct = _safe_float(_first_present(src, ("ret_pct", "net_ret_pct", "pnl_pct", "return_pct")))
        if not np.isfinite(ret_pct) and np.isfinite(entry_px) and entry_px > 0 and np.isfinite(exit_px):
            ret_pct = (exit_px / entry_px - 1.0) * 100.0

        row = {
            "market": market,
            "ticker": ticker,
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "ret_pct": ret_pct,
        }
        # Keep original fields that may be useful for audit.
        for k, v in src.items():
            if k not in row and k not in {"market"}:
                row[f"raw_{k}"] = v
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def _path_metrics(Wm: dict[str, pd.DataFrame], ticker: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp, entry_px: float):
    c = Wm["c"]
    h = Wm["h"]
    l = Wm["l"]
    idx = pd.DatetimeIndex(c.index)
    if ticker not in c.columns or pd.isna(entry_date) or pd.isna(exit_date):
        return {}
    sidx = idx.searchsorted(entry_date, side="left")
    eidx = idx.searchsorted(exit_date, side="right") - 1
    if sidx >= len(idx) or eidx < sidx or not np.isfinite(entry_px) or entry_px <= 0:
        return {}
    sub_h = pd.to_numeric(h[ticker].iloc[sidx:eidx + 1], errors="coerce")
    sub_l = pd.to_numeric(l[ticker].iloc[sidx:eidx + 1], errors="coerce")
    sub_c = pd.to_numeric(c[ticker].iloc[sidx:eidx + 1], errors="coerce")
    mfe = (sub_h / entry_px - 1.0).max(skipna=True) * 100.0
    mae = (sub_l / entry_px - 1.0).min(skipna=True) * 100.0
    mfe_i = int(np.nanargmax((sub_h.to_numpy(dtype=float) / entry_px - 1.0))) if sub_h.notna().any() else -1
    mae_i = int(np.nanargmin((sub_l.to_numpy(dtype=float) / entry_px - 1.0))) if sub_l.notna().any() else -1
    early_n = min(4, len(sub_h))
    early_h = sub_h.iloc[:early_n]
    early_l = sub_l.iloc[:early_n]
    early_mfe = ((early_h / entry_px - 1.0).max(skipna=True) * 100.0) if len(early_h) else np.nan
    early_mae = ((early_l / entry_px - 1.0).min(skipna=True) * 100.0) if len(early_l) else np.nan
    return {
        "mfe_pct": _safe_float(mfe),
        "mae_pct": _safe_float(mae),
        "mfe_bar": mfe_i,
        "mae_bar": mae_i,
        "early_mfe_pct": _safe_float(early_mfe),
        "early_mae_pct": _safe_float(early_mae),
        "entry_to_close_3d_pct": _safe_float((sub_c.iloc[min(3, len(sub_c)-1)] / entry_px - 1.0) * 100.0) if len(sub_c) else np.nan,
        "entry_to_close_5d_pct": _safe_float((sub_c.iloc[min(5, len(sub_c)-1)] / entry_px - 1.0) * 100.0) if len(sub_c) else np.nan,
    }


def enrich_trades(trades: pd.DataFrame, market_data: dict, W: dict[str, dict[str, pd.DataFrame]]):
    if trades.empty:
        return trades

    out = []
    feature_cache = {}
    for market, md in market_data.items():
        # MarketData removes low-coverage days (coverage < 30%) before it
        # calculates score/U. Therefore md.score and md.U are indexed by
        # md.dates, not necessarily by the raw W index. The previous version
        # used W.index here, which produced a deterministic shape mismatch on
        # real Yahoo data (for example (3582, 624)).
        idx = pd.DatetimeIndex(md.dates)
        cols = pd.Index(md.tickers)
        Wm_aligned = {
            k: W[market][k].reindex(index=idx, columns=cols)
            for k in ("o", "h", "l", "c", "v")
        }
        feature_cache[market] = {
            "idx": idx,
            "score": _score_frame(md, idx, cols),
            "U": _universe_frame(md, idx, cols),
            "market": _market_features(Wm_aligned),
        }

    for r in trades.to_dict("records"):
        market = r["market"]
        ticker = r["ticker"]
        Wm = W[market]
        idx = feature_cache[market]["idx"]
        score = feature_cache[market]["score"]
        U = feature_cache[market]["U"]
        mf = feature_cache[market]["market"]
        c = Wm["c"][ticker] if ticker in Wm["c"].columns else pd.Series(dtype=float)
        h = Wm["h"][ticker] if ticker in Wm["h"].columns else pd.Series(dtype=float)
        l = Wm["l"][ticker] if ticker in Wm["l"].columns else pd.Series(dtype=float)
        atr = _rolling_atr(h, l, c)
        ret1 = c.pct_change(fill_method=None)
        ret5 = c / c.shift(5) - 1.0
        ret21 = c / c.shift(21) - 1.0
        ret63 = c / c.shift(63) - 1.0
        vol20 = ret1.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std() * math.sqrt(252.0)

        sd = r["signal_date"]
        ed = r["entry_date"]
        signal_ix = idx.searchsorted(sd, side="left") if pd.notna(sd) else -1
        entry_ix = idx.searchsorted(ed, side="left") if pd.notna(ed) else -1
        if signal_ix >= len(idx):
            signal_ix = -1
        if entry_ix >= len(idx):
            entry_ix = -1

        def at(series, date, default=np.nan):
            try:
                return _safe_float(series.loc[date], default)
            except Exception:
                return default

        signal_close = at(c, sd)
        prev_date = _date_before(idx, ed)
        prev_close = at(c, prev_date)
        entry_open = _safe_float(r.get("entry_px"))
        if not np.isfinite(entry_open) and ticker in Wm["o"].columns:
            entry_open = at(Wm["o"][ticker], ed)

        score_sig = _score_percentile(score.loc[sd] if sd in score.index else pd.Series(dtype=float), ticker)
        score_entry = _score_percentile(score.loc[ed] if ed in score.index else pd.Series(dtype=float), ticker)
        rank_decay = score_entry - score_sig if np.isfinite(score_entry) and np.isfinite(score_sig) else np.nan

        gap_pct = (entry_open / prev_close - 1.0) * 100.0 if np.isfinite(entry_open) and np.isfinite(prev_close) and prev_close > 0 else np.nan
        atr_sig = at(atr, sd)
        atr_pct = atr_sig / signal_close * 100.0 if np.isfinite(atr_sig) and np.isfinite(signal_close) and signal_close > 0 else np.nan
        gap_atr = (abs(gap_pct) / atr_pct) if np.isfinite(gap_pct) and np.isfinite(atr_pct) and atr_pct > 0 else np.nan
        vol20_pct = at(vol20, sd, np.nan) * 100.0
        ret5_pct = at(ret5, sd, np.nan) * 100.0
        extension_z = (ret5_pct / (vol20_pct / math.sqrt(252.0) * math.sqrt(5.0))) if np.isfinite(vol20_pct) and vol20_pct > 0 and np.isfinite(ret5_pct) else np.nan

        # Universe-relative market state at signal date.
        market_breadth21 = _safe_float(mf["breadth_21d"].get(sd, np.nan))
        market_breadth5 = _safe_float(mf["breadth_5d"].get(sd, np.nan))
        market_med1d = _safe_float(mf["median_1d"].get(sd, np.nan)) * 100.0
        market_med21 = _safe_float(mf["median_21d"].get(sd, np.nan)) * 100.0
        market_med63 = _safe_float(mf["median_63d"].get(sd, np.nan)) * 100.0
        market_disp = _safe_float(mf["dispersion_1d"].get(sd, np.nan)) * 100.0

        stock21 = at(ret21, sd, np.nan) * 100.0
        stock63 = at(ret63, sd, np.nan) * 100.0
        rel21 = stock21 - market_med21 if np.isfinite(stock21) and np.isfinite(market_med21) else np.nan

        p = _path_metrics(Wm, ticker, r["entry_date"], r["exit_date"], entry_open)
        mfe_atr = p.get("mfe_pct", np.nan) / atr_pct if np.isfinite(p.get("mfe_pct", np.nan)) and np.isfinite(atr_pct) and atr_pct > 0 else np.nan
        mae_atr = p.get("mae_pct", np.nan) / atr_pct if np.isfinite(p.get("mae_pct", np.nan)) and np.isfinite(atr_pct) and atr_pct > 0 else np.nan

        # Diagnostic labels: ex-ante labels are separated from path/outcome labels.
        late_extension = bool(np.isfinite(extension_z) and extension_z >= LATE_EXTENSION_Z)
        gap_risk = bool(np.isfinite(gap_atr) and gap_atr >= GAP_ATR_THRESHOLD and gap_pct > 0)
        rank_decay_flag = bool(np.isfinite(rank_decay) and rank_decay <= RANK_DECAY_PP)
        market_stress = bool(np.isfinite(market_breadth21) and market_breadth21 <= STRESS_BREADTH)
        trend_fade = bool(np.isfinite(stock21) and stock21 > 0 and np.isfinite(ret5_pct) and ret5_pct < 0)
        early_mae_atr = p.get("early_mae_pct", np.nan) / atr_pct if np.isfinite(p.get("early_mae_pct", np.nan)) and np.isfinite(atr_pct) and atr_pct > 0 else np.nan
        early_adverse = bool(np.isfinite(early_mae_atr) and early_mae_atr <= EARLY_MAE_ATR)

        rr = dict(r)
        rr.update({
            "signal_score_pctile": score_sig,
            "entry_score_pctile": score_entry,
            "score_decay_pp": rank_decay,
            "ret_5d_pct": ret5_pct,
            "ret_21d_pct": stock21,
            "ret_63d_pct": stock63,
            "vol20_ann_pct": vol20_pct,
            "atr14_pct": atr_pct,
            "gap_pct": gap_pct,
            "gap_atr": gap_atr,
            "market_breadth_5d": market_breadth5,
            "market_breadth_21d": market_breadth21,
            "market_median_1d_pct": market_med1d,
            "market_median_21d_pct": market_med21,
            "market_median_63d_pct": market_med63,
            "market_dispersion_1d_pct": market_disp,
            "stock_vs_market_21d_pp": rel21,
            "momentum_extension_z": extension_z,
            **p,
            "mfe_atr": mfe_atr,
            "mae_atr": mae_atr,
            "early_mae_atr": early_mae_atr,
            "flag_late_extension": late_extension,
            "flag_gap_risk": gap_risk,
            "flag_rank_decay": rank_decay_flag,
            "flag_market_stress": market_stress,
            "flag_trend_fade": trend_fade,
            "flag_early_adverse": early_adverse,
        })
        out.append(rr)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["is_loss"] = df["ret_pct"] < 0
    df["loss_amount_pct"] = np.where(df["ret_pct"] < 0, -df["ret_pct"], 0.0)
    total_loss = float(df["loss_amount_pct"].sum())
    df["loss_contribution_pct"] = (df["loss_amount_pct"] / total_loss * 100.0) if total_loss > 0 else 0.0
    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def basket_table(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame()
    keys = ["market", "signal_date", "entry_date", "exit_date"]
    g = trades.groupby(keys, dropna=False)
    b = g["ret_pct"].agg(["mean", "median", "min", "max", "count"]).reset_index()
    b = b.rename(columns={"mean": "basket_mean_ret_pct", "median": "basket_median_ret_pct", "min": "worst_trade_pct", "max": "best_trade_pct", "count": "n_positions"})
    for flag in ("flag_late_extension", "flag_gap_risk", "flag_rank_decay", "flag_market_stress", "flag_trend_fade"):
        x = g[flag].mean().reset_index(name=flag.replace("flag_", "share_"))
        b = b.merge(x, on=keys, how="left")
    b["basket_loss"] = b["basket_mean_ret_pct"] < 0
    return b.sort_values(["exit_date", "market"]).reset_index(drop=True)


def fixed_buckets(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame()
    specs = [
        ("signal_score_pctile", [0, 75, 85, 90, 95, 100], "Score percentile at signal"),
        ("ret_5d_pct", [-np.inf, -5, 0, 5, 10, np.inf], "5d momentum at signal"),
        ("momentum_extension_z", [-np.inf, -1, 0, 1, 1.5, 2.5, np.inf], "5d extension z"),
        ("gap_atr", [-np.inf, 0.25, 0.50, 1.0, 2.0, np.inf], "Entry gap / ATR"),
        ("market_breadth_21d", [0, 0.25, 0.35, 0.50, 0.65, 1.0], "Market 21d breadth"),
        ("score_decay_pp", [-np.inf, -30, -15, 0, 15, np.inf], "Score decay signal→entry"),
    ]
    rows = []
    for col, bins, label in specs:
        if col not in trades.columns:
            continue
        s = pd.to_numeric(trades[col], errors="coerce")
        cat = pd.cut(s, bins=bins, include_lowest=True, duplicates="drop")
        tmp = trades.copy()
        tmp["bucket"] = cat.astype(str)
        for b, sub in tmp.groupby("bucket", dropna=True):
            if len(sub) == 0:
                continue
            losses = sub.loc[sub["ret_pct"] < 0, "ret_pct"]
            rows.append({
                "feature": label,
                "column": col,
                "bucket": b,
                "n": len(sub),
                "win_pct": float((sub["ret_pct"] > 0).mean() * 100.0),
                "mean_ret_pct": float(sub["ret_pct"].mean()),
                "median_ret_pct": float(sub["ret_pct"].median()),
                "loss_mean_pct": float(losses.mean()) if len(losses) else np.nan,
                "loss_share_pct": float((sub["ret_pct"] < 0).mean() * 100.0),
            })
    return pd.DataFrame(rows)


def patterns_table(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame()
    specs = [
        ("late_extension", "flag_late_extension"),
        ("gap_risk", "flag_gap_risk"),
        ("rank_decay", "flag_rank_decay"),
        ("market_stress", "flag_market_stress"),
        ("trend_fade", "flag_trend_fade"),
        ("early_adverse", "flag_early_adverse"),
    ]
    rows = []
    base_losses = trades.loc[trades["ret_pct"] < 0, "ret_pct"]
    base_loss_mean = float(base_losses.mean()) if len(base_losses) else np.nan
    for name, col in specs:
        sub = trades[trades[col].fillna(False)]
        loser = sub[sub["ret_pct"] < 0]
        overlap = (sub["ret_pct"] < 0).sum()
        total_loss = float(-base_losses.sum()) if len(base_losses) else np.nan
        pattern_loss = float(-loser["ret_pct"].sum()) if len(loser) else 0.0
        rows.append({
            "pattern": name,
            "all_trades_n": len(sub),
            "all_trade_share_pct": float(len(sub) / len(trades) * 100.0),
            "losers_n": int(len(loser)),
            "loser_share_pct": float((loser.shape[0] / len(trades[trades["ret_pct"] < 0]) * 100.0) if len(trades[trades["ret_pct"] < 0]) else 0.0),
            "pattern_win_pct": float((sub["ret_pct"] > 0).mean() * 100.0) if len(sub) else np.nan,
            "pattern_mean_ret_pct": float(sub["ret_pct"].mean()) if len(sub) else np.nan,
            "pattern_loss_mean_pct": float(loser["ret_pct"].mean()) if len(loser) else np.nan,
            "loss_contribution_pct": float(pattern_loss / total_loss * 100.0) if total_loss and np.isfinite(total_loss) else np.nan,
            "loss_mean_vs_base_pct": float((loser["ret_pct"].mean() - base_loss_mean)) if len(loser) and np.isfinite(base_loss_mean) else np.nan,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["loss_contribution_pct", "loser_share_pct"], ascending=False).reset_index(drop=True)


def block_summary(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame()
    dt = pd.to_datetime(trades["exit_date"], errors="coerce")
    years = sorted(dt.dropna().dt.year.unique())
    if not years:
        return pd.DataFrame()
    spans = []
    y0 = years[0]
    y1 = years[-1]
    step = max(1, int(math.ceil((y1 - y0 + 1) / 3)))
    start = y0
    while start <= y1:
        end = min(start + step - 1, y1)
        sub = trades[(dt.dt.year >= start) & (dt.dt.year <= end)]
        if len(sub):
            spans.append({
                "period": f"{start}-{end}",
                "n": len(sub),
                "win_pct": float((sub["ret_pct"] > 0).mean() * 100.0),
                "mean_ret_pct": float(sub["ret_pct"].mean()),
                "median_ret_pct": float(sub["ret_pct"].median()),
                "loss_mean_pct": float(sub.loc[sub["ret_pct"] < 0, "ret_pct"].mean()) if (sub["ret_pct"] < 0).any() else np.nan,
                "late_extension_loss_rate_pct": float((sub.loc[sub["ret_pct"] < 0, "flag_late_extension"].mean() if (sub["ret_pct"] < 0).any() else np.nan) * 100.0),
                "market_stress_loss_rate_pct": float((sub.loc[sub["ret_pct"] < 0, "flag_market_stress"].mean() if (sub["ret_pct"] < 0).any() else np.nan) * 100.0),
            })
        start = end + 1
    return pd.DataFrame(spans)


def top_repeat_tickers(trades: pd.DataFrame, n=20):
    los = trades[trades["ret_pct"] < 0].copy()
    if los.empty:
        return pd.DataFrame()
    g = los.groupby(["market", "ticker"])["ret_pct"].agg(["count", "mean", "sum"]).reset_index()
    g["loss_abs_pct"] = -g["sum"]
    return g.sort_values(["loss_abs_pct", "count"], ascending=False).head(n)


def fmt(x, d=2):
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def markdown_table(df: pd.DataFrame, n=None):
    if df is None or df.empty:
        return "—"
    x = df if n is None else df.head(n)
    try:
        return x.to_markdown(index=False, floatfmt=".2f")
    except Exception:
        return x.to_string(index=False)


def build_report(years, cost, start, end, metrics, patterns, buckets, blocks, tickers, data_stats):
    los = tickers[tickers["ret_pct"] < 0]
    win = tickers[tickers["ret_pct"] > 0]
    total_loss = -float(los["ret_pct"].sum()) if len(los) else 0.0
    largest = los.nsmallest(10, "ret_pct")[["market", "ticker", "signal_date", "entry_date", "exit_date", "ret_pct", "mfe_pct", "mae_pct", "mfe_atr", "mae_atr"]] if len(los) else pd.DataFrame()
    repeats = top_repeat_tickers(tickers, 15)

    top_pattern = patterns.iloc[0]["pattern"] if not patterns.empty else "—"
    top_pattern_loss_share = patterns.iloc[0]["loss_contribution_pct"] if not patterns.empty else np.nan

    lines = [
        "# Global Momentum Quant — BASE Failure Analysis",
        f"_Üretim: {pd.Timestamp.now():%Y-%m-%d %H:%M} · veri: {years} yıl + warm-up · ölçüm: {start.date()} → {end.date()}_",
        "",
        "## 1. Amaç",
        "BASE stratejisi değiştirilmeden, gerçek üretim portföy motorundaki kayıpların tekrarlayan nedenleri teşhis edildi.",
        "Bu çalışma bir optimizer değildir; teşhis etiketleri ve eşikleri önceden sabittir.",
        "",
        "## 2. Ana sonuç",
        f"- Pozisyon sayısı: **{len(tickers)}**",
        f"- Kazanan pozisyon: **{len(win)} / {(len(win) / len(tickers) * 100.0 if len(tickers) else np.nan):.2f}%**",
        f"- Kaybeden pozisyon: **{len(los)} / {(len(los) / len(tickers) * 100.0 if len(tickers) else np.nan):.2f}%**",
        f"- Ortalama işlem getirisi: **{fmt(tickers['ret_pct'].mean())}%**",
        f"- Ortalama kaybeden: **{fmt(los['ret_pct'].mean())}%**",
        f"- Ortalama kazanan: **{fmt(win['ret_pct'].mean())}%**",
        f"- Toplam negatif getiri miktarı (toplam pozisyon getirilerinin mutlak kaybı): **{fmt(total_loss)} puan**",
        f"- En baskın tanısal desen: **{top_pattern}**",
        f"- Bu desenin toplam kayıp katkısı: **{fmt(top_pattern_loss_share)}%**",
        "",
        "## 3. En kötü pozisyonlar",
        "",
        markdown_table(largest),
        "",
        "## 4. Tekrarlayan kayıp desenleri",
        "",
        markdown_table(patterns),
        "",
        "Yorum: `loss_contribution_pct` desen altında gerçekleşen kayıpların toplam kayıplara oranıdır. Desenler birbiriyle çakışabilir; toplamlarının %100 olması beklenmez.",
        "",
        "## 5. Ex-ante koşul bucket analizi",
        "",
        markdown_table(buckets, 80),
        "",
        "Bu bölüm olası bir zamanlama problemi için koşullu win/loss yoğunluklarını gösterir. Bucket eşikleri araştırma sırasında optimize edilmemiştir.",
        "",
        "## 6. Zaman içinde failure drift",
        "",
        markdown_table(blocks),
        "",
        "## 7. Tekrarlayan semboller",
        "",
        markdown_table(repeats),
        "",
        "## 8. Failure taxonomy",
        "",
        "### A. Timing / chase failure",
        "`late_extension` + `gap_risk` birlikte yüksekse BASE doğru trendi seçiyor ancak fiyat zaten fazla genişlemişken pozisyon açıyor olabilir.",
        "",
        "### B. Signal decay",
        "`rank_decay` yüksekse hisse sinyal kapanışından sonraki girişe kadar göreli skorda hızla zayıflıyor olabilir. Bu, sinyalin doğru ama giriş penceresinin gecikmeli olabileceğine işaret eder.",
        "",
        "### C. Regime / market breadth failure",
        "`market_stress` yüksekse kayıpların önemli bölümü tekil hisse hatasından çok ortak piyasa rejimi ile açıklanabilir.",
        "",
        "### D. Immediate adverse path",
        "`early_adverse` yüksekse işlem girişten hemen sonra ters gidiyor. Bu bulgu doğrudan bir filtre olarak kullanılmamalıdır; yalnızca kaybın başlangıç yapısını teşhis eder.",
        "",
        "## 9. Veri ve mimari denetimi",
        f"- BIST evreni: `{data_stats['bist_tickers']}`",
        f"- ABD evreni: `{data_stats['us_tickers']}`",
        f"- BIST veri satırı: `{data_stats['bist_rows']}`",
        f"- ABD veri satırı: `{data_stats['us_rows']}`",
        f"- USD/TRY: `{data_stats['fx_first']}` → `{data_stats['fx_last']}`",
        f"- Maliyet: **{cost:.2f}% RT**",
        "- Gerçek `engine.MarketData` + `portfolio.process_day` mekanikleri kullanıldı.",
        "- Canlı `state/order/NAV/trade` dosyaları yazılmadı; yalnızca araştırma çıktıları üretildi.",
        "- ABD geçmiş S&P üyelik tarihi bulunmuyorsa survivorship bias sınırlaması devam eder.",
        "",
        "## 10. Araştırma kararı",
        "",
        "Bu raporun görevi yeni bir strateji seçmek değil, bir sonraki araştırma hipotezini daraltmaktır.",
        "Canlı sisteme alınabilecek hiçbir değişiklik bu rapordan otomatik türetilmemelidir.",
        "",
        "**Önerilen sonraki araştırma ekseni:** yalnızca raporda tekrar eden ve maliyet/holdout ile ayrıca doğrulanabilecek tek bir failure mechanism üzerinde bağımsız A/B testi.",
        "",
        "**Canlı sistemi değiştirmez.**",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Production run
# ---------------------------------------------------------------------------
def run_project(years: int, cost_rt_pct: float, out_dir: Path):
    # Imports stay lazy so --synthetic can validate the analysis locally.
    import config as C
    import data as DA
    import engine as E
    import portfolio as P

    out_dir.mkdir(parents=True, exist_ok=True)

    panels = {
        "bist": DA.get_panel("bist", years=years, force_full=True),
        "us": DA.get_panel("us", years=years, force_full=True),
    }
    if panels["bist"].empty or panels["us"].empty:
        raise RuntimeError("BIST veya ABD paneli boş geldi.")
    fx = DA.get_fx(years)
    if fx.empty:
        raise RuntimeError("USD/TRY verisi boş geldi.")

    irx = pd.Series(dtype=float)
    try:
        yf = DA._yf()
        irx_raw = yf.download("^IRX", period=f"{years}y", interval="1d", auto_adjust=False, progress=False)["Close"]
        if isinstance(irx_raw, pd.DataFrame):
            irx_raw = irx_raw.iloc[:, 0]
        irx_raw.index = pd.to_datetime(irx_raw.index).tz_localize(None).normalize()
        irx = irx_raw.dropna()
    except Exception as exc:
        print(f"⚠️ ^IRX alınamadı; portfolio fallback kullanılacak: {exc}")

    W = {k: DA.to_wide(v) for k, v in panels.items()}

    # BASE spec: use the exact current candidate path from config.
    bist_spec = C.CANDIDATES["bist"][next(iter(C.CANDIDATES["bist"]))]
    us_spec = C.CANDIDATES["us"][next(iter(C.CANDIDATES["us"]))]

    base_bist = E.MarketData("bist", W["bist"], spec=bist_spec)
    base_us = E.MarketData("us", W["us"], spec=us_spec)

    common_idx = max(WARMUP_DAYS, 0)
    start = max(pd.Timestamp(base_bist.dates[common_idx]), pd.Timestamp(base_us.dates[common_idx]))

    # Clone cost in-memory only by using the engine's existing MarketData values.
    old_fc = {"bist": getattr(base_bist, "fc", None), "us": getattr(base_us, "fc", None)}
    if hasattr(base_bist, "fc"):
        base_bist.fc = float(cost_rt_pct) / 200.0
    if hasattr(base_us, "fc"):
        base_us.fc = float(cost_rt_pct) / 200.0

    P.set_hist_rates(irx)
    rb, _ = P.self_financed_returns(base_bist)
    ru_usd, _ = P.self_financed_returns(base_us)
    ru = P._to_tl_returns(ru_usd, "us", fx)

    def shadow(mk, d):
        s = rb if mk == "bist" else ru
        return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)

    days = sorted(d for d in set(base_bist.dates + base_us.dates) if start <= pd.Timestamp(d))
    state = P.new_state(C.CAPITAL_TL, P.fx_at(fx, start))

    for d in days:
        if d in base_bist.didx:
            P.process_day(state, "bist", base_bist, base_bist.didx[d], fx, shadow)
        if d in base_us.didx:
            P.process_day(state, "us", base_us, base_us.didx[d], fx, shadow)

    # Restore only in-memory objects.
    if hasattr(base_bist, "fc"):
        base_bist.fc = old_fc["bist"]
    if hasattr(base_us, "fc"):
        base_us.fc = old_fc["us"]

    nav = pd.DataFrame(state["pf"]["nav_tl"], columns=["date", "mk", "total_tl", "fx"])
    nav["date"] = pd.to_datetime(nav["date"])
    daily = nav.groupby("date")["total_tl"].last().sort_index()

    raw_trades = _extract_market_trades(state)
    trades = _normalize_trades(raw_trades, {"bist": base_bist, "us": base_us}, W)
    if trades.empty:
        raise RuntimeError("BASE trade log boş; failure analysis üretilemedi.")
    trades = enrich_trades(trades, {"bist": base_bist, "us": base_us}, W)
    baskets = basket_table(trades)
    patterns = patterns_table(trades)
    buckets = fixed_buckets(trades)
    blocks = block_summary(trades)

    data_stats = {
        "bist_tickers": int(panels["bist"]["ticker"].nunique()),
        "us_tickers": int(panels["us"]["ticker"].nunique()),
        "bist_rows": int(len(panels["bist"])),
        "us_rows": int(len(panels["us"])),
        "fx_first": str(fx.index.min().date()),
        "fx_last": str(fx.index.max().date()),
    }

    years_span = max((daily.index[-1] - daily.index[0]).days / 365.25, 0.01)
    cagr = (daily.iloc[-1] / daily.iloc[0]) ** (1.0 / years_span) - 1.0
    dd = (daily / daily.cummax() - 1.0).min()
    gains = float(trades.loc[trades.ret_pct > 0, "ret_pct"].sum())
    losses = float(-trades.loc[trades.ret_pct <= 0, "ret_pct"].sum())
    pf = gains / losses if losses > 0 else np.inf
    metrics = {
        "cagr_pct": cagr * 100.0,
        "maxdd_pct": dd * 100.0,
        "trade_win_pct": float((trades.ret_pct > 0).mean() * 100.0),
        "profit_factor": float(pf),
        "trades": int(len(trades)),
        "start": start,
        "end": daily.index[-1],
    }

    report = build_report(
        years,
        cost_rt_pct,
        start,
        daily.index[-1],
        metrics,
        patterns,
        buckets,
        blocks,
        trades,
        data_stats,
    )

    (out_dir / REPORT_FILE.name).write_text(report, encoding="utf-8")
    trades.to_csv(out_dir / TRADES_FILE.name, index=False)
    baskets.to_csv(out_dir / BASKETS_FILE.name, index=False)
    patterns.to_csv(out_dir / PATTERNS_FILE.name, index=False)
    buckets.to_csv(out_dir / BUCKETS_FILE.name, index=False)

    print("=== BASE FAILURE ANALYSIS ===")
    print(f"Measurement: {metrics['start'].date()} -> {metrics['end'].date()}")
    print(f"CAGR={metrics['cagr_pct']:.2f}% | MaxDD={metrics['maxdd_pct']:.2f}% | PF={metrics['profit_factor']:.2f} | Win={metrics['trade_win_pct']:.2f}%")
    print(f"Trades={metrics['trades']}")
    if not patterns.empty:
        p = patterns.iloc[0]
        print(f"Top diagnostic pattern: {p['pattern']} | loss contribution={p['loss_contribution_pct']:.2f}%")
    print(f"Report: {out_dir / REPORT_FILE.name}")
    print(f"Trades CSV: {out_dir / TRADES_FILE.name}")
    print("Live state/order/NAV files were not written by this research script.")


def run_synthetic(out_dir: Path):
    """Hızlı uçtan uca test: gerçek repo modülleri olmadan analiz kodunu smoke-test eder."""
    rng = np.random.default_rng(20261005)
    dates = pd.bdate_range("2020-01-01", periods=900)
    tickers = [f"T{i:03d}" for i in range(40)]
    base = rng.normal(0.0003, 0.02, size=(len(dates), len(tickers)))
    for j in range(0, len(tickers), 7):
        base[300:320, j] -= 0.03
    px = 100 * np.exp(np.cumsum(base, axis=0))
    c = pd.DataFrame(px, index=dates, columns=tickers)
    o = c.shift(1).fillna(c.iloc[0]) * (1 + rng.normal(0, 0.003, c.shape))
    h = pd.DataFrame(np.maximum(o.to_numpy(), c.to_numpy()) * (1 + rng.uniform(0, 0.01, c.shape)), index=dates, columns=tickers)
    l = pd.DataFrame(np.minimum(o.to_numpy(), c.to_numpy()) * (1 - rng.uniform(0, 0.01, c.shape)), index=dates, columns=tickers)
    v = pd.DataFrame(rng.lognormal(12, 0.2, c.shape), index=dates, columns=tickers)
    score = pd.DataFrame(rng.uniform(0.8, 1.0, c.shape), index=dates, columns=tickers)
    U = pd.DataFrame(True, index=dates, columns=tickers)

    class MD:
        def __init__(self):
            self.dates = list(dates)
            self.tickers = list(tickers)
            self.score = score.to_numpy()
            self.U = U.to_numpy()

    Wm = {"c": c, "o": o, "h": h, "l": l, "v": v}
    fake_raw = []
    for i in range(280, len(dates) - 30, 21):
        sd = dates[i]
        ed = dates[i + 1]
        xd = dates[i + 21]
        for t in tickers[:10]:
            entry = float(o.loc[ed, t])
            exit_ = float(c.loc[xd, t])
            fake_raw.append({"market": "bist", "ticker": t, "signal_date": sd, "entry_date": ed, "exit_date": xd,
                             "entry_px": entry, "exit_px": exit_, "ret_pct": (exit_ / entry - 1) * 100})
    md = MD()
    W = {"bist": Wm, "us": Wm}
    trades = _normalize_trades(pd.DataFrame(fake_raw), {"bist": md}, {"bist": Wm, "us": Wm})
    trades = enrich_trades(trades, {"bist": md}, {"bist": Wm, "us": Wm})
    baskets = basket_table(trades)
    patterns = patterns_table(trades)
    buckets = fixed_buckets(trades)
    blocks = block_summary(trades)
    start = dates[280]
    end = dates[-1]
    data_stats = {"bist_tickers": 40, "us_tickers": 0, "bist_rows": 36000, "us_rows": 0, "fx_first": str(start.date()), "fx_last": str(end.date())}
    report = build_report(1, DEFAULT_COST_RT, start, end,
                          {"cagr_pct": np.nan, "maxdd_pct": np.nan, "trade_win_pct": float((trades.ret_pct > 0).mean()*100), "profit_factor": np.nan, "trades": len(trades), "start": start, "end": end},
                          patterns, buckets, blocks, trades, data_stats)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_FILE.name).write_text(report, encoding="utf-8")
    trades.to_csv(out_dir / TRADES_FILE.name, index=False)
    baskets.to_csv(out_dir / BASKETS_FILE.name, index=False)
    patterns.to_csv(out_dir / PATTERNS_FILE.name, index=False)
    buckets.to_csv(out_dir / BUCKETS_FILE.name, index=False)
    print("Synthetic smoke test PASS")


def main() -> int:
    ap = argparse.ArgumentParser(description="Global Momentum Quant BASE failure analysis")
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST_RT)
    ap.add_argument("--out-dir", default=str(ROOT))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.years < 5 and not args.synthetic:
        raise ValueError("Gerçek proje testi için en az 5 yıl önerilir.")
    if args.cost < 0:
        raise ValueError("Maliyet negatif olamaz.")

    out_dir = Path(args.out_dir).resolve()
    if args.synthetic:
        run_synthetic(out_dir)
    else:
        run_project(args.years, args.cost, out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"❌ HATA: {exc}", file=sys.stderr)
        raise SystemExit(1)
