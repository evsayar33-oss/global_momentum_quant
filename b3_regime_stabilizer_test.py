#!/usr/bin/env python3
"""
B3 Regime Stabilizer — bağımsız araştırma / doğrulama testi.

AMAÇ
-----
B2'nin güçlü taraflarını koruyup 2020–2026 dönemindeki zayıflığını azaltmayı
hedefleyen, önceden tanımlanmış tek bir B3 kuralını test eder.

KARŞILAŞTIRMA
-------------
BASE:
    mom_12_1 + hi52 + low_max

B:
    mom_12_1 + hi52 + robust Amihud

B2:
    B2 robust blend + geçmiş B-vs-BASE conviction + Amihud stress

B3:
    B2'nin Amihud ağırlığına ek olarak piyasa trend rejimi kapısı uygulanır.
    Trend zayıfken BASE/low_max tarafına geri çekilir.
    Trend güçlüyken B2'ye daha yakın çalışır.

ÖNEMLİ
------
- engine.py / portfolio.py / signals.py kullanılmaz.
- config.py veya data/ altına yazılmaz.
- B3 parametreleri test verisine bakılarak otomatik optimize edilmez.
- Amaç "en yüksek CAGR" seçmek değil; B2'nin dönemler arası tutarsızlığını
  azaltıp daha sağlam bir risk/getiri profili elde edip etmediğini görmek.
- Sonuç canlı sisteme otomatik uygulanmaz.

B3 KURALI (ÖNCEDEN TANIMLI)
---------------------------
1) Robust Amihud:
   günlük |ret|/(close*volume)
   -> %5/%95 winsorization
   -> log
   -> cross-sectional percentile rank

2) B2 ağırlığı:
   nötr = 0.25
   geçmiş 12 tamamlanmış sepetin B-BASE farkı conviction olarak kullanılır
   mevcut piyasa Amihud stresi conviction ile birlikte ağırlığı değiştirir

3) Trend rejimi:
   - 200 günlük hareketli ortalama üzerindeki likit evren oranı = breadth
   - 63 günlük cross-sectional median momentum
   - trend_score = breadth + momentum bileşenlerinin ortalaması, [-1,+1]
   - trend_gate = 0.35 + 0.65 * ((trend_score + 1)/2)
   - B3 Amihud ağırlığı = B2 ağırlığı * trend_gate
   - son ağırlık [0.00, 0.50] arasında tutulur

Bu kapı geleceğe bakmaz; yalnızca sinyal gününe kadar bilinen fiyatlardan oluşur.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
UNIVERSE_FILE = ROOT / "data" / "universe_bist.json"
REPORT_FILE = ROOT / "b3_regime_stabilizer_raporu.md"
TRADES_FILE = ROOT / "b3_regime_stabilizer_islemler.csv"

DEFAULT_YEARS = 14
DEFAULT_COSTS = [0.35, 0.50, 0.75, 1.00, 1.25]

MAX_MOVE = 0.105
LIQ_PCT = 0.40
HOLD_DAYS = 21
N_PICKS = 10
WARMUP_DAYS = 252
MIN_LIQ_DAYS = 10

REGIME_WINDOW = 126
TREND_WINDOW = 63
TREND_MA = 200
CONVICTION_BASKETS = 12
CONVICTION_SCALE_PCT = 5.0

AMIHUD_MIN_WEIGHT = 0.00
AMIHUD_NEUTRAL_WEIGHT = 0.25
AMIHUD_MAX_WEIGHT = 0.50

SEED = 20261004


@dataclass
class VariantResult:
    name: str
    cost_rt_pct: float
    trades: pd.DataFrame
    portfolio: pd.DataFrame


def _yf():
    try:
        import yfinance as yf
        return yf
    except ImportError as exc:
        raise RuntimeError("yfinance kurulu değil.") from exc


def _norm_ticker(x: object) -> str:
    return str(x).upper().strip()


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"BIST evreni bulunamadı: {UNIVERSE_FILE}")
    with UNIVERSE_FILE.open("r", encoding="utf-8") as f:
        u = json.load(f)
    tickers = sorted({_norm_ticker(x) for x in (u.get("tickers") or []) if str(x).strip()})
    if len(tickers) < 30:
        raise RuntimeError(f"BIST evreni çok küçük: {len(tickers)}")
    return tickers


def _extract_symbol(data: pd.DataFrame, sym: str, single: bool) -> pd.DataFrame | None:
    sub = None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            lv0 = set(map(str, data.columns.get_level_values(0)))
            lv1 = set(map(str, data.columns.get_level_values(1)))
            if sym in lv0:
                sub = data[sym]
            elif sym in lv1:
                sub = data.xs(sym, axis=1, level=1)
        elif single:
            sub = data
    except Exception:
        sub = None
    if sub is None or len(sub) == 0:
        return None
    sub = sub.copy()
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = sub.columns.get_level_values(-1)
    sub.columns = [str(c).lower() for c in sub.columns]
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(sub.columns):
        return None
    sub = sub[["open", "high", "low", "close", "volume"]].reset_index()
    sub = sub.rename(columns={sub.columns[0]: "date"})
    return sub


def download_bist(tickers: list[str], years: int, chunk: int = 50, retries: int = 2) -> pd.DataFrame:
    yf = _yf()
    end = pd.Timestamp.now().normalize()
    start = (end - pd.DateOffset(years=years + 1)).strftime("%Y-%m-%d")
    symbols = [f"{t}.IS" for t in tickers]
    rows: list[pd.DataFrame] = []

    print(f"📥 B3 BIST verisi: {len(symbols)} hisse, yaklaşık {years + 1} yıl")

    for i in range(0, len(symbols), chunk):
        syms = symbols[i:i + chunk]
        data = None

        for attempt in range(retries + 1):
            try:
                data = yf.download(
                    syms,
                    start=start,
                    end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=True,
                )
                if data is not None and not data.empty:
                    break
            except Exception as exc:
                print(f"  yfinance deneme {attempt + 1}: {exc}")
            time.sleep(2 + attempt * 2)

        if data is None or data.empty:
            print(f"  ⚠️ {i + 1}-{i + len(syms)} grubu alınamadı")
            continue

        single = len(syms) == 1
        for raw_t, sym in zip(tickers[i:i + chunk], syms):
            try:
                sub = _extract_symbol(data, sym, single)
                if sub is None or sub.empty:
                    continue
                sub["ticker"] = raw_t
                rows.append(sub)
            except Exception as exc:
                print(f"  ⚠️ {raw_t}: {exc}")

        time.sleep(0.4)

    if not rows:
        raise RuntimeError("Hiç BIST verisi indirilemedi.")

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["close", "open"])
    out = out[(out["close"] > 0) & (out["open"] > 0) & (out["high"] >= out["low"])]
    out = out.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    return out


def make_wide(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for short, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
        x = panel.pivot(index="date", columns="ticker", values=col).sort_index()
        out[short] = x
    cols = out["c"].columns
    return {k: v.reindex(columns=cols) for k, v in out.items()}


def apply_data_filters(W: dict[str, pd.DataFrame]) -> pd.DataFrame:
    o, h, l, c, v = (W[k] for k in ("o", "h", "l", "c", "v"))
    r = c.pct_change(fill_method=None)
    traded_value = (c * v).where((v > 0) & c.notna())
    liq = traded_value.rolling(20, min_periods=MIN_LIQ_DAYS).median()
    valid = c.notna() & o.notna() & (v > 0) & (h >= l) & r.abs().le(MAX_MOVE)
    rank = liq.rank(axis=1, pct=True, method="average")
    return valid & liq.notna() & rank.ge(LIQ_PCT)


def cross_sectional_winsor(x: pd.DataFrame, low: float = 0.05, high: float = 0.95) -> pd.DataFrame:
    lo = x.quantile(low, axis=1)
    hi = x.quantile(high, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def prepare_features(W: dict[str, pd.DataFrame], universe: pd.DataFrame):
    o, h, l, c, v = (W[k] for k in ("o", "h", "l", "c", "v"))
    r = c.pct_change(fill_method=None)

    mom = c.shift(21) / c.shift(252) - 1.0
    hi52 = c / c.rolling(252, min_periods=200).max()
    low_max = -r.rolling(21, min_periods=15).max()

    traded_value = (c * v).where(v > 0)
    raw_am = (r.abs() / traded_value.replace(0, np.nan)).rolling(
        20, min_periods=MIN_LIQ_DAYS
    ).mean()

    # Robust Amihud: günlük cross-sectional winsorization + log + rank.
    am_w = cross_sectional_winsor(raw_am.where(universe))
    robust_am = np.log1p(am_w.clip(lower=0))
    robust_am = robust_am.where(universe)

    def rank(x):
        return x.where(universe).rank(axis=1, pct=True, method="average")

    rm = rank(mom)
    rh = rank(hi52)
    rl = rank(low_max)
    ra = rank(robust_am)

    scores = {
        "BASE": (rm + rh + rl) / 3.0,
        "B": (rm + rh + ra) / 3.0,
        "CORE": (rm + rh) / 2.0,
        "LOW": rl,
        "AM": ra,
    }

    # Piyasa genelinde Amihud stresi.
    market_am = robust_am.where(universe).median(axis=1)
    am_mean = market_am.rolling(REGIME_WINDOW, min_periods=60).mean()
    am_std = market_am.rolling(REGIME_WINDOW, min_periods=60).std()
    stress_z = ((market_am - am_mean) / am_std.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    # Trend rejimi: 200DMA breadth + 63 günlük medyan momentum.
    ma200 = c.rolling(TREND_MA, min_periods=TREND_MA).mean()
    above_ma = (c > ma200).where(universe)
    breadth = above_ma.mean(axis=1)

    median_mom = (c / c.shift(TREND_WINDOW) - 1.0).where(universe).median(axis=1)

    # [-1,+1] normalize edilmiş iki alt sinyal.
    breadth_score = ((breadth - 0.50) / 0.20).clip(-1.0, 1.0)
    momentum_score = (median_mom / 0.10).clip(-1.0, 1.0)
    trend_score = (breadth_score.fillna(0.0) + momentum_score.fillna(0.0)) / 2.0

    regime = pd.DataFrame({
        "market_amihud": market_am,
        "stress_z": stress_z,
        "breadth_above_200dma": breadth,
        "median_63d_momentum": median_mom,
        "trend_score": trend_score,
    })

    return scores, regime


def pick_top(score_row: pd.Series, mask_row: pd.Series) -> list[str]:
    s = score_row.where(mask_row).dropna().sort_values(ascending=False, kind="stable")
    return list(s.index[:N_PICKS])


def rebalance_dates(dates: pd.DatetimeIndex, start: pd.Timestamp) -> list[pd.Timestamp]:
    idx = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    if idx >= len(dates):
        return []
    return list(dates[idx::HOLD_DAYS])


def net_return(entry: float, exit_: float, cost_rt_pct: float) -> float:
    half = cost_rt_pct / 100.0 / 2.0
    e = entry * (1.0 + half)
    x = exit_ * (1.0 - half)
    return x / e - 1.0 if e > 0 else np.nan


def basket_returns(W, picks, entry_i, exit_i, cost_rt_pct):
    dates = W["c"].index
    out = []
    for ticker in picks:
        if ticker not in W["o"].columns:
            continue
        entry = W["o"].iloc[entry_i].get(ticker, np.nan)
        exit_ = W["c"].iloc[exit_i].get(ticker, np.nan)
        if not (pd.notna(entry) and pd.notna(exit_) and entry > 0 and exit_ > 0):
            continue
        nr = net_return(float(entry), float(exit_), cost_rt_pct)
        if np.isfinite(nr):
            out.append({
                "ticker": ticker,
                "entry_date": dates[entry_i],
                "exit_date": dates[exit_i],
                "entry_px": float(entry),
                "exit_px": float(exit_),
                "raw_ret_pct": (float(exit_) / float(entry) - 1.0) * 100.0,
                "net_ret_pct": nr * 100.0,
            })
    return out


def build_base_b_history(W, universe, scores, start, cost):
    dates = list(pd.DatetimeIndex(W["c"].index))
    rows = []
    for signal_date in rebalance_dates(pd.DatetimeIndex(dates), start):
        i = dates.index(signal_date)
        entry_i = i + 1
        exit_i = i + HOLD_DAYS
        if exit_i >= len(dates):
            break
        for name in ("BASE", "B"):
            picks = pick_top(scores[name].loc[signal_date], universe.loc[signal_date])
            if len(picks) < max(5, math.ceil(N_PICKS * 0.7)):
                continue
            tr = basket_returns(W, picks, entry_i, exit_i, cost)
            if len(tr) < max(5, math.ceil(N_PICKS * 0.7)):
                continue
            rows.append({
                "signal_date": signal_date,
                "variant": name,
                "exit_date": dates[exit_i],
                "basket_ret_pct": float(np.mean([x["net_ret_pct"] for x in tr])),
            })
    return pd.DataFrame(rows)


def b2_weight(signal_date, regime, history):
    stress_z = float(regime.loc[signal_date, "stress_z"]) if signal_date in regime.index else 0.0
    stress_signal = float(np.tanh(stress_z))

    prior = history[history["exit_date"] < signal_date].sort_values("exit_date")
    pivot = prior.pivot(index="exit_date", columns="variant", values="basket_ret_pct").dropna()
    if len(pivot):
        excess = pivot["B"] - pivot["BASE"]
        recent = excess.tail(CONVICTION_BASKETS)
        excess_mean = float(recent.mean()) if len(recent) else 0.0
    else:
        excess_mean = 0.0

    conviction = float(np.tanh(excess_mean / CONVICTION_SCALE_PCT))
    w = AMIHUD_NEUTRAL_WEIGHT + 0.25 * stress_signal * conviction
    return float(np.clip(w, AMIHUD_MIN_WEIGHT, AMIHUD_MAX_WEIGHT)), stress_z, excess_mean


def b3_weight(signal_date, regime, history):
    b2w, stress_z, excess_mean = b2_weight(signal_date, regime, history)

    row = regime.loc[signal_date] if signal_date in regime.index else None
    trend_score = float(row["trend_score"]) if row is not None and np.isfinite(row["trend_score"]) else 0.0

    # Önceden tanımlı trend kapısı:
    # trend=-1 -> 0.35; trend=+1 -> 1.00.
    trend_norm = (trend_score + 1.0) / 2.0
    trend_gate = 0.35 + 0.65 * trend_norm

    # Ek olarak çok yüksek piyasa stresinde agresif Amihud ağırlığını yumuşat.
    if stress_z > 2.0:
        stress_gate = 0.70
    elif stress_z > 1.0:
        stress_gate = 0.85
    else:
        stress_gate = 1.00

    w = b2w * trend_gate * stress_gate
    return float(np.clip(w, AMIHUD_MIN_WEIGHT, AMIHUD_MAX_WEIGHT)), stress_z, trend_score, trend_gate, excess_mean


def run_variant(name, W, universe, scores, regime, history, cost, start):
    dates = list(pd.DatetimeIndex(W["c"].index))
    rows = []
    portfolio = []
    capital = 1.0

    for signal_date in rebalance_dates(pd.DatetimeIndex(dates), start):
        i = dates.index(signal_date)
        entry_i = i + 1
        exit_i = i + HOLD_DAYS
        if entry_i >= len(dates) or exit_i >= len(dates):
            break

        am_weight = np.nan
        stress_z = np.nan
        trend_score = np.nan
        trend_gate = np.nan
        excess_mean = np.nan

        if name == "BASE":
            score_row = scores["BASE"].loc[signal_date]
        elif name == "B":
            score_row = scores["B"].loc[signal_date]
        elif name == "B2":
            am_weight, stress_z, excess_mean = b2_weight(signal_date, regime, history)
            blended = (1.0 - am_weight) * scores["LOW"].loc[signal_date] + am_weight * scores["AM"].loc[signal_date]
            score_row = 0.5 * scores["CORE"].loc[signal_date] + 0.5 * blended
        elif name == "B3":
            am_weight, stress_z, trend_score, trend_gate, excess_mean = b3_weight(
                signal_date, regime, history
            )
            blended = (1.0 - am_weight) * scores["LOW"].loc[signal_date] + am_weight * scores["AM"].loc[signal_date]
            score_row = 0.5 * scores["CORE"].loc[signal_date] + 0.5 * blended
        else:
            raise ValueError(name)

        picks = pick_top(score_row, universe.loc[signal_date])
        if len(picks) < max(5, math.ceil(N_PICKS * 0.7)):
            continue

        tr = basket_returns(W, picks, entry_i, exit_i, cost)
        if len(tr) < max(5, math.ceil(N_PICKS * 0.7)):
            continue

        basket_ret = float(np.mean([x["net_ret_pct"] for x in tr]))
        capital *= 1.0 + basket_ret / 100.0

        portfolio.append({
            "signal_date": signal_date,
            "entry_date": dates[entry_i],
            "exit_date": dates[exit_i],
            "basket_ret_pct": basket_ret,
            "nav": capital,
            "n_positions": len(tr),
            "amihud_weight": am_weight,
            "stress_z": stress_z,
            "trend_score": trend_score,
            "trend_gate": trend_gate,
            "recent_B_minus_BASE_pct": excess_mean,
        })

        for x in tr:
            rows.append({
                "variant": name,
                "signal_date": x["entry_date"] - pd.Timedelta(days=1),
                "entry_date": x["entry_date"],
                "exit_date": x["exit_date"],
                "ticker": x["ticker"],
                "entry_px": x["entry_px"],
                "exit_px": x["exit_px"],
                "raw_ret_pct": x["raw_ret_pct"],
                "net_ret_pct": x["net_ret_pct"],
                "cost_rt_pct": cost,
                "amihud_weight": am_weight,
                "stress_z": stress_z,
                "trend_score": trend_score,
                "trend_gate": trend_gate,
                "recent_B_minus_BASE_pct": excess_mean,
            })

    pf = pd.DataFrame(portfolio)
    trdf = pd.DataFrame(rows)
    if pf.empty:
        raise RuntimeError(f"{name}: tamamlanmış sepet yok.")
    pf["signal_date"] = pd.to_datetime(pf["signal_date"])
    pf["entry_date"] = pd.to_datetime(pf["entry_date"])
    pf["exit_date"] = pd.to_datetime(pf["exit_date"])
    return VariantResult(name, cost, trdf, pf.sort_values("exit_date").reset_index(drop=True))


def metrics(result: VariantResult) -> dict[str, float]:
    pf = result.portfolio.copy()
    tr = result.trades
    nav = pd.Series(pf["nav"].values, index=pf["exit_date"])
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    dd = nav / nav.cummax() - 1
    basket_win = pf["basket_ret_pct"] > 0
    trade_win = tr["net_ret_pct"] > 0 if not tr.empty else pd.Series(dtype=bool)
    gains = float(tr.loc[tr.net_ret_pct > 0, "net_ret_pct"].sum()) if not tr.empty else 0.0
    losses = float(-tr.loc[tr.net_ret_pct <= 0, "net_ret_pct"].sum()) if not tr.empty else 0.0
    pfactor = gains / losses if losses > 0 else np.inf

    monthly = pf.set_index("exit_date")["basket_ret_pct"].resample("ME").apply(
        lambda x: (np.prod(1 + x / 100.0) - 1.0) * 100.0 if len(x) else np.nan
    ).dropna()
    worst12 = np.nan
    if len(monthly) >= 12:
        roll12 = (1 + monthly / 100.0).rolling(12).apply(np.prod, raw=True) - 1.0
        worst12 = float(roll12.min() * 100.0)

    return {
        "CAGR_pct": cagr * 100.0,
        "Total_x": float(nav.iloc[-1] / nav.iloc[0]),
        "MaxDD_pct": float(dd.min() * 100.0),
        "BasketWin_pct": float(basket_win.mean() * 100.0),
        "TradeWin_pct": float(trade_win.mean() * 100.0) if len(trade_win) else np.nan,
        "AvgTrade_pct": float(tr.net_ret_pct.mean()) if not tr.empty else np.nan,
        "ProfitFactor": pfactor,
        "Worst12M_pct": worst12,
        "Baskets": int(len(pf)),
        "Trades": int(len(tr)),
    }


def paired_bootstrap(base, test, n_boot=5000, seed=SEED):
    def monthly(r):
        p = r.portfolio.set_index("exit_date")["basket_ret_pct"]
        return ((1 + p / 100.0).resample("ME").prod() - 1.0) * 100.0

    a = monthly(base)
    b = monthly(test)
    df = pd.concat([a.rename("base"), b.rename("test")], axis=1).dropna()
    if len(df) < 12:
        return {"months": int(len(df)), "mean": np.nan, "median": np.nan, "ci5": np.nan, "ci95": np.nan,
                "p_le0": np.nan, "positive_pct": np.nan}

    diff = (df.test - df.base).to_numpy(float)
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {
        "months": int(len(diff)),
        "mean": float(diff.mean()),
        "median": float(np.median(diff)),
        "ci5": float(np.quantile(means, 0.05)),
        "ci95": float(np.quantile(means, 0.95)),
        "p_le0": float((means <= 0).mean()),
        "positive_pct": float((diff > 0).mean() * 100.0),
    }


def cost_stress(W, universe, scores, regime, history, start, costs):
    rows = []
    for cost in costs:
        b0 = run_variant("BASE", W, universe, scores, regime, history, cost, start)
        b3 = run_variant("B3", W, universe, scores, regime, history, cost, start)
        m0 = metrics(b0)
        m3 = metrics(b3)
        rows.append({
            "Maliyet_RT_pct": cost,
            "BASE_CAGR_pct": m0["CAGR_pct"],
            "B3_CAGR_pct": m3["CAGR_pct"],
            "B3_minus_BASE_CAGR_pp": m3["CAGR_pct"] - m0["CAGR_pct"],
            "BASE_MaxDD_pct": m0["MaxDD_pct"],
            "B3_MaxDD_pct": m3["MaxDD_pct"],
            "BASE_PF": m0["ProfitFactor"],
            "B3_PF": m3["ProfitFactor"],
            "BASE_TradeWin_pct": m0["TradeWin_pct"],
            "B3_TradeWin_pct": m3["TradeWin_pct"],
        })
    return pd.DataFrame(rows)


def block_metrics(result: VariantResult, start: str, end: str):
    pf = result.portfolio.copy()
    tr = result.trades.copy()
    pf = pf[(pf.exit_date >= pd.Timestamp(start)) & (pf.exit_date <= pd.Timestamp(end))]
    tr["exit_date"] = pd.to_datetime(tr["exit_date"]) if not tr.empty else pd.Series(dtype="datetime64[ns]")
    tr = tr[(tr.exit_date >= pd.Timestamp(start)) & (tr.exit_date <= pd.Timestamp(end))] if not tr.empty else tr

    if pf.empty:
        return {"cagr": np.nan, "dd": np.nan, "win": np.nan, "pf": np.nan, "baskets": 0}

    nav = pd.Series(
        np.r_[1.0, (1.0 + pf.basket_ret_pct.to_numpy() / 100.0).cumprod()],
        index=[pf.exit_date.iloc[0] - pd.Timedelta(days=1)] + list(pf.exit_date)
    )
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1.0
    dd = (nav / nav.cummax() - 1.0).min()

    if tr.empty:
        win = np.nan
        pfactor = np.nan
    else:
        win = float((tr.net_ret_pct > 0).mean() * 100.0)
        gains = tr.loc[tr.net_ret_pct > 0, "net_ret_pct"].sum()
        losses = -tr.loc[tr.net_ret_pct <= 0, "net_ret_pct"].sum()
        pfactor = float(gains / losses) if losses > 0 else np.inf

    return {"cagr": cagr * 100.0, "dd": dd * 100.0, "win": win, "pf": pfactor, "baskets": int(len(pf))}


def make_blocks(start: pd.Timestamp, end: pd.Timestamp):
    # Sabit, önceden belirlenmiş 3 dönem:
    # geliştirme, ara doğrulama, son dönem/holdout.
    y0, y1 = int(start.year), int(end.year)
    candidates = [
        (f"{y0}-{min(y0+4, y1)}", f"{y0}-01-01", f"{min(y0+4, y1)}-12-31"),
        (f"{min(y0+5,y1)}-{min(y0+8,y1)}", f"{min(y0+5,y1)}-01-01", f"{min(y0+8,y1)}-12-31"),
        (f"{min(y0+9,y1)}-{y1}", f"{min(y0+9,y1)}-01-01", f"{y1}-12-31"),
    ]
    out = []
    for label, s, e in candidates:
        if pd.Timestamp(s) <= pd.Timestamp(e):
            out.append((label, s, e))
    return out


def fmt(x, digits=2):
    if x is None:
        return "—"
    try:
        if not np.isfinite(float(x)):
            return "—"
    except Exception:
        return str(x)
    return f"{float(x):.{digits}f}"


def build_report(years, start, end, results, cost_df, boot_b2, boot_b3, blocks, regime):
    m = {k: metrics(v) for k, v in results.items()}
    b3_delta = m["B3"]["CAGR_pct"] - m["BASE"]["CAGR_pct"]
    boot_ok = np.isfinite(boot_b3["ci5"]) and boot_b3["ci5"] > 0
    cost_ok = int((cost_df["B3_minus_BASE_CAGR_pp"] > 0).sum())
    cost_total = len(cost_df)

    if b3_delta >= 2 and boot_ok and cost_ok >= math.ceil(cost_total * 0.6):
        verdict = "ROBUST PASS ADAYI"
    elif b3_delta > 0 and cost_ok >= math.ceil(cost_total * 0.5):
        verdict = "WEAK PASS"
    else:
        verdict = "FAIL / YETERSİZ"

    lines = [
        "# B3 Regime Stabilizer — Bağımsız Araştırma Raporu",
        f"_Üretim: {pd.Timestamp.now():%Y-%m-%d %H:%M} · veri: {years} yıl + warm-up · ölçüm: {start.date()} → {end.date()}_",
        "",
        "## 1. Karar",
        f"**{verdict}**",
        "",
        "B3, BASE/B/B2'den bağımsız hesaplama yoluyla test edildi; `engine.py`, `portfolio.py`, `signals.py` kullanılmadı.",
        "B3 kuralı test öncesinde sabittir; test verisi içinde parametre taraması yapılmadı.",
        "",
        "## 2. Ana karşılaştırma",
        "",
        "| Metrik | BASE | B | B2 | B3 |",
        "|---|---:|---:|---:|---:|",
        f"| CAGR | {fmt(m['BASE']['CAGR_pct'])}% | {fmt(m['B']['CAGR_pct'])}% | {fmt(m['B2']['CAGR_pct'])}% | **{fmt(m['B3']['CAGR_pct'])}%** |",
        f"| Toplam çarpan | {fmt(m['BASE']['Total_x'])}x | {fmt(m['B']['Total_x'])}x | {fmt(m['B2']['Total_x'])}x | **{fmt(m['B3']['Total_x'])}x** |",
        f"| Max DD | {fmt(m['BASE']['MaxDD_pct'])}% | {fmt(m['B']['MaxDD_pct'])}% | {fmt(m['B2']['MaxDD_pct'])}% | {fmt(m['B3']['MaxDD_pct'])}% |",
        f"| Sepet win | {fmt(m['BASE']['BasketWin_pct'])}% | {fmt(m['B']['BasketWin_pct'])}% | {fmt(m['B2']['BasketWin_pct'])}% | {fmt(m['B3']['BasketWin_pct'])}% |",
        f"| İşlem win | {fmt(m['BASE']['TradeWin_pct'])}% | {fmt(m['B']['TradeWin_pct'])}% | {fmt(m['B2']['TradeWin_pct'])}% | {fmt(m['B3']['TradeWin_pct'])}% |",
        f"| İşlem başı net | {fmt(m['BASE']['AvgTrade_pct'])}% | {fmt(m['B']['AvgTrade_pct'])}% | {fmt(m['B2']['AvgTrade_pct'])}% | {fmt(m['B3']['AvgTrade_pct'])}% |",
        f"| Profit factor | {fmt(m['BASE']['ProfitFactor'])} | {fmt(m['B']['ProfitFactor'])} | {fmt(m['B2']['ProfitFactor'])} | {fmt(m['B3']['ProfitFactor'])} |",
        f"| En kötü 12 ay | {fmt(m['BASE']['Worst12M_pct'])}% | {fmt(m['B']['Worst12M_pct'])}% | {fmt(m['B2']['Worst12M_pct'])}% | {fmt(m['B3']['Worst12M_pct'])}% |",
        "",
        "## 3. B3 kuralı",
        "",
        "- B2'nin robust Amihud bileşeni korunur.",
        "- 200DMA üzerindeki likit evren oranı + 63 günlük medyan momentum ile trend skoru üretilir.",
        "- Trend zayıfsa Amihud ağırlığı BASE/low_max tarafına çekilir.",
        "- Trend güçlüyse B2'nin Amihud ağırlığı korunur.",
        "- Çok yüksek Amihud stresinde ek bir sınırlama uygulanır.",
        "- Ağırlık daima 0.00–0.50 aralığındadır.",
        "",
        "## 4. Maliyet stres testi",
        "",
        cost_df.to_markdown(index=False, floatfmt=".2f"),
        "",
        f"B3, {cost_total} maliyet senaryosunun **{cost_ok}/{cost_total}** tanesinde BASE CAGR'ını geçti.",
        "",
        "## 5. Eşleştirilmiş aylık bootstrap",
        "",
        "| Test | Ort. fark | Medyan | %5 CI | %95 CI | P(≤0) | Pozitif ay % |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| B − BASE | {fmt(boot_b2['mean'])} | {fmt(boot_b2['median'])} | {fmt(boot_b2['ci5'])} | {fmt(boot_b2['ci95'])} | {fmt(boot_b2['p_le0']*100)}% | {fmt(boot_b2['positive_pct'])}% |",
        f"| B3 − BASE | {fmt(boot_b3['mean'])} | {fmt(boot_b3['median'])} | {fmt(boot_b3['ci5'])} | {fmt(boot_b3['ci95'])} | {fmt(boot_b3['p_le0']*100)}% | {fmt(boot_b3['positive_pct'])}% |",
        "",
        "## 6. Sabit zaman blokları",
        "",
        "| Dönem | BASE CAGR | B CAGR | B2 CAGR | B3 CAGR | B3−BASE | B3 DD | B3 win |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for label, s, e in blocks:
        r0 = block_metrics(results["BASE"], s, e)
        rb = block_metrics(results["B"], s, e)
        r2 = block_metrics(results["B2"], s, e)
        r3 = block_metrics(results["B3"], s, e)
        lines.append(
            f"| {label} | {fmt(r0['cagr'])}% | {fmt(rb['cagr'])}% | {fmt(r2['cagr'])}% | "
            f"{fmt(r3['cagr'])}% | {fmt(r3['cagr']-r0['cagr'])} pp | {fmt(r3['dd'])}% | {fmt(r3['win'])}% |"
        )

    lines += [
        "",
        "## 7. Rejim dağılımı",
        "",
        f"- Ortalama trend skoru: **{fmt(regime['trend_score'].mean())}**",
        f"- Trend skoru medyanı: **{fmt(regime['trend_score'].median())}**",
        f"- 200DMA breadth ortalaması: **{fmt(regime['breadth_above_200dma'].mean()*100)}%**",
        f"- 63g median momentum ortalaması: **{fmt(regime['median_63d_momentum'].mean()*100)}%**",
        "",
        "## 8. Bilimsel yorum",
        "",
        "B3'ün hedefi B'nin yüksek CAGR'ını körlemesine takip etmek değildir.",
        "Hedef, Amihud bilgisini yalnızca daha elverişli trend ortamlarında daha yüksek ağırlıklandırıp, zayıf trendlerde BASE davranışına yaklaşmaktır.",
        "Bu sonuç canlıya alma izni değildir. Özellikle final holdout üzerinde sonuç görüldükten sonra kural değiştirilmemelidir.",
        "",
        "**Canlı sistemi değiştirmez.**",
    ]
    return "\n".join(lines) + "\n"


def synthetic_panel(years=8, n_tickers=80, seed=SEED):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=years * 252)
    tickers = [f"T{j:03d}" for j in range(n_tickers)]
    drift = rng.normal(0.00035, 0.0002, n_tickers)
    vol = np.abs(rng.normal(0.018, 0.004, n_tickers))
    px = np.full(n_tickers, 50.0)
    rows = []
    for d in days:
        shock = rng.normal(size=n_tickers)
        ret = drift + vol * shock
        px *= 1 + ret
        o = px / np.maximum(1 + ret * rng.uniform(0.2, 0.5, n_tickers), 0.1)
        h = np.maximum(o, px) * (1 + np.abs(rng.normal(0, 0.005, n_tickers)))
        l = np.minimum(o, px) * (1 - np.abs(rng.normal(0, 0.005, n_tickers)))
        v = rng.lognormal(15, 0.6, n_tickers)
        for j, t in enumerate(tickers):
            rows.append((d, t, o[j], h[j], l[j], px[j], v[j]))
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])


def run_synthetic():
    panel = synthetic_panel()
    W = make_wide(panel)
    U = apply_data_filters(W)
    S, R = prepare_features(W, U)
    start = W["c"].index[WARMUP_DAYS]
    hist = build_base_b_history(W, U, S, start, 0.35)
    for n in ("BASE", "B", "B2", "B3"):
        r = run_variant(n, W, U, S, R, hist, 0.35, start)
        print(n, metrics(r))
    print("SENTETİK SINAMA — gerçek performans değildir")


def parse_costs(s: str) -> list[float]:
    vals = []
    for p in s.split(","):
        x = float(p.strip())
        if not 0 <= x <= 10:
            raise ValueError("Maliyet 0–10% RT aralığında olmalı.")
        vals.append(x)
    if not vals:
        raise ValueError("En az bir maliyet seviyesi gerekli.")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="B3 Regime Stabilizer bağımsız araştırma testi")
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--costs", default=",".join(str(x) for x in DEFAULT_COSTS))
    ap.add_argument("--out", default=str(REPORT_FILE))
    ap.add_argument("--trades-out", default=str(TRADES_FILE))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        run_synthetic()
        return 0

    if args.years < 5:
        raise ValueError("En az 5 yıl önerilir.")

    costs = parse_costs(args.costs)
    tickers = load_universe()
    panel = download_bist(tickers, args.years)
    W = make_wide(panel)
    universe = apply_data_filters(W)
    scores, regime = prepare_features(W, universe)

    dates = pd.DatetimeIndex(W["c"].index)
    if len(dates) <= WARMUP_DAYS + HOLD_DAYS + 10:
        raise RuntimeError("Yeterli geçmiş oluşmadı.")

    start = dates[WARMUP_DAYS]
    end = dates[-HOLD_DAYS - 1]

    history = build_base_b_history(W, universe, scores, start, costs[0])

    results = {}
    for name in ("BASE", "B", "B2", "B3"):
        results[name] = run_variant(name, W, universe, scores, regime, history, costs[0], start)

    cost_df = cost_stress(W, universe, scores, regime, history, start, costs)
    boot_b2 = paired_bootstrap(results["BASE"], results["B2"])
    boot_b3 = paired_bootstrap(results["BASE"], results["B3"])
    blocks = make_blocks(start, end)

    report = build_report(
        years=args.years,
        start=start,
        end=end,
        results=results,
        cost_df=cost_df,
        boot_b2=boot_b2,
        boot_b3=boot_b3,
        blocks=blocks,
        regime=regime.loc[start:end],
    )

    Path(args.out).write_text(report, encoding="utf-8")
    pd.concat([r.trades for r in results.values()], ignore_index=True).to_csv(
        args.trades_out, index=False
    )

    print("\n=== B3 REGIME STABILIZER ===")
    for name in ("BASE", "B", "B2", "B3"):
        m = metrics(results[name])
        print(
            f"{name}: CAGR={m['CAGR_pct']:.2f}% | "
            f"DD={m['MaxDD_pct']:.2f}% | "
            f"PF={m['ProfitFactor']:.2f} | "
            f"Win={m['TradeWin_pct']:.2f}%"
        )
    print(f"Rapor: {args.out}")
    print(f"İşlemler: {args.trades_out}")
    print("Canlı data/state/orders dosyalarına yazılmadı.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"❌ HATA: {exc}", file=sys.stderr)
        raise SystemExit(1)
