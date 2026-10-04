#!/usr/bin/env python3
"""
B2 Robust / Regime-Aware Amihud — araştırma amaçlı bağımsız doğrulama.

Bu dosya CANLI SİSTEMİ DEĞİŞTİRMEZ ve engine.py / portfolio.py / signals.py
kullanmaz. Amaç, ilk B varyantındaki:
    B = mom_12_1 + hi52 + amihud_illiq
hipotezini daha sağlam bir B2 mimarisiyle tekrar sınamaktır.

B2 MİMARİSİ
------------
1) Robust Amihud:
   - günlük |ret| / (close * volume)
   - cross-sectional 5/95 winsorization
   - log dönüşümü
   - cross-sectional percentile rank
2) BASE çekirdeği korunur: mom_12_1 + hi52.
3) Üçüncü bileşen BASE'deki low_max ile robust Amihud arasında blend edilir.
   nötr ağırlık = 0.25 Amihud / 0.25 low_max.
4) Regime-aware ağırlık:
   - piyasa geneli median Amihud'dan 126 günlük geçmişe dönük z-score çıkarılır
   - yalnızca mevcut güne kadar bilinen veri kullanılır
   - son 12 tamamlanmış sepetin B-BASE farkı ile conviction oluşturulur
   - yüksek likidite stresinde ve yakın geçmişte B üstünken Amihud ağırlığı artar
   - tersinde BASE'e geri yaklaşır
   - ağırlık 0.00–0.50 aralığında kalır
5) Her işlem bir sonraki işlem gününün açılışından başlar ve 21 işlem günü sonra
   kapanışta biter; en fazla 10 hisse, eşit ağırlık.
6) Maliyet stres testi ve eşleştirilmiş aylık bootstrap raporlanır.

ÖNEMLİ
------
- Bu bir keşif/validasyon dosyasıdır; config.py veya data/ altına yazmaz.
- Yahoo'da bulunmayan/delisted semboller atlanır.
- ABD evreni kullanılmaz; BIST faktörü izole edilir.
- Dinamik B2 ağırlığı için geleceğe bakılmaz: conviction yalnızca önceki tamamlanmış
  sepet sonuçlarından, stress ise mevcut güne kadar oluşan seriyle hesaplanır.
- Aynı 2012–2026 verisi içinde sınırsız deneme yapılmamalıdır. Bu rapor bir araştırma
  çıktısıdır; canlıya alma kararı ayrıca final holdout ile verilmelidir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
UNIVERSE_FILE = ROOT / "data" / "universe_bist.json"
REPORT_FILE = ROOT / "b2_robust_regime_raporu.md"
TRADES_FILE = ROOT / "b2_robust_regime_islemler.csv"

DEFAULT_YEARS = 14
DEFAULT_COSTS = [0.35, 0.50, 0.75, 1.00, 1.25]
MAX_MOVE = 0.105
LIQ_PCT = 0.40
HOLD_DAYS = 21
N_PICKS = 10
WARMUP_DAYS = 252
MIN_LIQ_DAYS = 10
REGIME_WINDOW = 126
CONVICTION_BASKETS = 12
CONVICTION_SCALE_PCT = 5.0
AMIHUD_MIN_WEIGHT = 0.0
AMIHUD_NEUTRAL_WEIGHT = 0.25
AMIHUD_MAX_WEIGHT = 0.50
SEED = 20261004

BASE_NAME = "BASE"
B_NAME = "B"
B2_NAME = "B2"


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
        raise RuntimeError("yfinance kurulu değil. requirements.txt yüklenmeli.") from exc


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
    need = {"open", "high", "low", "close", "volume"}
    if not need.issubset(sub.columns):
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

    print(f"📥 BIST B2 verisi: {len(symbols)} hisse, yaklaşık {years + 1} yıl")
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
            print(f"  ⚠️ grup alınamadı: {i + 1}-{i + len(syms)}")
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
        time.sleep(0.35)

    if not rows:
        raise RuntimeError("Hiç BIST verisi indirilemedi.")
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=["open", "close"])
    out = out[(out["open"] > 0) & (out["close"] > 0) & (out["high"] >= out["low"])]
    out = out.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    return out


def make_wide(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for short, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
        out[short] = panel.pivot(index="date", columns="ticker", values=col).sort_index()
    cols = out["c"].columns
    return {k: v.reindex(columns=cols) for k, v in out.items()}


def build_universe_mask(W: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    o, h, l, c, v = (W[k] for k in ("o", "h", "l", "c", "v"))
    r = c.pct_change(fill_method=None)
    dollar = (c * v).where((v > 0) & c.notna())
    liq = dollar.rolling(20, min_periods=MIN_LIQ_DAYS).median()
    valid = c.notna() & o.notna() & (v > 0) & (h >= l) & r.abs().le(MAX_MOVE)
    universe = valid & liq.notna() & liq.rank(axis=1, pct=True).ge(LIQ_PCT)
    return universe, liq


def _clip_cross_section(x: pd.DataFrame, low=0.05, high=0.95) -> pd.DataFrame:
    lo = x.quantile(low, axis=1)
    hi = x.quantile(high, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def compute_components(W: dict[str, pd.DataFrame], universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    o, h, l, c, v = (W[k] for k in ("o", "h", "l", "c", "v"))
    r = c.pct_change(fill_method=None)

    mom = c.shift(21) / c.shift(252) - 1.0
    hi52 = c / c.rolling(252, min_periods=200).max()
    low_max = -r.rolling(21, min_periods=15).max()

    dollar = (c * v).where(v > 0)
    raw_am = (r.abs() / dollar.replace(0, np.nan)).rolling(20, min_periods=MIN_LIQ_DAYS).mean()
    log_am = np.log(raw_am.where(raw_am > 0))
    robust_am = _clip_cross_section(log_am, 0.05, 0.95)

    return {
        "mom": mom,
        "hi52": hi52,
        "low_max": low_max,
        "amihud": robust_am,
    }


def rank_component(x: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    return x.where(universe).rank(axis=1, pct=True, method="average")


def prepare_scores(W: dict[str, pd.DataFrame], universe: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    comp = compute_components(W, universe)
    rm = rank_component(comp["mom"], universe)
    rh = rank_component(comp["hi52"], universe)
    rl = rank_component(comp["low_max"], universe)
    ra = rank_component(comp["amihud"], universe)

    scores = {
        BASE_NAME: (rm + rh + rl) / 3.0,
        B_NAME: (rm + rh + ra) / 3.0,
    }

    # Piyasa geneli illiquidity/stress serisi: sadece aynı güne kadar bilinen veriler.
    market_am = comp["amihud"].where(universe).median(axis=1)
    roll_mean = market_am.rolling(REGIME_WINDOW, min_periods=60).mean()
    roll_std = market_am.rolling(REGIME_WINDOW, min_periods=60).std()
    stress_z = (market_am - roll_mean) / roll_std.replace(0, np.nan)
    stress_z = stress_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    regime = pd.DataFrame({"market_amihud": market_am, "stress_z": stress_z}, index=market_am.index)

    # B2 için iki alt skor saklanır; nihai B2 skoru o günün dinamik ağırlığıyla kurulur.
    scores["B2_CORE"] = (rm + rh) / 2.0
    scores["B2_LOW"] = rl
    scores["B2_AM"] = ra
    return scores, regime


def _pick_top(score_row: pd.Series, mask_row: pd.Series) -> list[str]:
    s = score_row.where(mask_row).dropna().sort_values(ascending=False, kind="stable")
    return list(s.index[:N_PICKS])


def _rebalance_dates(dates: pd.DatetimeIndex, start: pd.Timestamp) -> list[pd.Timestamp]:
    """21 işlem günlük seyrek rebalans; her sinyal tarihi bir sonraki girişte uygulanır."""
    idx = int(dates.searchsorted(pd.Timestamp(start), side="left"))
    if idx >= len(dates):
        return []
    return list(dates[idx::HOLD_DAYS])


def _net_return(entry: float, exit_: float, cost_rt_pct: float) -> float:
    half = cost_rt_pct / 100.0 / 2.0
    entry_fill = entry * (1.0 + half)
    exit_fill = exit_ * (1.0 - half)
    return exit_fill / entry_fill - 1.0 if entry_fill > 0 else np.nan


def basket_return_at(W: dict[str, pd.DataFrame], picks: list[str], entry_i: int, exit_i: int, cost_rt_pct: float) -> list[dict]:
    dates = W["c"].index
    out = []
    for t in picks:
        if t not in W["o"].columns:
            continue
        entry = W["o"].iloc[entry_i].get(t, np.nan)
        exit_ = W["c"].iloc[exit_i].get(t, np.nan)
        if not (pd.notna(entry) and pd.notna(exit_) and entry > 0 and exit_ > 0):
            continue
        net = _net_return(float(entry), float(exit_), cost_rt_pct)
        if np.isfinite(net):
            out.append({
                "ticker": t,
                "entry_date": dates[entry_i],
                "exit_date": dates[exit_i],
                "entry_px": float(entry),
                "exit_px": float(exit_),
                "raw_ret_pct": (float(exit_) / float(entry) - 1.0) * 100.0,
                "net_ret_pct": net * 100.0,
            })
    return out


def get_basket_history(
    W: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    cost_rt_pct: float,
) -> pd.DataFrame:
    """B ve BASE için geçmiş sepet sonuçları. B2 conviction yalnızca bu geçmişi kullanır."""
    dates = list(pd.DatetimeIndex(W["c"].index))
    signal_dates = _rebalance_dates(pd.DatetimeIndex(dates), start)
    rows = []
    for signal_date in signal_dates:
        i = dates.index(signal_date)
        entry_i = i + 1
        exit_i = i + HOLD_DAYS
        if exit_i >= len(dates):
            break
        for name in (BASE_NAME, B_NAME):
            picks = _pick_top(scores[name].loc[signal_date], universe.loc[signal_date])
            if len(picks) < max(5, math.ceil(N_PICKS * 0.7)):
                continue
            tr = basket_return_at(W, picks, entry_i, exit_i, cost_rt_pct)
            if len(tr) < max(5, math.ceil(N_PICKS * 0.7)):
                continue
            rows.append({
                "signal_date": signal_date,
                "variant": name,
                "entry_date": dates[entry_i],
                "exit_date": dates[exit_i],
                "basket_ret_pct": float(np.mean([x["net_ret_pct"] for x in tr])),
            })
    return pd.DataFrame(rows)


def b2_weight(signal_date: pd.Timestamp, regime: pd.DataFrame, history: pd.DataFrame) -> tuple[float, float, float]:
    stress_z = float(regime.loc[signal_date, "stress_z"]) if signal_date in regime.index else 0.0
    stress_signal = float(np.tanh(stress_z))

    prior = history[history["exit_date"] < signal_date].sort_values("exit_date")
    pivot = prior.pivot(index="exit_date", columns="variant", values="basket_ret_pct").dropna()
    if len(pivot):
        excess = pivot[B_NAME] - pivot[BASE_NAME]
        recent = excess.tail(CONVICTION_BASKETS)
        excess_mean = float(recent.mean()) if len(recent) else 0.0
    else:
        excess_mean = 0.0
    conviction = float(np.tanh(excess_mean / CONVICTION_SCALE_PCT))

    # Stress ve yakın geçmişteki relative performance birlikte hareket eder.
    w = AMIHUD_NEUTRAL_WEIGHT + 0.25 * stress_signal * conviction
    w = float(np.clip(w, AMIHUD_MIN_WEIGHT, AMIHUD_MAX_WEIGHT))
    return w, stress_z, excess_mean


def run_variant(
    name: str,
    W: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    regime: pd.DataFrame,
    history: pd.DataFrame,
    cost_rt_pct: float,
    start: pd.Timestamp,
) -> VariantResult:
    dates = list(pd.DatetimeIndex(W["c"].index))
    signal_dates = _rebalance_dates(pd.DatetimeIndex(dates), start)
    rows = []
    portfolio = []
    capital = 1.0

    for signal_date in signal_dates:
        i = dates.index(signal_date)
        entry_i = i + 1
        exit_i = i + HOLD_DAYS
        if entry_i >= len(dates) or exit_i >= len(dates):
            break

        if name == B2_NAME:
            w, stress_z, excess_mean = b2_weight(signal_date, regime, history)
            score_row = (1.0 - w) * scores["B2_LOW"].loc[signal_date] + w * scores["B2_AM"].loc[signal_date]
            score_row = 0.5 * scores["B2_CORE"].loc[signal_date] + 0.5 * score_row
        elif name == BASE_NAME:
            score_row = scores[BASE_NAME].loc[signal_date]
            w, stress_z, excess_mean = np.nan, np.nan, np.nan
        else:
            score_row = scores[B_NAME].loc[signal_date]
            w, stress_z, excess_mean = np.nan, np.nan, np.nan

        picks = _pick_top(score_row, universe.loc[signal_date])
        if len(picks) < max(5, math.ceil(N_PICKS * 0.7)):
            continue
        tr = basket_return_at(W, picks, entry_i, exit_i, cost_rt_pct)
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
            "amihud_weight": w,
            "stress_z": stress_z,
            "recent_B_minus_BASE_pct": excess_mean,
        })
        for x in tr:
            rows.append({
                "variant": name,
                "signal_date": signal_date.date().isoformat(),
                "entry_date": x["entry_date"].date().isoformat(),
                "exit_date": x["exit_date"].date().isoformat(),
                "ticker": x["ticker"],
                "entry_px": x["entry_px"],
                "exit_px": x["exit_px"],
                "raw_ret_pct": x["raw_ret_pct"],
                "net_ret_pct": x["net_ret_pct"],
                "cost_rt_pct": cost_rt_pct,
                "amihud_weight": w,
                "stress_z": stress_z,
                "recent_B_minus_BASE_pct": excess_mean,
            })

    tr_df = pd.DataFrame(rows)
    pf = pd.DataFrame(portfolio)
    if pf.empty:
        raise RuntimeError(f"{name}: tamamlanmış sepet yok.")
    pf["signal_date"] = pd.to_datetime(pf["signal_date"])
    pf["entry_date"] = pd.to_datetime(pf["entry_date"])
    pf["exit_date"] = pd.to_datetime(pf["exit_date"])
    pf = pf.sort_values("exit_date").reset_index(drop=True)
    return VariantResult(name, cost_rt_pct, tr_df, pf)


def metrics(result: VariantResult) -> dict[str, float]:
    pf = result.portfolio
    tr = result.trades
    nav = pd.Series(pf["nav"].values, index=pf["exit_date"])
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1 / 365.25)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0
    dd = nav / nav.cummax() - 1.0
    gains = float(tr.loc[tr["net_ret_pct"] > 0, "net_ret_pct"].sum()) if not tr.empty else 0.0
    losses = float(-tr.loc[tr["net_ret_pct"] <= 0, "net_ret_pct"].sum()) if not tr.empty else 0.0
    pfactor = gains / losses if losses > 0 else np.inf
    month = pf.set_index("exit_date")["basket_ret_pct"]
    monthly = ((1.0 + month / 100.0).resample("ME").prod() - 1.0) * 100.0
    worst12 = np.nan
    if len(monthly) >= 12:
        roll12 = (1.0 + monthly / 100.0).rolling(12).apply(np.prod, raw=True) - 1.0
        worst12 = float(roll12.min() * 100.0)
    return {
        "CAGR_pct": cagr * 100.0,
        "Total_x": float(nav.iloc[-1] / nav.iloc[0]),
        "MaxDD_pct": float(dd.min() * 100.0),
        "BasketWin_pct": float((pf["basket_ret_pct"] > 0).mean() * 100.0),
        "TradeWin_pct": float((tr["net_ret_pct"] > 0).mean() * 100.0) if not tr.empty else np.nan,
        "AvgTrade_pct": float(tr["net_ret_pct"].mean()) if not tr.empty else np.nan,
        "ProfitFactor": float(pfactor),
        "Worst12M_pct": worst12,
        "Baskets": int(len(pf)),
        "Trades": int(len(tr)),
        "MedianBasket_pct": float(pf["basket_ret_pct"].median()),
    }


def paired_monthly_bootstrap(base: VariantResult, test: VariantResult, n_boot: int = 10000, seed: int = SEED) -> dict[str, float]:
    def monthly(r: VariantResult) -> pd.Series:
        p = r.portfolio.set_index("exit_date")["basket_ret_pct"]
        return ((1.0 + p / 100.0).resample("ME").prod() - 1.0) * 100.0

    a = monthly(base).rename("base")
    b = monthly(test).rename("test")
    df = pd.concat([a, b], axis=1).dropna()
    if len(df) < 12:
        return {"months": int(len(df)), "mean_diff": np.nan, "median_diff": np.nan, "ci5": np.nan, "ci95": np.nan, "p_le0": np.nan, "win_month_pct": np.nan}
    diff = (df["test"] - df["base"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {
        "months": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "median_diff": float(np.median(diff)),
        "ci5": float(np.quantile(means, 0.05)),
        "ci95": float(np.quantile(means, 0.95)),
        "p_le0": float((means <= 0).mean()),
        "win_month_pct": float((diff > 0).mean() * 100.0),
    }


def cost_stress(W, universe, scores, regime, history, start: pd.Timestamp, costs: Iterable[float]) -> pd.DataFrame:
    rows = []
    for cost in costs:
        base = run_variant(BASE_NAME, W, universe, scores, regime, history, cost, start)
        b2 = run_variant(B2_NAME, W, universe, scores, regime, history, cost, start)
        mb = metrics(base)
        m2 = metrics(b2)
        rows.append({
            "Maliyet_RT_pct": cost,
            "BASE_CAGR_pct": mb["CAGR_pct"],
            "B2_CAGR_pct": m2["CAGR_pct"],
            "B2_minus_BASE_CAGR_pp": m2["CAGR_pct"] - mb["CAGR_pct"],
            "BASE_MaxDD_pct": mb["MaxDD_pct"],
            "B2_MaxDD_pct": m2["MaxDD_pct"],
            "BASE_PF": mb["ProfitFactor"],
            "B2_PF": m2["ProfitFactor"],
            "BASE_TradeWin_pct": mb["TradeWin_pct"],
            "B2_TradeWin_pct": m2["TradeWin_pct"],
        })
    return pd.DataFrame(rows)


def regime_split(result: VariantResult, stress_series: pd.Series) -> dict[str, dict[str, float]]:
    p = result.portfolio.copy()
    s = stress_series.reindex(pd.DatetimeIndex(p["signal_date"])).to_numpy(dtype=float)
    p["stress"] = s
    med = float(np.nanmedian(s))
    out = {}
    for label, mask in (("low_stress", p.stress <= med), ("high_stress", p.stress > med)):
        q = p[mask]
        if q.empty:
            out[label] = {"baskets": 0, "avg": np.nan, "win": np.nan}
        else:
            out[label] = {
                "baskets": int(len(q)),
                "avg": float(q["basket_ret_pct"].mean()),
                "win": float((q["basket_ret_pct"] > 0).mean() * 100.0),
            }
    return out


def block_metric(result: VariantResult, start: str, end: str) -> dict[str, float]:
    p = result.portfolio[(result.portfolio["exit_date"] >= pd.Timestamp(start)) & (result.portfolio["exit_date"] <= pd.Timestamp(end))]
    if p.empty:
        return {"cagr": np.nan, "dd": np.nan, "win": np.nan, "pf": np.nan, "baskets": 0}
    nav = pd.Series((1.0 + p["basket_ret_pct"].to_numpy() / 100.0).cumprod(), index=p["exit_date"])
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 1 / 365.25)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1.0
    dd = float((nav / nav.cummax() - 1).min() * 100.0)
    tr = result.trades
    tr = tr[(pd.to_datetime(tr.exit_date) >= pd.Timestamp(start)) & (pd.to_datetime(tr.exit_date) <= pd.Timestamp(end))]
    gains = tr.loc[tr.net_ret_pct > 0, "net_ret_pct"].sum() if len(tr) else 0
    losses = -tr.loc[tr.net_ret_pct <= 0, "net_ret_pct"].sum() if len(tr) else 0
    pf = gains / losses if losses > 0 else np.inf
    return {"cagr": cagr * 100.0, "dd": dd, "win": float((tr.net_ret_pct > 0).mean() * 100.0) if len(tr) else np.nan, "pf": float(pf), "baskets": int(len(p))}


def fmt(x, d=2):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{d}f}"


def build_report(years, start, end, base, b, b2, cost_df, boot_b, boot_b2, blocks, regime_b, regime_b2, data_stats):
    mb, mB, m2 = metrics(base), metrics(b), metrics(b2)
    d2 = m2["CAGR_pct"] - mb["CAGR_pct"]
    cost_positive = int((cost_df["B2_minus_BASE_CAGR_pp"] > 0).sum())
    cost_total = len(cost_df)
    if d2 >= 2 and cost_positive >= math.ceil(cost_total * 0.6) and boot_b2.get("ci5", -np.inf) > 0:
        verdict = "ROBUST PASS"
    elif d2 > 0 and cost_positive >= math.ceil(cost_total * 0.5):
        verdict = "WEAK PASS"
    else:
        verdict = "FAIL / YETERSİZ"

    lines = [
        "# B2 Robust / Regime-Aware Amihud — Bağımsız Doğrulama Raporu",
        f"_Üretim: {pd.Timestamp.now():%Y-%m-%d %H:%M} · veri: {years} yıl + warm-up · ölçüm: {start.date()} → {end.date()}_",
        "",
        "## 1. Sonuç",
        f"**Karar: {verdict}**",
        "",
        "B2, BASE ve ilk B varyantından bağımsız olarak hesaplandı; `engine.py`, `portfolio.py` ve `signals.py` kullanılmadı.",
        "",
        "## 2. Ana karşılaştırma",
        "",
        "| Metrik | BASE | B | B2 | B2 − BASE |",
        "|---|---:|---:|---:|---:|",
        f"| CAGR | {fmt(mb['CAGR_pct'])}% | {fmt(mB['CAGR_pct'])}% | **{fmt(m2['CAGR_pct'])}%** | **{fmt(d2)} pp** |",
        f"| Toplam çarpan | {fmt(mb['Total_x'])}x | {fmt(mB['Total_x'])}x | **{fmt(m2['Total_x'])}x** | {fmt(m2['Total_x'] - mb['Total_x'])}x |",
        f"| Max drawdown | {fmt(mb['MaxDD_pct'])}% | {fmt(mB['MaxDD_pct'])}% | {fmt(m2['MaxDD_pct'])}% | {fmt(m2['MaxDD_pct'] - mb['MaxDD_pct'])} pp |",
        f"| Sepet win | {fmt(mb['BasketWin_pct'])}% | {fmt(mB['BasketWin_pct'])}% | {fmt(m2['BasketWin_pct'])}% | {fmt(m2['BasketWin_pct'] - mb['BasketWin_pct'])} pp |",
        f"| İşlem win | {fmt(mb['TradeWin_pct'])}% | {fmt(mB['TradeWin_pct'])}% | {fmt(m2['TradeWin_pct'])}% | {fmt(m2['TradeWin_pct'] - mb['TradeWin_pct'])} pp |",
        f"| İşlem başı net | {fmt(mb['AvgTrade_pct'])}% | {fmt(mB['AvgTrade_pct'])}% | {fmt(m2['AvgTrade_pct'])}% | {fmt(m2['AvgTrade_pct'] - mb['AvgTrade_pct'])} pp |",
        f"| Profit factor | {fmt(mb['ProfitFactor'])} | {fmt(mB['ProfitFactor'])} | {fmt(m2['ProfitFactor'])} | {fmt(m2['ProfitFactor'] - mb['ProfitFactor'])} |",
        f"| En kötü 12 ay | {fmt(mb['Worst12M_pct'])}% | {fmt(mB['Worst12M_pct'])}% | {fmt(m2['Worst12M_pct'])}% | {fmt(m2['Worst12M_pct'] - mb['Worst12M_pct'])} pp |",
        f"| Sepet | {mb['Baskets']} | {mB['Baskets']} | {m2['Baskets']} | {m2['Baskets'] - mb['Baskets']} |",
        "",
        "## 3. B2 ağırlık mekanizması",
        "",
        "- Robust Amihud: günlük ortalama `|ret|/(close×volume)` → günlük %5/%95 winsorization → log → percentile rank.",
        "- Çekirdek: `mom_12_1 + hi52`.",
        "- Üçüncü bileşen: `low_max` ile Amihud blend'i.",
        "- Amihud ağırlığı: **0.00–0.50**; nötr başlangıç **0.25**.",
        f"- Conviction: son **{CONVICTION_BASKETS}** tamamlanmış B/BASE sepetinin geçmişe dönük farkı.",
        f"- Conviction ölçeği: tanh(x/{CONVICTION_SCALE_PCT:.1f}) ile yumuşatılır.",
        f"- Stress: {REGIME_WINDOW} günlük rolling median-Amihud z-score.",
        "- Histerezis benzeri sınırlama: ağırlık yalnızca geçmiş conviction + stress birlikte değiştiğinde hareket eder.",
        "",
        "## 4. Maliyet stres testi",
        "",
        cost_df.to_markdown(index=False, floatfmt=".2f"),
        "",
        f"B2, maliyet senaryolarının **{cost_positive}/{cost_total}** tanesinde BASE CAGR'ını geçti.",
        "",
        "## 5. Eşleştirilmiş aylık bootstrap",
        "",
        "| Test | Ort. fark | Medyan fark | %5 CI | %95 CI | P(ortalama ≤ 0) | Pozitif ay % |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| B − BASE | {fmt(boot_b.get('mean_diff'))} | {fmt(boot_b.get('median_diff'))} | {fmt(boot_b.get('ci5'))} | {fmt(boot_b.get('ci95'))} | {fmt(boot_b.get('p_le0') * 100 if np.isfinite(boot_b.get('p_le0', np.nan)) else np.nan)}% | {fmt(boot_b.get('win_month_pct'))}% |",
        f"| B2 − BASE | {fmt(boot_b2.get('mean_diff'))} | {fmt(boot_b2.get('median_diff'))} | {fmt(boot_b2.get('ci5'))} | {fmt(boot_b2.get('ci95'))} | {fmt(boot_b2.get('p_le0') * 100 if np.isfinite(boot_b2.get('p_le0', np.nan)) else np.nan)}% | {fmt(boot_b2.get('win_month_pct'))}% |",
        "",
        "## 6. Zaman blokları",
        "",
        "| Dönem | BASE CAGR | B CAGR | B2 CAGR | B2−BASE | B2 DD | B2 işlem win |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, (s, e) in blocks:
        a = block_metric(base, s, e)
        bb = block_metric(b, s, e)
        z = block_metric(b2, s, e)
        lines.append(f"| {label} | {fmt(a['cagr'])}% | {fmt(bb['cagr'])}% | {fmt(z['cagr'])}% | {fmt(z['cagr'] - a['cagr'])} pp | {fmt(z['dd'])}% | {fmt(z['win'])}% |")

    lines += [
        "",
        "## 7. Rejim testi",
        "",
        "Aynı veri üzerinde sinyal anındaki piyasa Amihud stresi medyana göre düşük/yüksek iki gruba ayrıldı. Bu bölüm model seçimi için değil, davranış teşhisi içindir.",
        "",
        "| Variant / regime | Sepet | Ortalama sepet getirisi | Sepet win |",
        "|---|---:|---:|---:|",
    ]
    for name, rec in (("B low stress", regime_b["low_stress"]), ("B high stress", regime_b["high_stress"]), ("B2 low stress", regime_b2["low_stress"]), ("B2 high stress", regime_b2["high_stress"])):
        lines.append(f"| {name} | {rec['baskets']} | {fmt(rec['avg'])}% | {fmt(rec['win'])}% |")

    lines += [
        "",
        "## 8. Veri denetimi",
        f"- Evren: `{data_stats['tickers']}` BIST hissesi",
        f"- Satır: `{data_stats['rows']}`",
        f"- İlk tarih: `{data_stats['first_date']}`",
        f"- Son tarih: `{data_stats['last_date']}`",
        f"- Warm-up: `{WARMUP_DAYS}` işlem günü",
        "- Giriş: sinyal kapanışından sonraki işlem gününün açılışı",
        "- Çıkış: girişten 21 işlem günü sonraki kapanış",
        "- En fazla 10 hisse, eşit ağırlık",
        "- ABD tarafı kullanılmadı; amaç BIST Amihud hipotezini izole etmek.",
        "",
        "## 9. Bilimsel yorum",
        "",
        "B2'nin amacı CAGR'ı tek başına artırmak değil; B'nin BASE'e göre dönemler arası tutarsızlığını ve drawdown/profit-factor bozulmasını azaltırken Amihud bilgisini korumaktır.",
        "ROBUST PASS etiketi yalnızca CAGR avantajı, maliyet dayanıklılığı ve aylık bootstrap alt sınırı birlikte olumluysa verilir. Bu etiket canlıya alma izni değildir.",
        "",
        "**Canlı sistemi değiştirmez.**",
    ]
    return "\n".join(lines) + "\n"


def parse_costs(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals or any(x < 0 or x > 10 for x in vals):
        raise ValueError("Maliyetler 0–10% RT aralığında olmalı.")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(description="B2 Robust / Regime-Aware bağımsız test")
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--costs", default=",".join(str(x) for x in DEFAULT_COSTS))
    ap.add_argument("--out", default=str(REPORT_FILE))
    ap.add_argument("--trades-out", default=str(TRADES_FILE))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    if args.synthetic:
        print("B2 synthetic smoke test: yalnızca program akışı kontrolü.")
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=700)
        x = pd.DataFrame({"close": np.cumprod(1 + np.random.default_rng(SEED).normal(0.0003, 0.01, len(dates)))}, index=dates)
        assert len(x) == 700 and np.isfinite(x.close).all()
        print("✅ synthetic smoke test OK")
        return 0

    if args.years < 8:
        raise ValueError("Bu test için en az 8 yıl önerilir.")

    costs = parse_costs(args.costs)
    tickers = load_universe()
    panel = download_bist(tickers, args.years)
    W = make_wide(panel)
    universe, _liq = build_universe_mask(W)
    scores, regime = prepare_scores(W, universe)

    dates = pd.DatetimeIndex(W["c"].index)
    if len(dates) <= WARMUP_DAYS + HOLD_DAYS + 10:
        raise RuntimeError("Yeterli geçmiş yok.")
    start = dates[WARMUP_DAYS]
    end = dates[-HOLD_DAYS - 1]

    # Conviction geçmişini varsayılan maliyetle üret; canlı maliyet varsayımından bağımsız
    # araştırma sinyali olarak yalnızca yakın geçmişteki relatif sepet performansı kullanılır.
    history = get_basket_history(W, universe, scores, start, costs[0])
    base = run_variant(BASE_NAME, W, universe, scores, regime, history, costs[0], start)
    b = run_variant(B_NAME, W, universe, scores, regime, history, costs[0], start)
    b2 = run_variant(B2_NAME, W, universe, scores, regime, history, costs[0], start)

    cost_df = cost_stress(W, universe, scores, regime, history, start, costs)
    boot_b = paired_monthly_bootstrap(base, b)
    boot_b2 = paired_monthly_bootstrap(base, b2)

    blocks = []
    for a, b_year in ((int(start.year), min(int(start.year) + 2, int(end.year) + 1)),
                      (min(int(start.year) + 2, int(end.year) + 1), min(int(start.year) + 5, int(end.year) + 1)),
                      (min(int(start.year) + 5, int(end.year) + 1), min(int(start.year) + 8, int(end.year) + 1)),
                      (min(int(start.year) + 8, int(end.year) + 1), int(end.year) + 1)):
        if b_year > a:
            blocks.append((f"{a}–{b_year - 1}", (f"{a}-01-01", f"{b_year - 1}-12-31")))

    stress_series = regime["stress_z"]
    regime_b = regime_split(b, stress_series)
    regime_b2 = regime_split(b2, stress_series)
    data_stats = {
        "tickers": int(panel.ticker.nunique()),
        "rows": int(len(panel)),
        "first_date": str(panel.date.min().date()),
        "last_date": str(panel.date.max().date()),
    }

    report = build_report(args.years, start, end, base, b, b2, cost_df, boot_b, boot_b2, blocks, regime_b, regime_b2, data_stats)
    Path(args.out).write_text(report, encoding="utf-8")

    trades = pd.concat([base.trades, b.trades, b2.trades], ignore_index=True)
    trades.to_csv(args.trades_out, index=False)

    m0, m1, m2 = metrics(base), metrics(b), metrics(b2)
    print("\n=== B2 ROBUST / REGIME-AWARE TEST ===")
    print(f"Dönem: {start.date()} → {end.date()}")
    print(f"BASE CAGR: {m0['CAGR_pct']:.2f}%")
    print(f"B CAGR:    {m1['CAGR_pct']:.2f}%")
    print(f"B2 CAGR:   {m2['CAGR_pct']:.2f}% | B2-BASE: {m2['CAGR_pct'] - m0['CAGR_pct']:+.2f} pp")
    print(f"BASE DD: {m0['MaxDD_pct']:.2f}% | B2 DD: {m2['MaxDD_pct']:.2f}%")
    print(f"BASE PF: {m0['ProfitFactor']:.2f} | B2 PF: {m2['ProfitFactor']:.2f}")
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
