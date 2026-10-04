#!/usr/bin/env python3
"""
B Varyantı — bağımsız ikinci doğrulama testi.

AMAÇ
-----
İlk varyant testinin bulduğu BIST varyantını:
    B = mom_12_1 + hi52 + amihud_illiq
ile mevcut temel skorun:
    BASE = mom_12_1 + hi52 + low_max
karşılaştırmak.

Bu dosya bilinçli olarak engine.py / portfolio.py / signals.py kullanmaz.
Böylece ikinci test, ilk backtest motorundan bağımsız bir hesaplama yolu ile
aynı ekonomik hipotezi tekrar sınar.

METODOLOJİ
----------
- BIST evreni: repo'daki data/universe_bist.json
- Veri: yfinance, günlük OHLCV, auto_adjust=True
- Sinyal: kapanıştan sonra hesaplanır; pozisyon bir sonraki işlem gününün açılışında alınır.
- Tutma: 21 işlem günü.
- Yenileme: her 21 işlem gününde bir.
- Pozisyon sayısı: 10, eşit ağırlık.
- Likidite: 20 günlük medyan işlem değerinin alt %40'ı dışarıda bırakılır.
- Max move filtresi: günlük mutlak getiri <= %10.5.
- B: momentum + 52h zirvesi + Amihud illiquidity.
- BASE: momentum + 52h zirvesi + low_max.
- Maliyet: varsayılan %0.35 round-trip; ek olarak stres seviyeleri test edilir.
- Her tarih/portföy karşılaştırmasında iki varyant aynı rebalans tarihleri ve
  aynı uygulanabilir evren üzerinden karşılaştırılır.
- Rapor canlı data/state/orders dosyalarına yazmaz.

ÖNEMLİ SINIR
------------
ABD/S&P üyelik tarihi bu testte hiç kullanılmaz; test yalnızca BIST kolunu
izole eder. Bu, B varyantının BIST faktöründeki sağlamlığını sınamak içindir.
Sonuç başarılı olursa ayrıca birleşik BIST+ABD portföy testine geçilebilir.
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
REPORT_FILE = ROOT / "b_varyant_validasyon_raporu.md"
TRADES_FILE = ROOT / "b_varyant_validasyon_islemler.csv"

DEFAULT_YEARS = 14
DEFAULT_COSTS = [0.35, 0.50, 0.75, 1.00, 1.25]
MAX_MOVE = 0.105
LIQ_PCT = 0.40
HOLD_DAYS = 21
N_PICKS = 10
WARMUP_DAYS = 252
MIN_LIQ_DAYS = 10
SEED = 20261004

BASE_SPEC = ("mom_12_1", "hi52", "low_max")
B_SPEC = ("mom_12_1", "hi52", "amihud_illiq")


@dataclass
class VariantResult:
    name: str
    cost_rt_pct: float
    trades: pd.DataFrame
    portfolio: pd.DataFrame


def _norm_ticker(x: object) -> str:
    return str(x).upper().strip()


def load_universe() -> list[str]:
    if not UNIVERSE_FILE.exists():
        raise FileNotFoundError(f"BIST evren dosyası bulunamadı: {UNIVERSE_FILE}")
    with UNIVERSE_FILE.open("r", encoding="utf-8") as f:
        u = json.load(f)
    tickers = sorted({_norm_ticker(x) for x in (u.get("tickers") or []) if str(x).strip()})
    if len(tickers) < 30:
        raise RuntimeError(f"BIST evreni çok küçük ({len(tickers)} hisse). Veri bozuk olabilir.")
    return tickers


def _yf():
    try:
        import yfinance as yf
        return yf
    except ImportError as exc:
        raise RuntimeError("yfinance kurulu değil. requirements.txt yüklenmeli.") from exc


def download_bist(tickers: list[str], years: int, chunk: int = 50, retries: int = 2) -> pd.DataFrame:
    """Günlük adjusted OHLCV'yi bağımsız olarak indirir."""
    yf = _yf()
    end = pd.Timestamp.now(tz=None).normalize()
    start = (end - pd.DateOffset(years=years + 1)).strftime("%Y-%m-%d")
    rows: list[pd.DataFrame] = []
    symbols = [f"{t}.IS" for t in tickers]

    print(f"📥 BIST doğrulama verisi: {len(symbols)} hisse, yaklaşık {years + 1} yıl")
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
                sub = sub.copy()
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


def _extract_symbol(data: pd.DataFrame, sym: str, single: bool) -> pd.DataFrame | None:
    sub = None
    if isinstance(data.columns, pd.MultiIndex):
        lv0 = set(map(str, data.columns.get_level_values(0)))
        lv1 = set(map(str, data.columns.get_level_values(1)))
        if sym in lv0:
            sub = data[sym]
        elif sym in lv1:
            sub = data.xs(sym, axis=1, level=1)
    elif single:
        sub = data
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
    first = sub.columns[0]
    sub = sub.rename(columns={first: "date"})
    return sub


def make_wide(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for short, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
        x = panel.pivot(index="date", columns="ticker", values=col).sort_index()
        out[short] = x
    cols = out["c"].columns
    return {k: v.reindex(columns=cols) for k, v in out.items()}


def apply_data_filters(W: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tarih x ticker kullanılabilirlik maskesi."""
    o, h, l, c, v = (W[k] for k in ("o", "h", "l", "c", "v"))
    r = c.pct_change(fill_method=None)
    traded_value = (c * v).where((v > 0) & c.notna())
    liq = traded_value.rolling(20, min_periods=MIN_LIQ_DAYS).median()
    valid = c.notna() & o.notna() & (v > 0) & (h >= l) & r.abs().le(MAX_MOVE)
    rank = liq.rank(axis=1, pct=True)
    universe = valid & liq.notna() & rank.ge(LIQ_PCT)
    return universe


def _rank_pct(x: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return x.where(mask).rank(axis=1, pct=True)


def compute_scores(W: dict[str, pd.DataFrame], universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    o, h, l, c, v = (W[k] for k in ("o", "h", "l", "c", "v"))
    r = c.pct_change(fill_method=None)

    mom = c.shift(21) / c.shift(252) - 1.0
    hi52 = c / c.rolling(252, min_periods=200).max()
    low_max = -r.rolling(21, min_periods=15).max()

    traded_value = (c * v).where(v > 0)
    amihud = (r.abs() / traded_value.replace(0, np.nan)).rolling(20, min_periods=MIN_LIQ_DAYS).mean()

    scores = {}
    for name, components in (("BASE", (mom, hi52, low_max)), ("B", (mom, hi52, amihud))):
        ranked = [_rank_pct(x, universe) for x in components]
        s = sum(ranked) / len(ranked)
        scores[name] = s
    return scores


def _pick_top(score_row: pd.Series, mask_row: pd.Series) -> list[str]:
    s = score_row.where(mask_row).dropna().sort_values(ascending=False, kind="stable")
    return list(s.index[:N_PICKS])


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return float("nan")
    return b / a - 1.0


def _net_return(entry: float, exit_: float, cost_rt_pct: float) -> float:
    """Round-trip maliyeti iki yana simetrik uygular."""
    half = (cost_rt_pct / 100.0) / 2.0
    entry_fill = entry * (1.0 + half)
    exit_fill = exit_ * (1.0 - half)
    if entry_fill <= 0:
        return float("nan")
    return exit_fill / entry_fill - 1.0


def run_variant(
    name: str,
    W: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    cost_rt_pct: float,
    start: pd.Timestamp,
) -> VariantResult:
    dates = list(pd.DatetimeIndex(W["c"].index))
    eligible_dates = [d for d in dates if d >= start]
    if len(eligible_dates) <= HOLD_DAYS + 5:
        raise RuntimeError("Test dönemi çok kısa.")

    rebal_dates = eligible_dates[::HOLD_DAYS]
    trades = []
    portfolio_rows = []

    capital = 1.0
    equity_dates = []
    equity_values = []
    previous_exit_date = None

    for signal_date in rebal_dates:
        try:
            sig_i = dates.index(signal_date)
        except ValueError:
            continue
        entry_i = sig_i + 1
        exit_i = sig_i + HOLD_DAYS
        if entry_i >= len(dates) or exit_i >= len(dates):
            break

        entry_date = dates[entry_i]
        exit_date = dates[exit_i]
        picks = _pick_top(scores[name].loc[signal_date], universe.loc[signal_date])
        if len(picks) < N_PICKS:
            continue

        basket = []
        for ticker in picks:
            if ticker not in W["o"].columns:
                continue
            entry = float(W["o"].loc[entry_date, ticker]) if pd.notna(W["o"].loc[entry_date, ticker]) else np.nan
            exit_ = float(W["c"].loc[exit_date, ticker]) if pd.notna(W["c"].loc[exit_date, ticker]) else np.nan
            raw_ret = _safe_ratio(entry, exit_)
            if not np.isfinite(raw_ret):
                continue
            net_ret = _net_return(entry, exit_, cost_rt_pct)
            basket.append((ticker, entry, exit_, raw_ret, net_ret))

        if len(basket) < max(5, math.ceil(N_PICKS * 0.7)):
            continue

        basket_rets = [x[4] for x in basket]
        basket_ret = float(np.mean(basket_rets))
        capital *= 1.0 + basket_ret

        for ticker, entry, exit_, raw_ret, net_ret in basket:
            trades.append({
                "variant": name,
                "signal_date": signal_date.date().isoformat(),
                "entry_date": entry_date.date().isoformat(),
                "exit_date": exit_date.date().isoformat(),
                "ticker": ticker,
                "entry_px": entry,
                "exit_px": exit_,
                "raw_ret_pct": raw_ret * 100.0,
                "net_ret_pct": net_ret * 100.0,
                "cost_rt_pct": cost_rt_pct,
            })

        # Eşit ağırlıklı sepetin step NAV'ı. Portföy tarihi tutarlı ve non-overlap.
        equity_dates.append(exit_date)
        equity_values.append(capital)
        portfolio_rows.append({
            "signal_date": signal_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "basket_ret_pct": basket_ret * 100.0,
            "nav": capital,
            "n_positions": len(basket),
            "hold_days": HOLD_DAYS,
        })
        previous_exit_date = exit_date

    trades_df = pd.DataFrame(trades)
    pf = pd.DataFrame(portfolio_rows)
    if pf.empty:
        raise RuntimeError(f"{name}: hiç tamamlanmış sepet oluşmadı.")
    pf["exit_date"] = pd.to_datetime(pf["exit_date"])
    pf = pf.sort_values("exit_date").reset_index(drop=True)
    return VariantResult(name, cost_rt_pct, trades_df, pf)


def metrics(result: VariantResult) -> dict[str, float]:
    pf = result.portfolio.copy()
    tr = result.trades
    nav = pd.Series(pf["nav"].values, index=pf["exit_date"])
    first_nav = float(nav.iloc[0])
    last_nav = float(nav.iloc[-1])
    span_days = max((nav.index[-1] - nav.index[0]).days, 1)
    years = span_days / 365.25
    cagr = (last_nav / first_nav) ** (1.0 / years) - 1.0 if years > 0 and last_nav > 0 else np.nan
    dd = nav / nav.cummax() - 1.0
    basket_win = pf["basket_ret_pct"] > 0
    trade_win = tr["net_ret_pct"] > 0 if not tr.empty else pd.Series(dtype=bool)
    gains = float(tr.loc[tr["net_ret_pct"] > 0, "net_ret_pct"].sum()) if not tr.empty else 0.0
    losses = float(-tr.loc[tr["net_ret_pct"] <= 0, "net_ret_pct"].sum()) if not tr.empty else 0.0
    pfactor = gains / losses if losses > 0 else np.inf
    monthly = pf.set_index("exit_date")["basket_ret_pct"].resample("ME").apply(lambda x: (np.prod(1 + x / 100.0) - 1.0) * 100.0 if len(x) else np.nan).dropna()
    worst12 = None
    if len(monthly) >= 12:
        roll12 = (1 + monthly / 100.0).rolling(12).apply(np.prod, raw=True) - 1.0
        worst12 = float(roll12.min() * 100.0)

    return {
        "CAGR_pct": cagr * 100.0,
        "Total_x": last_nav / first_nav,
        "MaxDD_pct": float(dd.min() * 100.0),
        "BasketWin_pct": float(basket_win.mean() * 100.0),
        "TradeWin_pct": float(trade_win.mean() * 100.0) if len(trade_win) else np.nan,
        "AvgTrade_pct": float(tr["net_ret_pct"].mean()) if not tr.empty else np.nan,
        "ProfitFactor": pfactor,
        "Worst12M_pct": worst12 if worst12 is not None else np.nan,
        "Baskets": int(len(pf)),
        "Trades": int(len(tr)),
        "MedianBasket_pct": float(pf["basket_ret_pct"].median()),
    }


def block_metrics(result: VariantResult, start: str, end: str) -> dict[str, float]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    pf = result.portfolio[(result.portfolio["exit_date"] >= s) & (result.portfolio["exit_date"] <= e)].copy()
    tr = result.trades.copy()
    if tr.empty:
        return {"baskets": 0, "cagr": np.nan, "dd": np.nan, "win": np.nan, "pf": np.nan}
    tr["exit_date"] = pd.to_datetime(tr["exit_date"])
    tr = tr[(tr.exit_date >= s) & (tr.exit_date <= e)]
    if pf.empty:
        return {"baskets": 0, "cagr": np.nan, "dd": np.nan, "win": float((tr.net_ret_pct > 0).mean()) if len(tr) else np.nan, "pf": np.nan}
    nav = pd.Series(np.r_[1.0, (1.0 + pf["basket_ret_pct"].to_numpy() / 100.0).cumprod()], index=[pf.exit_date.iloc[0] - pd.Timedelta(days=1)] + list(pf.exit_date))
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    dd = (nav / nav.cummax() - 1).min()
    wins = tr.loc[tr.net_ret_pct > 0, "net_ret_pct"].sum() if len(tr) else 0
    losses = -tr.loc[tr.net_ret_pct <= 0, "net_ret_pct"].sum() if len(tr) else 0
    pfactor = wins / losses if losses > 0 else np.inf
    return {"baskets": int(len(pf)), "cagr": cagr * 100, "dd": dd * 100, "win": float((tr.net_ret_pct > 0).mean() * 100) if len(tr) else np.nan, "pf": float(pfactor)}


def paired_monthly_bootstrap(base: VariantResult, test: VariantResult, n_boot: int = 5000, seed: int = SEED) -> dict[str, float]:
    """Aynı takvimde B-BASE aylık farkını bootstrap eder."""
    def monthly(r: VariantResult) -> pd.Series:
        p = r.portfolio.set_index("exit_date")["basket_ret_pct"]
        return ((1 + p / 100.0).resample("ME").prod() - 1.0) * 100.0

    a = monthly(base)
    b = monthly(test)
    df = pd.concat([a.rename("base"), b.rename("b")], axis=1).dropna()
    if len(df) < 12:
        return {"months": int(len(df)), "mean_diff": np.nan, "median_diff": np.nan, "ci5": np.nan, "ci95": np.nan, "p_b_le_base": np.nan, "win_month_pct": np.nan}
    diff = (df["b"] - df["base"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sample_mean = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {
        "months": int(len(diff)),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
        "ci5": float(np.quantile(sample_mean, 0.05)),
        "ci95": float(np.quantile(sample_mean, 0.95)),
        "p_b_le_base": float((sample_mean <= 0).mean()),
        "win_month_pct": float((diff > 0).mean() * 100.0),
    }


def cost_summary(
    W: dict[str, pd.DataFrame],
    universe: pd.DataFrame,
    scores: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    costs: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for cost in costs:
        base = run_variant("BASE", W, universe, scores, cost, start)
        test = run_variant("B", W, universe, scores, cost, start)
        mb = metrics(base)
        mt = metrics(test)
        rows.append({
            "Maliyet_RT_pct": cost,
            "BASE_CAGR_pct": mb["CAGR_pct"],
            "B_CAGR_pct": mt["CAGR_pct"],
            "B_minus_BASE_CAGR_pp": mt["CAGR_pct"] - mb["CAGR_pct"],
            "BASE_MaxDD_pct": mb["MaxDD_pct"],
            "B_MaxDD_pct": mt["MaxDD_pct"],
            "BASE_PF": mb["ProfitFactor"],
            "B_PF": mt["ProfitFactor"],
            "BASE_TradeWin_pct": mb["TradeWin_pct"],
            "B_TradeWin_pct": mt["TradeWin_pct"],
        })
    return pd.DataFrame(rows)


def format_num(x, digits=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (np.floating, float)):
        return f"{x:.{digits}f}"
    return str(x)


def build_report(
    years: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    base: VariantResult,
    test: VariantResult,
    cost_df: pd.DataFrame,
    bootstrap: dict[str, float],
    blocks: list[tuple[str, str]],
    data_stats: dict[str, int | float | str],
) -> str:
    mb = metrics(base)
    mt = metrics(test)
    delta = mt["CAGR_pct"] - mb["CAGR_pct"]
    dd_delta = mt["MaxDD_pct"] - mb["MaxDD_pct"]

    stress_positive = int((cost_df["B_minus_BASE_CAGR_pp"] > 0).sum())
    stress_total = len(cost_df)
    if delta >= 5 and stress_positive >= math.ceil(stress_total * 0.6) and bootstrap.get("ci5", -np.inf) > 0:
        verdict = "PASS"
    elif delta > 0 and stress_positive >= math.ceil(stress_total * 0.5):
        verdict = "WEAK PASS"
    else:
        verdict = "FAIL / YETERSİZ"

    lines = [
        "# B Varyantı — Bağımsız İkinci Doğrulama Raporu",
        f"_Test üretimi: {pd.Timestamp.now():%Y-%m-%d %H:%M} · veri penceresi: {years} yıl · ölçüm: {start.date()} → {end.date()}_",
        "",
        "## 1. Sonuç",
        f"**Karar: {verdict}**",
        "",
        "Bu test, ilk backtestte kullanılan `engine.py`, `portfolio.py` ve `signals.py` modüllerini çağırmadan aynı BIST hipotezini bağımsız bir eşit-ağırlıklı sepet motoru ile sınar.",
        "",
        "## 2. Ana karşılaştırma",
        "",
        "| Metrik | BASE | B | B − BASE |",
        "|---|---:|---:|---:|",
        f"| CAGR | {format_num(mb['CAGR_pct'])}% | **{format_num(mt['CAGR_pct'])}%** | **{format_num(delta)} puan** |",
        f"| Toplam çarpan | {format_num(mb['Total_x'])}x | **{format_num(mt['Total_x'])}x** | {format_num(mt['Total_x'] - mb['Total_x'])}x |",
        f"| Max drawdown | {format_num(mb['MaxDD_pct'])}% | {format_num(mt['MaxDD_pct'])}% | {format_num(dd_delta)} puan |",
        f"| Sepet win rate | {format_num(mb['BasketWin_pct'])}% | {format_num(mt['BasketWin_pct'])}% | {format_num(mt['BasketWin_pct'] - mb['BasketWin_pct'])} puan |",
        f"| İşlem win rate | {format_num(mb['TradeWin_pct'])}% | {format_num(mt['TradeWin_pct'])}% | {format_num(mt['TradeWin_pct'] - mb['TradeWin_pct'])} puan |",
        f"| İşlem başı net | {format_num(mb['AvgTrade_pct'])}% | {format_num(mt['AvgTrade_pct'])}% | {format_num(mt['AvgTrade_pct'] - mb['AvgTrade_pct'])} puan |",
        f"| Profit factor | {format_num(mb['ProfitFactor'])} | {format_num(mt['ProfitFactor'])} | {format_num(mt['ProfitFactor'] - mb['ProfitFactor'])} |",
        f"| En kötü 12 ay | {format_num(mb['Worst12M_pct'])}% | {format_num(mt['Worst12M_pct'])}% | {format_num(mt['Worst12M_pct'] - mb['Worst12M_pct'])} puan |",
        f"| Sepet sayısı | {mb['Baskets']} | {mt['Baskets']} | {mt['Baskets'] - mb['Baskets']} |",
        "",
        "## 3. Maliyet stres testi",
        "",
        cost_df.to_markdown(index=False, floatfmt=".2f"),
        "",
        f"B, {stress_total} maliyet senaryosunun **{stress_positive}/{stress_total}** tanesinde BASE CAGR'ından yüksek çıktı.",
        "",
        "## 4. Eşleştirilmiş aylık bootstrap",
        "",
        f"Ortak ay sayısı: **{bootstrap.get('months', 0)}**",
        f"Ortalama aylık B − BASE farkı: **{format_num(bootstrap.get('mean_diff'))} puan**",
        f"Medyan aylık B − BASE farkı: **{format_num(bootstrap.get('median_diff'))} puan**",
        f"Bootstrap %5–%95 güven aralığı: **[{format_num(bootstrap.get('ci5'))}, {format_num(bootstrap.get('ci95'))}] puan**",
        f"Bootstrap örnekleminde B ≤ BASE olasılığı: **{format_num(bootstrap.get('p_b_le_base') * 100 if np.isfinite(bootstrap.get('p_b_le_base', np.nan)) else np.nan)}%**",
        f"B'nin pozitif aylık fark ürettiği aylar: **{format_num(bootstrap.get('win_month_pct'))}%**",
        "",
        "## 5. Zaman blokları",
        "",
        "| Dönem | BASE CAGR | B CAGR | B − BASE | BASE DD | B DD | B işlem win |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, (s, e) in zip([x[0] for x in blocks], blocks):
        bb = block_metrics(base, s, e)
        bt = block_metrics(test, s, e)
        lines.append(
            f"| {label} | {format_num(bb['cagr'])}% | {format_num(bt['cagr'])}% | {format_num(bt['cagr'] - bb['cagr'])} pp | "
            f"{format_num(bb['dd'])}% | {format_num(bt['dd'])}% | {format_num(bt['win'])}% |"
        )

    lines += [
        "",
        "## 6. Veri denetimi",
        "",
        f"- Evren: `{data_stats['tickers']}` BIST hissesi",
        f"- Veri satırı: `{data_stats['rows']}`",
        f"- İlk tarih: `{data_stats['first_date']}`",
        f"- Son tarih: `{data_stats['last_date']}`",
        f"- Ölçüm dönemi: `{start.date()}` → `{end.date()}`",
        f"- Sinyal ısınması: `{WARMUP_DAYS}` işlem günü",
        "- Yatırım anı: sinyal kapanışından sonraki işlem gününün açılışı",
        "- Çıkış: 21 işlem günü sonrasının kapanışı",
        "- Pozisyonlar: en fazla 10 hisse, eşit ağırlık",
        "- ABD evreni bu testte kullanılmadı; amaç BIST faktörünü izole etmek.",
        "",
        "## 7. Yorum",
        "",
        "B varyantının ilk testteki üstünlüğünün bu bağımsız kurulumda da korunması beklenir; ancak bu tek başına canlıya alma kanıtı değildir.",
        "Özellikle yüksek survivorship bias, veri/kurumsal olay düzeltmeleri ve işlem maliyeti varsayımları ayrıca incelenmelidir.",
        "",
        "### Kullanılan hipotezler",
        "- BASE: `mom_12_1 + hi52 + low_max`",
        "- B: `mom_12_1 + hi52 + amihud_illiq`",
        "- Amihud: 20 günlük ortalamaya dayalı günlük |ret| / (fiyat × hacim)",
        "",
        "**Canlı sistemi değiştirmez.** Bu rapor yalnızca araştırma çıktısıdır.",
    ]
    return "\n".join(lines) + "\n"


def synthetic_panel(years: int = 8, n_tickers: int = 80, seed: int = SEED) -> pd.DataFrame:
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


def parse_costs(s: str) -> list[float]:
    vals = []
    for p in s.split(","):
        x = float(p.strip())
        if x < 0 or x > 10:
            raise ValueError("Maliyet senaryosu 0 ile 10% round-trip arasında olmalı.")
        vals.append(x)
    if not vals:
        raise ValueError("En az bir maliyet seviyesi gerekli.")
    return vals


def run_synthetic() -> None:
    panel = synthetic_panel()
    W = make_wide(panel)
    U = apply_data_filters(W)
    S = compute_scores(W, U)
    start = W["c"].index[WARMUP_DAYS]
    b = run_variant("BASE", W, U, S, 0.35, start)
    x = run_variant("B", W, U, S, 0.35, start)
    print("SENTETİK SINAMA — gerçek performans değildir")
    print(pd.DataFrame([metrics(b), metrics(x)], index=["BASE", "B"]).T.to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description="B varyantı bağımsız ikinci doğrulama testi")
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--costs", default=",".join(str(x) for x in DEFAULT_COSTS), help="RT maliyet yüzdeleri, virgülle")
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
    scores = compute_scores(W, universe)

    dates = pd.DatetimeIndex(W["c"].index)
    if len(dates) <= WARMUP_DAYS + HOLD_DAYS + 10:
        raise RuntimeError("Yeterli geçmiş oluşmadı.")
    start = dates[WARMUP_DAYS]
    end = dates[-HOLD_DAYS - 1]

    base = run_variant("BASE", W, universe, scores, costs[0], start)
    test = run_variant("B", W, universe, scores, costs[0], start)

    cost_df = cost_summary(W, universe, scores, start, costs)
    bootstrap = paired_monthly_bootstrap(base, test)

    # Sabit takvim blokları; mümkün olduğunca tam yıllık ayrımlar.
    blocks = []
    start_year = int(start.year)
    end_year = int(end.year)
    for a, b in ((start_year, min(start_year + 2, end_year)),
                 (min(start_year + 2, end_year), min(start_year + 5, end_year)),
                 (min(start_year + 5, end_year), min(start_year + 8, end_year)),
                 (min(start_year + 8, end_year), end_year + 1)):
        if b <= a:
            continue
        blocks.append((f"{a}–{b - 1}", (f"{a}-01-01", f"{b}-12-31")))

    data_stats = {
        "tickers": int(panel["ticker"].nunique()),
        "rows": int(len(panel)),
        "first_date": str(panel["date"].min().date()),
        "last_date": str(panel["date"].max().date()),
    }

    report = build_report(
        years=args.years,
        start=start,
        end=end,
        base=base,
        test=test,
        cost_df=cost_df,
        bootstrap=bootstrap,
        blocks=blocks,
        data_stats=data_stats,
    )

    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")

    all_trades = pd.concat([base.trades, test.trades], ignore_index=True)
    trade_path = Path(args.trades_out)
    all_trades.to_csv(trade_path, index=False)

    mb = metrics(base)
    mt = metrics(test)
    print("\n=== B VARYANTI BAĞIMSIZ DOĞRULAMA ===")
    print(f"Dönem: {start.date()} → {end.date()}")
    print(f"BASE CAGR: {mb['CAGR_pct']:.2f}% | B CAGR: {mt['CAGR_pct']:.2f}% | fark: {mt['CAGR_pct'] - mb['CAGR_pct']:+.2f} pp")
    print(f"BASE MaxDD: {mb['MaxDD_pct']:.2f}% | B MaxDD: {mt['MaxDD_pct']:.2f}%")
    print(f"BASE PF: {mb['ProfitFactor']:.2f} | B PF: {mt['ProfitFactor']:.2f}")
    print(f"Rapor: {out_path}")
    print(f"İşlemler: {trade_path}")
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
