#!/usr/bin/env python3
"""
Global Momentum Quant — B2 Full-System Final Validation

Bu test, B2'yi gerçek Global Momentum Quant portföy motorunun içine alıp
BASE v1.4 ile karşılaştırır. CANLI DOSYALAR DEĞİŞTİRİLMEZ.

Gerçek üretim mekanikleri korunur:
- engine.MarketData
- engine.step_market
- portfolio.process_day
- risk parity
- 4 tranche / 5 günlük kaydırma / 21 işlem günü
- felaket stopu
- nakit getirisi
- BIST + ABD tek TL portföyü

BIST skor farkı yalnızca B2'dedir. ABD tarafı BASE ile aynıdır.
B2 sinyali, önceki bağımsız B2 raporundaki robust Amihud + conviction formülünü
korur. B2'nin dinamik ağırlık yolu temel maliyet (%0.35 RT) ile önceden
hesaplanır ve maliyet stres testlerinde DONDURULUR; böylece maliyet senaryosu
üzerinde yeniden optimizasyon yapılmaz.

ÖNEMLİ:
- engine.py / portfolio.py / config.py dosyaları yazılmaz.
- data/state.json, orders, nav/trades gibi canlı dosyalar yazılmaz.
- Sadece rapor ve test işlem CSV'si oluşturulur.
- ABD evreni repo'daki mevcut üyelik listesiyle çalışır; geçmiş S&P üyelik tarihi
  yoksa survivorship bias devam eder.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data as DA
import engine as E
import portfolio as P

ROOT = Path(__file__).resolve().parent
REPORT_FILE = ROOT / "b2_full_system_validation_raporu.md"
TRADES_FILE = ROOT / "b2_full_system_validation_islemler.csv"

DEFAULT_YEARS = 14
DEFAULT_COSTS = [0.35, 0.50, 0.75, 1.00, 1.25]
WARMUP_DAYS = 260
HOLD_DAYS = 21
N_PICKS = 10
CONVICTION_BASKETS = 12
CONVICTION_SCALE_PCT = 5.0
AMIHUD_NEUTRAL_WEIGHT = 0.25
AMIHUD_MIN_WEIGHT = 0.0
AMIHUD_MAX_WEIGHT = 0.50
REGIME_WINDOW = 126
SEED = 20261004

BASE_BIST_SPEC = [["mom_12_1", 1], ["hi52", 1], ["low_max", 1]]
US_SPEC = None  # config.py içindeki mevcut aktif CANDIDATES yolu kullanılır.


@dataclass
class Scenario:
    name: str
    bist_md: E.MarketData
    us_md: E.MarketData
    daily: pd.Series
    state: dict
    trades: pd.DataFrame


class CostContext:
    def __init__(self, bist_cost: float, us_cost: float):
        self.bist_cost = float(bist_cost)
        self.us_cost = float(us_cost)
        self.old = None

    def __enter__(self):
        self.old = {
            "bist": C.MARKETS["bist"]["cost_rt_pct"],
            "us": C.MARKETS["us"]["cost_rt_pct"],
        }
        C.MARKETS["bist"]["cost_rt_pct"] = self.bist_cost
        C.MARKETS["us"]["cost_rt_pct"] = self.us_cost
        return self

    def __exit__(self, *args):
        C.MARKETS["bist"]["cost_rt_pct"] = self.old["bist"]
        C.MARKETS["us"]["cost_rt_pct"] = self.old["us"]


def _extract_adjusted_bist(W: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """engine.MarketData içindeki BIST kurumsal olay düzeltmesini birebir uygular."""
    c_raw = W["c"]
    ratio = c_raw / c_raw.ffill().shift(1)
    evm = (ratio < 0.75) | (ratio > 1.30)
    f = ratio.where(evm, 1.0).fillna(1.0)
    cum = f.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return {
        k: (W[k] * cum if k in ("o", "h", "l", "c") else W[k] / cum)
        for k in W
    }


def _universe_mask(W: dict[str, pd.DataFrame], liq_pct: float = None) -> pd.DataFrame:
    liq_pct = C.MARKETS["bist"].get("liq_min_pct", C.LIQ_MIN_PCT) if liq_pct is None else liq_pct
    o, h, l, c, v = (W[k] for k in ("o", "h", "l", "c", "v"))
    r = c.pct_change(fill_method=None)
    vt = (c * v).where(v > 0)
    liq = vt.rolling(20, min_periods=10).median()
    valid = c.notna() & (v > 0) & (h >= l) & r.abs().le(C.MARKETS["bist"]["max_move"])
    return valid & liq.notna() & liq.rank(axis=1, pct=True).ge(liq_pct)


def _winsor_cs(x: pd.DataFrame, lo: float = 0.05, hi: float = 0.95) -> pd.DataFrame:
    qlo = x.quantile(lo, axis=1)
    qhi = x.quantile(hi, axis=1)
    return x.clip(lower=qlo, upper=qhi, axis=0)


def build_b2_signal_path(W_raw: dict[str, pd.DataFrame], base_cost: float = 0.35):
    """Önceki bağımsız B2 testindeki formülle B2 score/U matrisini üretir."""
    W = _extract_adjusted_bist(W_raw)
    o, h, l, c, v = (W[k] for k in ("o", "h", "l", "c", "v"))
    U = _universe_mask(W)
    r = c.pct_change(fill_method=None)

    mom = c.shift(21) / c.shift(252) - 1.0
    hi52 = c / c.rolling(252, min_periods=200).max()
    low_max = -r.rolling(21, min_periods=15).max()

    traded_value = (c * v).where(v > 0)
    raw_am = (r.abs() / traded_value.replace(0, np.nan)).rolling(20, min_periods=10).mean()
    robust_am_raw = _winsor_cs(raw_am.where(U)).clip(lower=0)
    robust_am = np.log1p(robust_am_raw).where(U)

    def rank(x):
        return x.where(U).rank(axis=1, pct=True, method="average")

    rm = rank(mom)
    rh = rank(hi52)
    rl = rank(low_max)
    ra = rank(robust_am)

    core = (rm + rh) / 2.0
    b_score = (rm + rh + ra) / 3.0
    base_score = (rm + rh + rl) / 3.0

    market_am = robust_am.where(U).median(axis=1)
    am_mean = market_am.rolling(REGIME_WINDOW, min_periods=60).mean()
    am_std = market_am.rolling(REGIME_WINDOW, min_periods=60).std()
    stress_z = ((market_am - am_mean) / am_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # B2 robust rule: same as previous independent validation.
    # Need a causal B-vs-BASE history. We calculate candidate basket returns once
    # and use only observations whose exit is strictly before the signal date.
    dates = list(pd.DatetimeIndex(c.index))
    rebal_dates = dates[WARMUP_DAYS::HOLD_DAYS]
    history_rows = []
    for signal_date in rebal_dates:
        if signal_date not in c.index:
            continue
        i = dates.index(signal_date)
        entry_i = i + 1
        exit_i = i + HOLD_DAYS
        if exit_i >= len(dates):
            break
        for nm, sc in (("BASE", base_score), ("B", b_score)):
            row = sc.loc[signal_date].where(U.loc[signal_date]).dropna().sort_values(ascending=False, kind="stable")
            picks = list(row.index[:N_PICKS])
            if len(picks) < 7:
                continue
            rr = []
            for t in picks:
                if pd.isna(W_raw["o"].loc[dates[entry_i], t]) or pd.isna(W_raw["c"].loc[dates[exit_i], t]):
                    continue
                entry = float(W_raw["o"].loc[dates[entry_i], t])
                exit_ = float(W_raw["c"].loc[dates[exit_i], t])
                if entry > 0 and exit_ > 0:
                    half = base_cost / 100.0 / 2.0
                    rr.append(((exit_ * (1.0 - half)) / (entry * (1.0 + half)) - 1.0) * 100.0)
            if len(rr) >= 7:
                history_rows.append({
                    "signal_date": signal_date,
                    "exit_date": dates[exit_i],
                    "variant": nm,
                    "basket_ret_pct": float(np.mean(rr)),
                })
    hist = pd.DataFrame(history_rows)

    b2_score = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    weight_rows = []
    for signal_date in c.index:
        prior = hist[hist["exit_date"] < signal_date].sort_values("exit_date")
        pivot = prior.pivot(index="exit_date", columns="variant", values="basket_ret_pct").dropna() if not prior.empty else pd.DataFrame()
        if pivot.empty:
            excess_mean = 0.0
        else:
            ex = pivot["B"] - pivot["BASE"]
            excess_mean = float(ex.tail(CONVICTION_BASKETS).mean()) if len(ex) else 0.0
        conviction = float(np.tanh(excess_mean / CONVICTION_SCALE_PCT))
        stress_signal = float(np.tanh(float(stress_z.loc[signal_date]))) if pd.notna(stress_z.loc[signal_date]) else 0.0
        w = float(np.clip(AMIHUD_NEUTRAL_WEIGHT + 0.25 * stress_signal * conviction,
                          AMIHUD_MIN_WEIGHT, AMIHUD_MAX_WEIGHT))
        score = 0.5 * core.loc[signal_date] + 0.5 * ((1.0 - w) * rl.loc[signal_date] + w * ra.loc[signal_date])
        b2_score.loc[signal_date] = score
        weight_rows.append({"date": signal_date, "amihud_weight": w, "stress_z": stress_z.loc[signal_date], "conviction_excess_pct": excess_mean})

    weights = pd.DataFrame(weight_rows).set_index("date")
    return b2_score, U, base_score, b_score, weights


class PrecomputedMarketData(E.MarketData):
    """Gerçek E.MarketData/engine altyapısını koruyup sadece score/U'yu önceden verir."""
    def __init__(self, mk, W, score_df=None, U_df=None, spec=None):
        super().__init__(mk, W, spec=spec)
        if score_df is not None:
            self.score = score_df.reindex(self.dates).to_numpy(dtype=float)
        if U_df is not None:
            self.U = U_df.reindex(self.dates, columns=W["c"].columns).fillna(False).to_numpy(dtype=bool)


def _fx_at(fx: pd.Series, d):
    return P.fx_at(fx, d)


def simulate_full(name: str, mdb, mdu, fx, us_tbill, start, cost):
    """P.run_backtest mantığının aynısı, sonucu ve işlemleri dışarı verir."""
    with CostContext(cost, cost):
        P.set_hist_rates(us_tbill)
        rb, _ = P.self_financed_returns(mdb)
        ru_usd, _ = P.self_financed_returns(mdu)
        ru = P._to_tl_returns(ru_usd, "us", fx)

        def shadow(mk, d):
            s = rb if mk == "bist" else ru
            return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)

        end = max(mdb.dates[-1], mdu.dates[-1])
        days = sorted(set(d for d in mdb.dates + mdu.dates if pd.Timestamp(start) <= d <= end))
        state = P.new_state(C.CAPITAL_TL, _fx_at(fx, start))
        for d in days:
            if d in mdb.didx:
                P.process_day(state, "bist", mdb, mdb.didx[d], fx, shadow)
            if d in mdu.didx:
                P.process_day(state, "us", mdu, mdu.didx[d], fx, shadow)

        nav = pd.DataFrame(state["pf"]["nav_tl"], columns=["date", "mk", "total_tl", "fx"])
        if nav.empty:
            raise RuntimeError(f"{name}: NAV oluşmadı.")
        nav["date"] = pd.to_datetime(nav["date"])
        daily = nav.groupby("date")["total_tl"].last().sort_index()
        trades = []
        for mk in ("bist", "us"):
            for tr in state["markets"][mk].get("trades", []):
                row = dict(tr)
                row["scenario"] = name
                trades.append(row)
        return Scenario(name, mdb, mdu, daily, state, pd.DataFrame(trades))


def metrics(sc: Scenario):
    nav = sc.daily.dropna().sort_index()
    if len(nav) < 30:
        return {}
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / yrs) - 1.0
    dd = nav / nav.cummax() - 1.0
    tr = sc.trades
    wins = tr.loc[tr.ret_pct > 0, "ret_pct"].sum() if not tr.empty else 0.0
    losses = -tr.loc[tr.ret_pct <= 0, "ret_pct"].sum() if not tr.empty else 0.0
    pf = wins / losses if losses > 0 else np.inf
    months = nav.resample("ME").last().pct_change().dropna()
    worst12 = np.nan
    pos12 = np.nan
    if len(months) >= 12:
        roll12 = (1 + months).rolling(12).apply(np.prod, raw=True) - 1.0
        worst12 = float(roll12.min() * 100.0)
        pos12 = float((roll12 > 0).mean() * 100.0)
    return {
        "CAGR_pct": cagr * 100.0,
        "Total_x": float(nav.iloc[-1] / nav.iloc[0]),
        "MaxDD_pct": float(dd.min() * 100.0),
        "TradeWin_pct": float((tr.ret_pct > 0).mean() * 100.0) if not tr.empty else np.nan,
        "AvgTrade_pct": float(tr.ret_pct.mean()) if not tr.empty else np.nan,
        "ProfitFactor": float(pf),
        "Worst12M_pct": worst12,
        "Positive12M_pct": pos12,
        "Trades": int(len(tr)),
    }


def bootstrap_monthly(base: Scenario, test: Scenario, n_boot=5000):
    a = base.daily.resample("ME").last().pct_change()
    b = test.daily.resample("ME").last().pct_change()
    df = pd.concat([a.rename("base"), b.rename("b2")], axis=1).dropna()
    if len(df) < 12:
        return {"months": len(df), "mean": np.nan, "median": np.nan, "ci5": np.nan, "ci95": np.nan, "p_le0": np.nan, "positive_pct": np.nan}
    diff = (df.b2 - df.base).to_numpy(float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return {
        "months": int(len(diff)),
        "mean": float(diff.mean() * 100),
        "median": float(np.median(diff) * 100),
        "ci5": float(np.quantile(samples, 0.05) * 100),
        "ci95": float(np.quantile(samples, 0.95) * 100),
        "p_le0": float((samples <= 0).mean() * 100),
        "positive_pct": float((diff > 0).mean() * 100),
    }


def block_metrics(sc: Scenario, start: pd.Timestamp, end: pd.Timestamp):
    nav = sc.daily[(sc.daily.index >= start) & (sc.daily.index <= end)].dropna()
    if len(nav) < 20:
        return {"cagr": np.nan, "dd": np.nan, "trades": 0, "win": np.nan, "pf": np.nan}
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    tr = sc.trades.copy()
    if not tr.empty:
        tr["exit_date"] = pd.to_datetime(tr["exit_date"])
        tr = tr[(tr.exit_date >= start) & (tr.exit_date <= end)]
    if tr.empty:
        return {"cagr": cagr * 100, "dd": dd * 100, "trades": 0, "win": np.nan, "pf": np.nan}
    wins = tr.loc[tr.ret_pct > 0, "ret_pct"].sum()
    losses = -tr.loc[tr.ret_pct <= 0, "ret_pct"].sum()
    return {"cagr": cagr * 100, "dd": dd * 100, "trades": len(tr), "win": (tr.ret_pct > 0).mean() * 100, "pf": wins / losses if losses > 0 else np.inf}


def fmt(x, d=2):
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def build_report(years, start, end, results, stress_df, boot, blocks, b2_weights, data_stats):
    m = {k: metrics(v) for k, v in results.items()}
    base = m["BASE"]
    b2 = m["B2"]
    cagr_delta = b2["CAGR_pct"] - base["CAGR_pct"]
    dd_delta = b2["MaxDD_pct"] - base["MaxDD_pct"]
    pf_delta = b2["ProfitFactor"] - base["ProfitFactor"]
    holdout = blocks[-1] if blocks else None
    hold_base = block_metrics(results["BASE"], pd.Timestamp(holdout[1]), pd.Timestamp(holdout[2])) if holdout else {}
    hold_b2 = block_metrics(results["B2"], pd.Timestamp(holdout[1]), pd.Timestamp(holdout[2])) if holdout else {}

    stress_pass = int((stress_df["B2_minus_BASE_CAGR_pp"] > 0).sum()) if not stress_df.empty else 0
    stress_total = len(stress_df)
    robust = (
        cagr_delta >= 0.5
        and dd_delta <= 1.0
        and pf_delta >= -0.05
        and holdout
        and hold_b2.get("cagr", np.nan) >= hold_base.get("cagr", np.nan) - 2.0
        and np.isfinite(boot.get("ci5", np.nan)) and boot["ci5"] > 0
    )
    weak = (
        cagr_delta > 0
        and dd_delta <= 2.0
        and pf_delta >= -0.10
        and stress_pass >= max(1, math.ceil(stress_total / 2))
    )
    verdict = "ROBUST PASS ADAYI" if robust else ("WEAK PASS" if weak else "FAIL / YETERSİZ")

    lines = [
        "# B2 — Full System Final Validation",
        f"_Üretim: {pd.Timestamp.now():%Y-%m-%d %H:%M} · veri: {years} yıl + engine warm-up · ölçüm: {start.date()} → {end.date()}_",
        "",
        "## 1. Karar",
        f"**{verdict}**",
        "",
        "Bu test B2 sinyal yolunu gerçek Global Momentum Quant portföy motoruna bağlar. `engine.MarketData`, `portfolio.process_day`, risk parity, 4 tranche, 21 günlük yenileme, nakit faizi ve felaket stopu üretim motoruyla çalıştırılmıştır.",
        "Canlı dosyalara otomatik entegrasyon yapılmaz.",
        "",
        "## 2. Ana sonuç",
        "",
        "| Metrik | BASE | B2 | B2 − BASE |",
        "|---|---:|---:|---:|",
        f"| CAGR | {fmt(base['CAGR_pct'])}% | **{fmt(b2['CAGR_pct'])}%** | **{fmt(cagr_delta)} pp** |",
        f"| Toplam çarpan | {fmt(base['Total_x'])}x | **{fmt(b2['Total_x'])}x** | {fmt(b2['Total_x']-base['Total_x'])}x |",
        f"| Max DD | {fmt(base['MaxDD_pct'])}% | {fmt(b2['MaxDD_pct'])}% | {fmt(dd_delta)} pp |",
        f"| İşlem win | {fmt(base['TradeWin_pct'])}% | {fmt(b2['TradeWin_pct'])}% | {fmt(b2['TradeWin_pct']-base['TradeWin_pct'])} pp |",
        f"| İşlem başı net | {fmt(base['AvgTrade_pct'])}% | {fmt(b2['AvgTrade_pct'])}% | {fmt(b2['AvgTrade_pct']-base['AvgTrade_pct'])} pp |",
        f"| Profit factor | {fmt(base['ProfitFactor'])} | {fmt(b2['ProfitFactor'])} | {fmt(pf_delta)} |",
        f"| En kötü 12 ay | {fmt(base['Worst12M_pct'])}% | {fmt(b2['Worst12M_pct'])}% | {fmt(b2['Worst12M_pct']-base['Worst12M_pct'])} pp |",
        f"| Pozitif 12 ay | {fmt(base['Positive12M_pct'])}% | {fmt(b2['Positive12M_pct'])}% | {fmt(b2['Positive12M_pct']-base['Positive12M_pct'])} pp |",
        f"| İşlem sayısı | {base['Trades']} | {b2['Trades']} | {b2['Trades']-base['Trades']} |",
        "",
        "## 3. Maliyet stres testi",
        "",
        stress_df.to_markdown(index=False, floatfmt=".2f") if not stress_df.empty else "—",
        "",
        f"B2, {stress_total} maliyet senaryosunun **{stress_pass}/{stress_total}** tanesinde BASE CAGR'ını geçti.",
        "B2 sinyal/weight yolu temel %0.35 RT koşulunda oluşturulmuş ve maliyet stresinde sabit tutulmuştur; bu bölüm yeniden optimizasyon değildir.",
        "",
        "## 4. Aylık eşleştirilmiş bootstrap",
        "",
        f"Ortak ay: **{boot.get('months', 0)}**",
        f"Ortalama aylık B2 − BASE: **{fmt(boot.get('mean'))} pp**",
        f"Medyan: **{fmt(boot.get('median'))} pp**",
        f"%5–%95 CI: **[{fmt(boot.get('ci5'))}, {fmt(boot.get('ci95'))}] pp**",
        f"P(ortalama ≤ 0): **{fmt(boot.get('p_le0'))}%**",
        f"Pozitif ay oranı: **{fmt(boot.get('positive_pct'))}%**",
        "",
        "## 5. Zaman blokları / final holdout",
        "",
        "| Dönem | BASE CAGR | B2 CAGR | B2−BASE | BASE DD | B2 DD | B2 Win |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for label, s, e in blocks:
        rb = block_metrics(results["BASE"], pd.Timestamp(s), pd.Timestamp(e))
        rx = block_metrics(results["B2"], pd.Timestamp(s), pd.Timestamp(e))
        lines.append(
            f"| {label} | {fmt(rb['cagr'])}% | {fmt(rx['cagr'])}% | {fmt(rx['cagr']-rb['cagr'])} pp | {fmt(rb['dd'])}% | {fmt(rx['dd'])}% | {fmt(rx['win'])}% |"
        )

    if not b2_weights.empty:
        lines += [
            "",
            "## 6. B2 ağırlık davranışı",
            "",
            f"Amihud ağırlığı ortalama: **{fmt(b2_weights['amihud_weight'].mean())}**",
            f"Medyan: **{fmt(b2_weights['amihud_weight'].median())}**",
            f"Minimum: **{fmt(b2_weights['amihud_weight'].min())}**",
            f"Maksimum: **{fmt(b2_weights['amihud_weight'].max())}**",
            "",
            "B2 ağırlığı geleceğe bakmadan yalnızca o güne kadar tamamlanmış B-vs-BASE performans farkı ve Amihud stress bilgisinden türetildi.",
        ]

    lines += [
        "",
        "## 7. Veri ve mimari notları",
        "",
        f"- BIST evreni: `{data_stats['bist_tickers']}` hisse",
        f"- ABD evreni: `{data_stats['us_tickers']}` hisse",
        f"- BIST veri satırı: `{data_stats['bist_rows']}`",
        f"- ABD veri satırı: `{data_stats['us_rows']}`",
        f"- USD/TRY ilk/son tarih: `{data_stats['fx_first']}` → `{data_stats['fx_last']}`",
        "- ABD evreninde geçmiş S&P üyelik geçmişi kullanılmadı; survivorship bias sınırlaması devam eder.",
        "- B2 yalnızca BIST sinyal katmanını değiştirir; ABD stratejisi ve portföy mekanikleri aynıdır.",
        "- Bu araştırma `data/state.json`, `data/orders_*.json`, `data/nav.csv`, `data/trades.csv` dosyalarını yazmaz.",
        "",
        "## 8. Karar mantığı",
        "",
        "ROBUST PASS ADAYI koşulları: B2 CAGR +0.5 pp veya daha fazla, MaxDD farkı ≤ +1 pp, PF farkı ≥ -0.05, final holdout CAGR'ı BASE'in 2 pp'den fazla gerisinde değil ve bootstrap %5 alt sınırı > 0.",
        "WEAK PASS daha düşük ama pozitif ve maliyet dayanıklı iyileşmeyi ifade eder. Hiçbir PASS canlı entegrasyonunu otomatikleştirmez.",
        "",
        "**Canlı sistemi değiştirmez.**",
    ]
    return "\n".join(lines) + "\n"


def prepare_data(years: int):
    panels = {}
    panels["bist"] = DA.get_panel("bist", years=years, force_full=True)
    if panels["bist"].empty:
        raise RuntimeError("BIST verisi indirilemedi.")
    panels["us"] = DA.get_panel("us", years=years, force_full=True)
    if panels["us"].empty:
        raise RuntimeError("ABD verisi indirilemedi.")
    fx = DA.get_fx(years)
    if fx.empty:
        raise RuntimeError("USD/TRY verisi indirilemedi.")

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

    return {k: DA.to_wide(v) for k, v in panels.items()}, fx, irx, panels


def active_specs():
    # Mevcut canlı config içindeki default adaylar: her pazarın ilk adayı.
    bist_spec = C.CANDIDATES["bist"][next(iter(C.CANDIDATES["bist"]))]
    us_spec = C.CANDIDATES["us"][next(iter(C.CANDIDATES["us"]))]
    return bist_spec, us_spec


def run_all(years: int, costs: list[float], out_path: Path, trades_path: Path):
    W, fx, irx, raw_panels = prepare_data(years)

    b2_scores, b2_U, base_score, b_score, b2_weights = build_b2_signal_path(W["bist"], base_cost=costs[0])

    bist_spec, us_spec = active_specs()

    # BASE market data: actual engine.
    # B2 market data: actual engine with only precomputed B2 score/U replacing score/U.
    base_bist = E.MarketData("bist", W["bist"], spec=bist_spec)
    b2_bist = PrecomputedMarketData("bist", W["bist"], score_df=b2_scores, U_df=b2_U, spec=bist_spec)
    base_us = E.MarketData("us", W["us"], spec=us_spec)
    b2_us = E.MarketData("us", W["us"], spec=us_spec)

    common_start_idx = max(WARMUP_DAYS, 0)
    start_candidates = [base_bist.dates[common_start_idx], base_us.dates[common_start_idx]]
    start = max(pd.Timestamp(x) for x in start_candidates)
    end = max(base_bist.dates[-1], base_us.dates[-1])

    print(f"🧪 Full system ölçüm: {start.date()} → {end.date()}")

    results = {}
    results["BASE"] = simulate_full("BASE", base_bist, base_us, fx, irx, start, costs[0])
    results["B2"] = simulate_full("B2", b2_bist, b2_us, fx, irx, start, costs[0])

    stress_rows = []
    for cost in costs:
        print(f"💰 Maliyet stres: {cost:.2f}% RT")
        # MarketData fee is reconstructed under current in-memory cost. Scores remain fixed.
        with CostContext(cost, cost):
            b0_bist = PrecomputedMarketData("bist", W["bist"], score_df=base_score, U_df=b2_U, spec=bist_spec)
            bx_bist = PrecomputedMarketData("bist", W["bist"], score_df=b2_scores, U_df=b2_U, spec=bist_spec)
            b0_us = E.MarketData("us", W["us"], spec=us_spec)
            bx_us = b0_us
        # Above objects captured cfg by reference; cost context would restore before use.
        # Rebuild under the active cost and simulate immediately.
        base_s = simulate_full(f"BASE_{cost:g}", b0_bist, b0_us, fx, irx, start, cost)
        b2_s = simulate_full(f"B2_{cost:g}", bx_bist, bx_us, fx, irx, start, cost)
        mb, mx = metrics(base_s), metrics(b2_s)
        stress_rows.append({
            "Maliyet_RT_pct": cost,
            "BASE_CAGR_pct": mb["CAGR_pct"],
            "B2_CAGR_pct": mx["CAGR_pct"],
            "B2_minus_BASE_CAGR_pp": mx["CAGR_pct"] - mb["CAGR_pct"],
            "BASE_MaxDD_pct": mb["MaxDD_pct"],
            "B2_MaxDD_pct": mx["MaxDD_pct"],
            "BASE_PF": mb["ProfitFactor"],
            "B2_PF": mx["ProfitFactor"],
            "BASE_TradeWin_pct": mb["TradeWin_pct"],
            "B2_TradeWin_pct": mx["TradeWin_pct"],
        })

    stress_df = pd.DataFrame(stress_rows)
    boot = bootstrap_monthly(results["BASE"], results["B2"])

    # Three fixed periods. Final block is used as holdout diagnostic.
    y0, y1 = int(start.year), int(end.year)
    blocks = [
        (f"{y0}-{min(y0 + 4, y1)}", f"{y0}-01-01", f"{min(y0 + 4, y1)}-12-31"),
        (f"{min(y0 + 5, y1)}-{min(y0 + 8, y1)}", f"{min(y0 + 5, y1)}-01-01", f"{min(y0 + 8, y1)}-12-31"),
        (f"{min(y0 + 9, y1)}-{y1}", f"{min(y0 + 9, y1)}-01-01", f"{y1}-12-31"),
    ]
    blocks = [b for b in blocks if pd.Timestamp(b[1]) <= pd.Timestamp(b[2])]

    data_stats = {
        "bist_tickers": int(raw_panels["bist"]["ticker"].nunique()),
        "us_tickers": int(raw_panels["us"]["ticker"].nunique()),
        "bist_rows": int(len(raw_panels["bist"])),
        "us_rows": int(len(raw_panels["us"])),
        "fx_first": str(fx.index.min().date()),
        "fx_last": str(fx.index.max().date()),
    }

    report = build_report(years, start, end, results, stress_df, boot, blocks, b2_weights, data_stats)
    out_path.write_text(report, encoding="utf-8")

    all_trades = pd.concat([results["BASE"].trades, results["B2"].trades], ignore_index=True)
    all_trades.to_csv(trades_path, index=False)

    print("\n=== B2 FULL SYSTEM FINAL VALIDATION ===")
    for name in ("BASE", "B2"):
        m = metrics(results[name])
        print(f"{name}: CAGR={m['CAGR_pct']:.2f}% | DD={m['MaxDD_pct']:.2f}% | PF={m['ProfitFactor']:.2f} | Win={m['TradeWin_pct']:.2f}%")
    print(f"Bootstrap CI5={boot['ci5']:.2f} pp | CI95={boot['ci95']:.2f} pp")
    print(f"Rapor: {out_path}")
    print(f"İşlemler: {trades_path}")
    print("Canlı data/state/orders dosyalarına yazılmadı.")


def main() -> int:
    ap = argparse.ArgumentParser(description="B2 full-system final validation")
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--costs", default=",".join(str(x) for x in DEFAULT_COSTS))
    ap.add_argument("--out", default=str(REPORT_FILE))
    ap.add_argument("--trades-out", default=str(TRADES_FILE))
    args = ap.parse_args()
    costs = [float(x.strip()) for x in args.costs.split(",") if x.strip()]
    if args.years < 5:
        raise ValueError("En az 5 yıl önerilir.")
    if not costs:
        raise ValueError("En az bir maliyet seviyesi gerekli.")
    run_all(args.years, costs, Path(args.out), Path(args.trades_out))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"❌ HATA: {exc}", file=sys.stderr)
        raise SystemExit(1)
