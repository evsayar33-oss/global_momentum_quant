#!/usr/bin/env python3
"""
Global Momentum Quant — Master Improvement Research

Amaç
----
Mevcut BASE stratejisini değiştirmeden, tek çalıştırmada şu araştırma yönlerini
bağımsız olarak tarar:

1) SIGNAL QUALITY
   BASE seçimini korur; seçilmiş adayların içinden trend-persistence + düşük
   volatilite kalitesini ölçen hafif bir kalite katmanı test eder.

2) ADAPTIVE RISK ALLOCATION
   Hisse seçimleri BASE ile aynıdır. Sadece BIST/ABD sermaye dağılımı mevcut
   inverse-vol risk parity'ye ek olarak son dönem drawdown riskini dikkate alır.

3) REGIME RISK OVERLAY
   Hisse seçimini BASE ile aynı tutar. Piyasa stresi arttığında yeni risk
   ve/veya mevcut pozisyon maruziyeti soft şekilde azaltılır. Hard "sinyali
   kapat" filtresi kullanılmaz.

4) REGIME-SPECIFIC ADAPTIVE SELECTION
   BASE ile kalite skoru arasında sabit olmayan, yalnızca mevcut güne kadar
   bilinen piyasa-stress bilgisinden türetilen bir blend yapılır. Amaç,
   rejime göre seçimin değişmesinin ekstra katkı üretip üretmediğini görmek.

5) ROBUSTNESS / DATA VALIDATION
   Yukarıdaki dört adayın hepsine ortak maliyet stresleri, bootstrap, dönem
   blokları, final holdout ve veri/survivorship denetimi uygulanır.

ÖNEMLİ
------
- Canlı config.py / engine.py / portfolio.py / signals.py DEĞİŞTİRİLMEZ.
- data/state.json, orders, nav.csv, trades.csv gibi canlı durum dosyaları yazılmaz.
- Yalnızca araştırma raporu üretilir.
- Araştırma adayları önceden tanımlı hipotezlerdir; test sonucu görüldükten
  sonra parametre optimizasyonu yapılmaz.
- Her aday BASE'e karşı bağımsızdır; adaylar birbirine zincirlenmez.
- Final OOS sonuçları optimizasyon için kullanılmaz.
- ABD evreni repo'nun mevcut üyelik listesi ile çalıştığından historical
  S&P membership kullanılmıyorsa survivorship bias sınırlaması raporlanır.

Bu script "live integration" yapmaz. Çıktı yalnızca araştırma/karar raporudur.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import config as C
import data as DA
import engine as E
import portfolio as P


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / "gmq_master_improvement_raporu.md"

DEFAULT_YEARS = 14
DEFAULT_COSTS = [0.35, 0.50, 0.75, 1.00, 1.25]
BASE_COST = 0.35

WARMUP_DAYS = 260
N_BOOT = 5000
SEED = 20261005

# Önceden sabitlenmiş / optimize edilmeyen araştırma katsayıları.
QUALITY_BLEND = 0.20
REGIME_QUALITY_MIN = 0.10
REGIME_QUALITY_MAX = 0.40

# Soft regime overlay: C.INS_EXPOSURE = portföyde kalan maruziyet.
OVERLAY_EXPOSURE = 0.60
OVERLAY_TRIGGER = 0.70
OVERLAY_RELEASE = 0.35

# Karar eşikleri: araştırma için önceden tanımlanır, sonuçlara göre oynatılmaz.
MIN_CAGR_EDGE_PP = 0.25
MAX_ALLOWED_CAGR_LOSS_PP = 0.50
MAX_ALLOWED_DD_WORSEN_PP = 1.00
MIN_DD_IMPROVEMENT_PP = 2.00
MAX_PF_LOSS = 0.05
MAX_HOLDOUT_LAG_PP = 2.00
MIN_STRESS_GOOD = 3


@dataclass
class Scenario:
    name: str
    daily: pd.Series
    state: dict
    trades: pd.DataFrame


class CostClone:
    """Bir MarketData kopyasının yalnızca işlem maliyetini değiştirir."""

    @staticmethod
    def md_with_cost(md: E.MarketData, cost_rt_pct: float) -> E.MarketData:
        out = copy.copy(md)
        out.fc = float(cost_rt_pct) / 200.0
        return out


def _extract_adjusted_bist(W: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """engine.MarketData içindeki BIST olay düzeltmesini birebir uygular."""
    c_raw = W["c"]
    ratio = c_raw / c_raw.ffill().shift(1)
    evm = (ratio < 0.75) | (ratio > 1.30)
    f = ratio.where(evm, 1.0).fillna(1.0)
    cum = f.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return {
        k: (W[k] * cum if k in ("o", "h", "l", "c") else W[k] / cum)
        for k in W
    }


def _winsor_cs(x: pd.DataFrame, lo: float = 0.05, hi: float = 0.95) -> pd.DataFrame:
    qlo = x.quantile(lo, axis=1)
    qhi = x.quantile(hi, axis=1)
    return x.clip(lower=qlo, upper=qhi, axis=0)


def _rank_cs(x: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    return x.where(mask).rank(axis=1, pct=True, method="average")


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def build_bist_research_scores(
    W_raw: dict[str, pd.DataFrame],
    base_dates: list[pd.Timestamp],
    base_U_df: pd.DataFrame,
):
    """
    BASE + Signal Quality + Regime Adaptive Selection skorlarını hazırlar.

    Özellikler yalnızca t ve öncesine bakar.
    """
    W = _extract_adjusted_bist(W_raw)
    c = W["c"]
    r = c.pct_change(fill_method=None)

    U = base_U_df.reindex(index=c.index, columns=c.columns).fillna(False)

    mom = c.shift(21) / c.shift(252) - 1.0
    hi52 = c / c.rolling(252, min_periods=200).max()
    low_max = -r.rolling(21, min_periods=15).max()

    # Signal Quality: mevcut signals.py'deki kanıtlı/üretim kodunda bulunan iki
    # bağımsız kalite boyutu: düşük gerçekleşen volatilite ve pozitif gün oranı.
    low_vol = -r.rolling(60, min_periods=40).std()
    pos_days = (r > 0).astype(float).where(r.notna()).rolling(
        252, min_periods=150
    ).mean()

    rm = _rank_cs(mom, U)
    rh = _rank_cs(hi52, U)
    rl = _rank_cs(low_max, U)
    rqv = _rank_cs(low_vol, U)
    rqp = _rank_cs(pos_days, U)

    base_score = ((rm + rh + rl) / 3.0).where(U)

    quality_component = ((rqv + rqp) / 2.0).where(U)
    quality_score = (
        (1.0 - QUALITY_BLEND) * base_score
        + QUALITY_BLEND * quality_component
    ).where(U)

    # Rejim için candidate-independent market stress:
    # BIST cross-sectional median return + rolling volatility + negative breadth.
    market_ret_bist = r.where(U).median(axis=1)
    vol20 = market_ret_bist.rolling(20, min_periods=15).std()
    vol126_med = vol20.rolling(126, min_periods=60).median()
    vol_ratio = _safe_div(vol20, vol126_med).replace([np.inf, -np.inf], np.nan)

    trend63 = market_ret_bist.rolling(63, min_periods=40).sum()
    negative_breadth = (r.where(U) < 0).mean(axis=1)

    vol_stress = ((vol_ratio - 1.0) / 0.75).clip(0.0, 1.0)
    draw_stress = (-trend63 / 0.20).clip(0.0, 1.0)
    breadth_stress = ((negative_breadth - 0.50) / 0.35).clip(0.0, 1.0)

    bist_stress = (
        0.45 * vol_stress.fillna(0.0)
        + 0.35 * draw_stress.fillna(0.0)
        + 0.20 * breadth_stress.fillna(0.0)
    ).clip(0.0, 1.0)

    # Regime adaptive selection:
    # normal rejimde kalite etkisi %10, stress yükselince en fazla %40.
    regime_q_weight = (
        REGIME_QUALITY_MIN
        + (REGIME_QUALITY_MAX - REGIME_QUALITY_MIN) * bist_stress
    ).clip(REGIME_QUALITY_MIN, REGIME_QUALITY_MAX)

    regime_score = (
        (1.0 - regime_q_weight).to_numpy()[:, None] * base_score.to_numpy()
        + regime_q_weight.to_numpy()[:, None] * quality_component.to_numpy()
    )
    regime_score = pd.DataFrame(
        regime_score, index=c.index, columns=c.columns
    ).where(U)

    base_score = base_score.reindex(index=base_dates)
    quality_score = quality_score.reindex(index=base_dates)
    regime_score = regime_score.reindex(index=base_dates)
    U_out = U.reindex(index=base_dates, columns=base_U_df.columns)

    regime_meta = pd.DataFrame(
        {
            "bist_stress": bist_stress,
            "regime_quality_weight": regime_q_weight,
            "bist_trend63": trend63,
            "bist_vol_ratio": vol_ratio,
            "bist_negative_breadth": negative_breadth,
        }
    ).reindex(index=base_dates)

    return {
        "BASE": base_score,
        "SIGNAL_QUALITY": quality_score,
        "REGIME_ADAPTIVE": regime_score,
        "U": U_out,
        "regime_meta": regime_meta,
    }


def build_market_regime_context(
    W_bist: dict[str, pd.DataFrame],
    W_us: dict[str, pd.DataFrame],
    bist_U: pd.DataFrame,
    us_U: pd.DataFrame,
    dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """
    Candidate-independent, point-in-time market stress haritası.

    BIST + ABD cross-sectional median return, volatility ratio ve negative
    breadth kullanılır. Her t gözlemi yalnızca <=t verisinden hesaplanır.
    """
    rb = W_bist["c"].pct_change(fill_method=None).where(bist_U).median(axis=1)
    ru = W_us["c"].pct_change(fill_method=None).where(us_U).median(axis=1)

    merged = pd.concat(
        [
            rb.rename("bist"),
            ru.rename("us"),
        ],
        axis=1,
    )
    market_ret = merged.mean(axis=1, skipna=True)

    vol20 = market_ret.rolling(20, min_periods=15).std()
    vol126_med = vol20.rolling(126, min_periods=60).median()
    vol_ratio = _safe_div(vol20, vol126_med).replace([np.inf, -np.inf], np.nan)

    trend63 = market_ret.rolling(63, min_periods=40).sum()

    neg_bist = (
        W_bist["c"].pct_change(fill_method=None).where(bist_U) < 0
    ).mean(axis=1)
    neg_us = (
        W_us["c"].pct_change(fill_method=None).where(us_U) < 0
    ).mean(axis=1)
    neg_breadth = pd.concat(
        [neg_bist.rename("bist"), neg_us.rename("us")], axis=1
    ).mean(axis=1, skipna=True)

    vol_stress = ((vol_ratio - 1.0) / 0.75).clip(0.0, 1.0)
    draw_stress = (-trend63 / 0.20).clip(0.0, 1.0)
    breadth_stress = ((neg_breadth - 0.50) / 0.35).clip(0.0, 1.0)

    stress = (
        0.45 * vol_stress.fillna(0.0)
        + 0.35 * draw_stress.fillna(0.0)
        + 0.20 * breadth_stress.fillna(0.0)
    ).clip(0.0, 1.0)

    out = pd.DataFrame(
        {
            "stress": stress,
            "market_trend63": trend63,
            "vol_ratio": vol_ratio,
            "negative_breadth": neg_breadth,
        }
    )
    return out.reindex(index=dates)


def adaptive_rp_weights(
    ret_bist_tl: pd.Series,
    ret_us_tl: pd.Series,
):
    """
    Adaptive Risk Allocation:
      baseline = inverse volatility
      overlay  = son 63g drawdown cezası

    Sadece P.process_day tarafından verilen geçmiş shadow serisini kullanır.
    Gelecek veri içermez.
    """
    df = pd.concat(
        [ret_bist_tl.rename("bist"), ret_us_tl.rename("us")],
        axis=1,
    ).dropna().tail(126)

    if len(df) < 40:
        return dict(C.RP_DEFAULT)

    vol = df.std()
    if (vol <= 0).any() or not np.isfinite(vol).all():
        return dict(C.RP_DEFAULT)

    iv = 1.0 / vol

    growth = (1.0 + df).cumprod()
    peaks = growth.cummax()
    dd = (growth / peaks - 1.0).iloc[-1].clip(lower=-0.50, upper=0.0)

    # 0.50 tabanlı drawdown multiplier:
    # DD = 0    -> 1.00
    # DD = -20% -> 0.80
    # DD <=-50% -> 0.50
    dd_mult = (1.0 + dd).clip(lower=0.50, upper=1.00)
    score = iv * dd_mult

    if (score <= 0).any() or not np.isfinite(score).all():
        return dict(C.RP_DEFAULT)

    w = score / score.sum()
    wb = float(w["bist"])
    wb = min(max(wb, C.RP_BOUNDS[0]), C.RP_BOUNDS[1])

    return {"bist": wb, "us": 1.0 - wb}


class PrecomputedMarketData(E.MarketData):
    """Gerçek E.MarketData altyapısının veri/mekanik nesnesini paylaşır; yalnızca score/U değişir."""

    def __init__(
        self,
        md_source: E.MarketData,
        score_df: pd.DataFrame,
        U_df: pd.DataFrame,
        cost_rt_pct: float,
    ):
        # Yeniden MarketData.__init__ çalıştırmak gereksiz ve pahalıdır; kaynak
        # nesnenin zaten düzeltilmiş fiyat, tarih, evren ve motor durumunu paylaşırız.
        self.__dict__ = md_source.__dict__.copy()
        self.score = score_df.reindex(
            self.dates, columns=self.tickers
        ).to_numpy(dtype=float)
        self.U = U_df.reindex(
            self.dates, columns=self.tickers
        ).fillna(False).to_numpy(dtype=bool)
        self.fc = float(cost_rt_pct) / 200.0


def metrics_from_nav_trades(daily: pd.Series, trades: pd.DataFrame) -> dict:
    nav = daily.dropna().sort_index()
    if len(nav) < 30:
        return {}

    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / yrs) - 1.0
    dd = nav / nav.cummax() - 1.0

    tr = trades.copy()
    if not tr.empty and "ret_pct" in tr.columns:
        wins = tr.loc[tr.ret_pct > 0, "ret_pct"].sum()
        losses = -tr.loc[tr.ret_pct <= 0, "ret_pct"].sum()
        pf = wins / losses if losses > 0 else np.inf
        trade_win = float((tr.ret_pct > 0).mean() * 100.0)
        avg_trade = float(tr.ret_pct.mean())
    else:
        pf = np.nan
        trade_win = np.nan
        avg_trade = np.nan

    months = nav.resample("ME").last().pct_change().dropna()
    worst12 = np.nan
    pos12 = np.nan
    if len(months) >= 12:
        roll12 = (1.0 + months).rolling(12).apply(np.prod, raw=True) - 1.0
        worst12 = float(roll12.min() * 100.0)
        pos12 = float((roll12 > 0).mean() * 100.0)

    turnover_proxy = np.nan
    if not tr.empty and "pnl" in tr.columns:
        turnover_proxy = float(len(tr))

    return {
        "CAGR_pct": float(cagr * 100.0),
        "Total_x": float(nav.iloc[-1] / nav.iloc[0]),
        "MaxDD_pct": float(dd.min() * 100.0),
        "TradeWin_pct": trade_win,
        "AvgTrade_pct": avg_trade,
        "ProfitFactor": float(pf),
        "Worst12M_pct": worst12,
        "Positive12M_pct": pos12,
        "Trades": int(len(tr)),
        "TradeCountProxy": turnover_proxy,
    }


def block_metrics(
    daily: pd.Series,
    trades: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    nav = daily[(daily.index >= start) & (daily.index <= end)].dropna()
    if len(nav) < 20:
        return {
            "cagr": np.nan,
            "dd": np.nan,
            "trades": 0,
            "win": np.nan,
            "pf": np.nan,
        }

    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / yrs) - 1.0
    dd = nav / nav.cummax() - 1.0

    tr = trades.copy()
    if not tr.empty and "exit_date" in tr.columns:
        tr["exit_date"] = pd.to_datetime(tr["exit_date"])
        tr = tr[
            (tr["exit_date"] >= pd.Timestamp(start))
            & (tr["exit_date"] <= pd.Timestamp(end))
        ]

    if tr.empty:
        return {
            "cagr": float(cagr * 100.0),
            "dd": float(dd.min() * 100.0),
            "trades": 0,
            "win": np.nan,
            "pf": np.nan,
        }

    wins = tr.loc[tr.ret_pct > 0, "ret_pct"].sum()
    losses = -tr.loc[tr.ret_pct <= 0, "ret_pct"].sum()
    pf = wins / losses if losses > 0 else np.inf

    return {
        "cagr": float(cagr * 100.0),
        "dd": float(dd.min() * 100.0),
        "trades": int(len(tr)),
        "win": float((tr.ret_pct > 0).mean() * 100.0),
        "pf": float(pf),
    }


def bootstrap_monthly(
    base_daily: pd.Series,
    test_daily: pd.Series,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    a = base_daily.resample("ME").last().pct_change()
    b = test_daily.resample("ME").last().pct_change()

    df = pd.concat(
        [a.rename("base"), b.rename("test")],
        axis=1,
    ).dropna()

    if len(df) < 12:
        return {
            "months": int(len(df)),
            "mean": np.nan,
            "median": np.nan,
            "ci5": np.nan,
            "ci95": np.nan,
            "p_le0": np.nan,
            "positive_pct": np.nan,
        }

    diff = (df["test"] - df["base"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        diff,
        size=(n_boot, len(diff)),
        replace=True,
    ).mean(axis=1)

    return {
        "months": int(len(diff)),
        "mean": float(diff.mean() * 100.0),
        "median": float(np.median(diff) * 100.0),
        "ci5": float(np.quantile(samples, 0.05) * 100.0),
        "ci95": float(np.quantile(samples, 0.95) * 100.0),
        "p_le0": float((samples <= 0).mean() * 100.0),
        "positive_pct": float((diff > 0).mean() * 100.0),
    }


def fmt(x, d: int = 2) -> str:
    try:
        v = float(x)
        if not np.isfinite(v):
            return "—"
        return f"{v:.{d}f}"
    except Exception:
        return "—"


def _new_state_overlay_date(state: dict, d: pd.Timestamp):
    state["pf"]["_master_overlay_date"] = str(pd.Timestamp(d).date())


def _install_adaptive_rp():
    old = E.rp_weights
    E.rp_weights = adaptive_rp_weights
    return old


def _install_overlay(regime_map: pd.Series, exposure: float):
    old_ins = E.insurance_update
    old_exposure = getattr(C, "INS_EXPOSURE", 0.25)
    C.INS_EXPOSURE = float(exposure)

    def overlay_insurance_update(pf: dict, total_tl: float):
        pf["peak_tl"] = max(pf.get("peak_tl", total_tl), total_tl)
        pf["dd"] = total_tl / pf["peak_tl"] - 1.0 if pf.get("peak_tl", 0) > 0 else 0.0

        date_key = pf.get("_master_overlay_date")
        stress = float(regime_map.get(pd.Timestamp(date_key), 0.0)) if date_key else 0.0

        if not pf.get("insurance") and stress >= OVERLAY_TRIGGER:
            pf["insurance"] = True
            return "trigger"

        if pf.get("insurance") and stress <= OVERLAY_RELEASE:
            pf["insurance"] = False
            return "release"

        return None

    E.insurance_update = overlay_insurance_update
    return old_ins, old_exposure


def _restore_overlay(old_ins, old_exposure):
    E.insurance_update = old_ins
    C.INS_EXPOSURE = old_exposure


def simulate_full(
    name: str,
    mdb: E.MarketData,
    mdu: E.MarketData,
    fx: pd.Series,
    us_tbill: pd.Series,
    start: pd.Timestamp,
    cost: float,
    *,
    adaptive_risk: bool = False,
    regime_overlay: bool = False,
    regime_map: pd.Series | None = None,
):
    """
    Gerçek P.process_day motorunu kullanır.
    """
    old_rp = None
    old_ins = None
    old_exposure = None

    try:
        if adaptive_risk:
            old_rp = _install_adaptive_rp()

        if regime_overlay:
            if regime_map is None:
                raise ValueError("Regime overlay için regime_map gerekli.")
            old_ins, old_exposure = _install_overlay(regime_map, OVERLAY_EXPOSURE)

        P.set_hist_rates(us_tbill)

        rb, _ = P.self_financed_returns(mdb)
        ru_usd, _ = P.self_financed_returns(mdu)
        ru = P._to_tl_returns(ru_usd, "us", fx)

        def shadow(mk, d):
            s = rb if mk == "bist" else ru
            return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)

        end = max(mdb.dates[-1], mdu.dates[-1])
        days = sorted(
            set(
                d
                for d in (mdb.dates + mdu.dates)
                if start <= pd.Timestamp(d) <= end
            )
        )

        state = P.new_state(
            C.CAPITAL_TL,
            P.fx_at(fx, start),
        )

        for d in days:
            _new_state_overlay_date(state, d)
            if d in mdb.didx:
                P.process_day(
                    state,
                    "bist",
                    mdb,
                    mdb.didx[d],
                    fx,
                    shadow,
                )
            if d in mdu.didx:
                P.process_day(
                    state,
                    "us",
                    mdu,
                    mdu.didx[d],
                    fx,
                    shadow,
                )

        nav = pd.DataFrame(
            state["pf"]["nav_tl"],
            columns=["date", "mk", "total_tl", "fx"],
        )
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

        return Scenario(
            name=name,
            daily=daily,
            state=state,
            trades=pd.DataFrame(trades),
        )
    finally:
        if adaptive_risk and old_rp is not None:
            E.rp_weights = old_rp
        if regime_overlay and old_ins is not None:
            _restore_overlay(old_ins, old_exposure)


def _scenario_kind_flags(name: str):
    return {
        "BASE": dict(adaptive_risk=False, regime_overlay=False),
        "SIGNAL_QUALITY": dict(adaptive_risk=False, regime_overlay=False),
        "ADAPTIVE_RISK": dict(adaptive_risk=True, regime_overlay=False),
        "REGIME_OVERLAY": dict(adaptive_risk=False, regime_overlay=True),
        "REGIME_ADAPTIVE": dict(adaptive_risk=False, regime_overlay=False),
    }[name]


def _decision(
    base_m: dict,
    test_m: dict,
    stress_rows: list[dict],
    holdout_delta: float,
    boot: dict,
) -> tuple[str, str]:
    cagr_delta = test_m["CAGR_pct"] - base_m["CAGR_pct"]
    dd_delta = test_m["MaxDD_pct"] - base_m["MaxDD_pct"]
    pf_delta = test_m["ProfitFactor"] - base_m["ProfitFactor"]

    stress_good = int(
        sum(
            1
            for row in stress_rows
            if row["CAGR_delta_pp"] >= -MAX_ALLOWED_CAGR_LOSS_PP
            and row["DD_delta_pp"] <= MAX_ALLOWED_DD_WORSEN_PP
            and row["PF_delta"] >= -MAX_PF_LOSS
        )
    )

    if (
        (
            cagr_delta >= MIN_CAGR_EDGE_PP
            and dd_delta <= MAX_ALLOWED_DD_WORSEN_PP
        )
        or (
            dd_delta <= -MIN_DD_IMPROVEMENT_PP
            and cagr_delta >= -MAX_ALLOWED_CAGR_LOSS_PP
        )
    ) and pf_delta >= -MAX_PF_LOSS and stress_good >= MIN_STRESS_GOOD and holdout_delta >= -MAX_HOLDOUT_LAG_PP:
        if np.isfinite(boot.get("ci5", np.nan)) and boot["ci5"] > 0:
            return (
                "DEVELOP FURTHER — STRONG",
                "Ana dönem, holdout, maliyet stresi ve bootstrap aynı yönde veya risk açısından anlamlı iyileşmeyi destekliyor.",
            )
        return (
            "DEVELOP FURTHER",
            "Ek araştırmayı hak eden pozitif performans/risk dengesi var; ancak bootstrap tarafında tam güçlü kanıt yok.",
        )

    if (
        dd_delta <= -MIN_DD_IMPROVEMENT_PP
        and cagr_delta >= -2.0
        and pf_delta >= -0.10
    ):
        return (
            "SECOND LOOK",
            "Drawdown/risk tarafında dikkate değer iyileşme var; alpha doğrulaması yeterince güçlü değil.",
        )

    return (
        "REJECT FOR NOW",
        "BASE'e karşı yeterli performans/risk katkısı veya stres dayanıklılığı göstermiyor.",
    )


def research_rank(base_m: dict, test_m: dict, stress_rows: list[dict], holdout_delta: float, boot: dict) -> float:
    """
    Karar skoru değil; araştırma sıralaması için normalize edilmiş yardımcı skor.
    """
    cagr_edge = np.clip((test_m["CAGR_pct"] - base_m["CAGR_pct"]) / 2.0, -1.0, 1.0)
    dd_edge = np.clip(-(test_m["MaxDD_pct"] - base_m["MaxDD_pct"]) / 5.0, -1.0, 1.0)
    pf_edge = np.clip((test_m["ProfitFactor"] - base_m["ProfitFactor"]) / 0.15, -1.0, 1.0)

    if stress_rows:
        stress_edge = np.mean(
            [
                1.0
                if r["CAGR_delta_pp"] > 0
                else 0.5
                if r["CAGR_delta_pp"] >= -0.5
                else 0.0
                for r in stress_rows
            ]
        )
    else:
        stress_edge = 0.0

    hold_edge = np.clip(holdout_delta / 2.0, -1.0, 1.0)
    stat_edge = (
        1.0
        if np.isfinite(boot.get("ci5", np.nan)) and boot["ci5"] > 0
        else 0.5
        if np.isfinite(boot.get("ci95", np.nan)) and boot["ci95"] > 0
        else 0.0
    )

    score = (
        0.35 * cagr_edge
        + 0.20 * dd_edge
        + 0.15 * pf_edge
        + 0.15 * stress_edge
        + 0.10 * hold_edge
        + 0.05 * stat_edge
    )
    return float(score)


def prepare_data(years: int):
    bist_panel = DA.get_panel("bist", years=years, force_full=False)
    us_panel = DA.get_panel("us", years=years, force_full=False)

    if bist_panel.empty:
        raise RuntimeError("BIST verisi indirilemedi.")
    if us_panel.empty:
        raise RuntimeError("ABD verisi indirilemedi.")

    fx = DA.get_fx(years)
    if fx.empty:
        raise RuntimeError("USD/TRY verisi indirilemedi.")

    irx = pd.Series(dtype=float)
    try:
        yf = DA._yf()
        raw = yf.download(
            "^IRX",
            period=f"{years}y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )["Close"]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
        irx = raw.dropna()
    except Exception as exc:
        print(f"⚠️ ^IRX alınamadı; portfolio fallback kullanılacak: {exc}")

    return {
        "bist": DA.to_wide(bist_panel),
        "us": DA.to_wide(us_panel),
        "fx": fx,
        "irx": irx,
        "raw_bist": bist_panel,
        "raw_us": us_panel,
    }


def main_research(years: int, costs: list[float], out_path: Path):
    if years < 8:
        raise ValueError("Master araştırma için en az 8 yıl önerilir.")

    W = prepare_data(years)

    # Gerçek üretim motorundan referans MarketData'lar.
    bist_base_md = E.MarketData(
        "bist",
        W["bist"],
        spec=C.CANDIDATES["bist"][next(iter(C.CANDIDATES["bist"]))],
    )
    us_md = E.MarketData(
        "us",
        W["us"],
        spec=C.CANDIDATES["us"][next(iter(C.CANDIDATES["us"]))],
    )

    base_U_df = pd.DataFrame(
        bist_base_md.U,
        index=bist_base_md.dates,
        columns=bist_base_md.tickers,
    )
    us_U_df = pd.DataFrame(
        us_md.U,
        index=us_md.dates,
        columns=us_md.tickers,
    )

    score_bundle = build_bist_research_scores(
        W["bist"],
        bist_base_md.dates,
        base_U_df,
    )

    regime_context = build_market_regime_context(
        W["bist"],
        W["us"],
        base_U_df,
        us_U_df,
        bist_base_md.dates,
    )

    # Gerçek BASE score'u kullan.
    base_score_df = pd.DataFrame(
        bist_base_md.score,
        index=bist_base_md.dates,
        columns=bist_base_md.tickers,
    )

    score_bundle["BASE"] = base_score_df

    candidate_scores = {
        "BASE": score_bundle["BASE"],
        "SIGNAL_QUALITY": score_bundle["SIGNAL_QUALITY"],
        "REGIME_ADAPTIVE": score_bundle["REGIME_ADAPTIVE"],
    }

    # Aday MarketData nesneleri.
    candidate_md_templates: dict[str, E.MarketData] = {}

    for name, score_df in candidate_scores.items():
        candidate_md_templates[name] = PrecomputedMarketData(
            bist_base_md,
            score_df=score_df,
            U_df=score_bundle["U"],
            cost_rt_pct=C.MARKETS["bist"]["cost_rt_pct"],
        )

    # BASE US aynı kalır.
    us_template = us_md

    common_start_idx = min(WARMUP_DAYS, len(bist_base_md.dates) - 1, len(us_md.dates) - 1)
    start = max(
        pd.Timestamp(bist_base_md.dates[common_start_idx]),
        pd.Timestamp(us_md.dates[common_start_idx]),
    )
    end = max(
        pd.Timestamp(bist_base_md.dates[-1]),
        pd.Timestamp(us_md.dates[-1]),
    )

    # Stress serisini ortak, candidate-independent map olarak hazırla.
    overlay_regime_series = regime_context["stress"].fillna(0.0)

    # Tam ölçüm:
    # BASE
    # SIGNAL_QUALITY
    # ADAPTIVE_RISK
    # REGIME_OVERLAY
    # REGIME_ADAPTIVE
    scenario_names = [
        "BASE",
        "SIGNAL_QUALITY",
        "ADAPTIVE_RISK",
        "REGIME_OVERLAY",
        "REGIME_ADAPTIVE",
    ]

    print(f"🧪 Master ölçüm: {start.date()} → {end.date()}")
    print(f"🔬 Adaylar: {', '.join(scenario_names)}")

    base_template = candidate_md_templates["BASE"]
    quality_template = candidate_md_templates["SIGNAL_QUALITY"]
    regime_template = candidate_md_templates["REGIME_ADAPTIVE"]

    results_by_cost: dict[str, dict[str, Scenario]] = {}
    metrics_by_cost: dict[str, dict[str, dict]] = {}

    for cost in costs:
        print(f"\n💰 Maliyet senaryosu: {cost:.2f}% RT")

        md_base = CostClone.md_with_cost(base_template, cost)
        md_quality = CostClone.md_with_cost(quality_template, cost)
        md_regime = CostClone.md_with_cost(regime_template, cost)
        md_us = CostClone.md_with_cost(us_template, cost)

        scenarios_for_cost: dict[str, Scenario] = {}

        scenarios_for_cost["BASE"] = simulate_full(
            "BASE",
            md_base,
            md_us,
            W["fx"],
            W["irx"],
            start,
            cost,
        )

        scenarios_for_cost["SIGNAL_QUALITY"] = simulate_full(
            "SIGNAL_QUALITY",
            md_quality,
            CostClone.md_with_cost(us_template, cost),
            W["fx"],
            W["irx"],
            start,
            cost,
        )

        scenarios_for_cost["ADAPTIVE_RISK"] = simulate_full(
            "ADAPTIVE_RISK",
            md_base,
            CostClone.md_with_cost(us_template, cost),
            W["fx"],
            W["irx"],
            start,
            cost,
            adaptive_risk=True,
        )

        scenarios_for_cost["REGIME_OVERLAY"] = simulate_full(
            "REGIME_OVERLAY",
            md_base,
            CostClone.md_with_cost(us_template, cost),
            W["fx"],
            W["irx"],
            start,
            cost,
            regime_overlay=True,
            regime_map=overlay_regime_series,
        )

        scenarios_for_cost["REGIME_ADAPTIVE"] = simulate_full(
            "REGIME_ADAPTIVE",
            md_regime,
            CostClone.md_with_cost(us_template, cost),
            W["fx"],
            W["irx"],
            start,
            cost,
        )

        results_by_cost[f"{cost:.4f}"] = scenarios_for_cost
        metrics_by_cost[f"{cost:.4f}"] = {
            k: metrics_from_nav_trades(v.daily, v.trades)
            for k, v in scenarios_for_cost.items()
        }

    base_key = f"{BASE_COST:.4f}"
    if base_key not in results_by_cost:
        # En azından kullanılan ilk maliyet ile karar alınabilir.
        base_key = f"{costs[0]:.4f}"

    base_result = results_by_cost[base_key]["BASE"]
    base_m = metrics_by_cost[base_key]["BASE"]

    blocks = [
        (
            f"{start.year}-{min(start.year + 4, end.year)}",
            pd.Timestamp(f"{start.year}-01-01"),
            pd.Timestamp(f"{min(start.year + 4, end.year)}-12-31"),
        ),
        (
            f"{min(start.year + 5, end.year)}-{min(start.year + 8, end.year)}",
            pd.Timestamp(f"{min(start.year + 5, end.year)}-01-01"),
            pd.Timestamp(f"{min(start.year + 8, end.year)}-12-31"),
        ),
        (
            f"{min(start.year + 9, end.year)}-{end.year}",
            pd.Timestamp(f"{min(start.year + 9, end.year)}-01-01"),
            pd.Timestamp(f"{end.year}-12-31"),
        ),
    ]
    blocks = [b for b in blocks if b[1] <= b[2]]

    report_rows = []
    detail = {}

    for candidate in scenario_names[1:]:
        test_result = results_by_cost[base_key][candidate]
        test_m = metrics_by_cost[base_key][candidate]

        boot = bootstrap_monthly(
            base_result.daily,
            test_result.daily,
        )

        block_rows = []
        for label, bs, be in blocks:
            bm = block_metrics(
                base_result.daily,
                base_result.trades,
                bs,
                be,
            )
            tm = block_metrics(
                test_result.daily,
                test_result.trades,
                bs,
                be,
            )
            block_rows.append(
                {
                    "label": label,
                    "base": bm,
                    "test": tm,
                    "cagr_delta": tm["cagr"] - bm["cagr"],
                    "dd_delta": tm["dd"] - bm["dd"],
                }
            )

        holdout_delta = (
            block_rows[-1]["cagr_delta"]
            if block_rows
            else np.nan
        )

        stress_rows = []
        for key, mm in metrics_by_cost.items():
            cost = float(key)
            stress_rows.append(
                {
                    "cost": cost,
                    "BASE_CAGR": mm["BASE"]["CAGR_pct"],
                    "TEST_CAGR": mm[candidate]["CAGR_pct"],
                    "CAGR_delta_pp": mm[candidate]["CAGR_pct"] - mm["BASE"]["CAGR_pct"],
                    "BASE_DD": mm["BASE"]["MaxDD_pct"],
                    "TEST_DD": mm[candidate]["MaxDD_pct"],
                    "DD_delta_pp": mm[candidate]["MaxDD_pct"] - mm["BASE"]["MaxDD_pct"],
                    "BASE_PF": mm["BASE"]["ProfitFactor"],
                    "TEST_PF": mm[candidate]["ProfitFactor"],
                    "PF_delta": mm[candidate]["ProfitFactor"] - mm["BASE"]["ProfitFactor"],
                    "BASE_TradeWin": mm["BASE"]["TradeWin_pct"],
                    "TEST_TradeWin": mm[candidate]["TradeWin_pct"],
                }
            )

        verdict, reason = _decision(
            base_m,
            test_m,
            stress_rows,
            holdout_delta,
            boot,
        )

        rank_score = research_rank(
            base_m,
            test_m,
            stress_rows,
            holdout_delta,
            boot,
        )

        report_rows.append(
            {
                "candidate": candidate,
                "CAGR": test_m["CAGR_pct"],
                "CAGR_delta": test_m["CAGR_pct"] - base_m["CAGR_pct"],
                "MaxDD": test_m["MaxDD_pct"],
                "MaxDD_delta": test_m["MaxDD_pct"] - base_m["MaxDD_pct"],
                "PF": test_m["ProfitFactor"],
                "PF_delta": test_m["ProfitFactor"] - base_m["ProfitFactor"],
                "Win": test_m["TradeWin_pct"],
                "Worst12M": test_m["Worst12M_pct"],
                "Worst12M_delta": test_m["Worst12M_pct"] - base_m["Worst12M_pct"],
                "Trades": test_m["Trades"],
                "BootMean": boot["mean"],
                "BootCI5": boot["ci5"],
                "BootCI95": boot["ci95"],
                "BootPLe0": boot["p_le0"],
                "HoldoutDelta": holdout_delta,
                "StressGood": int(
                    sum(
                        1
                        for x in stress_rows
                        if x["CAGR_delta_pp"] >= -MAX_ALLOWED_CAGR_LOSS_PP
                        and x["DD_delta_pp"] <= MAX_ALLOWED_DD_WORSEN_PP
                        and x["PF_delta"] >= -MAX_PF_LOSS
                    )
                ),
                "RankScore": rank_score,
                "Decision": verdict,
                "Reason": reason,
            }
        )

        detail[candidate] = {
            "metrics": test_m,
            "bootstrap": boot,
            "blocks": block_rows,
            "stress": stress_rows,
        }

    ranking_df = pd.DataFrame(report_rows).sort_values(
        ["RankScore", "CAGR_delta"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Araştırma verisi özetleri.
    data_stats = {
        "bist_tickers": int(W["raw_bist"]["ticker"].nunique()),
        "us_tickers": int(W["raw_us"]["ticker"].nunique()),
        "bist_rows": int(len(W["raw_bist"])),
        "us_rows": int(len(W["raw_us"])),
        "bist_first": str(pd.to_datetime(W["raw_bist"]["tarih"]).min().date()),
        "bist_last": str(pd.to_datetime(W["raw_bist"]["tarih"]).max().date()),
        "us_first": str(pd.to_datetime(W["raw_us"]["tarih"]).min().date()),
        "us_last": str(pd.to_datetime(W["raw_us"]["tarih"]).max().date()),
        "fx_first": str(W["fx"].index.min().date()),
        "fx_last": str(W["fx"].index.max().date()),
    }

    # BASE sanity snapshot.
    base_sanity = {
        "CAGR": base_m["CAGR_pct"],
        "MaxDD": base_m["MaxDD_pct"],
        "PF": base_m["ProfitFactor"],
        "Win": base_m["TradeWin_pct"],
        "Worst12M": base_m["Worst12M_pct"],
        "Trades": base_m["Trades"],
        "Total": base_m["Total_x"],
    }

    lines = [
        "# Global Momentum Quant — Master Improvement Research",
        "",
        f"_Üretim: {pd.Timestamp.now():%Y-%m-%d %H:%M} · veri: {years} yıl · ölçüm: {start.date()} → {end.date()}_",
        "",
        "## 1. Amaç",
        "",
        "Mevcut BASE stratejisi canlı sisteme dokunulmadan beş araştırma alanında taranmıştır. Dört alan bağımsız strateji/overlay adayıdır; beşinci alan ortak robustness/data doğrulamasıdır.",
        "",
        "Test edilen adaylar:",
        "",
        "1. `SIGNAL_QUALITY` — BASE + düşük volatilite/pozitif-gün kalitesi.",
        "2. `ADAPTIVE_RISK` — BASE seçimleri + drawdown-aware BIST/ABD risk allocation.",
        "3. `REGIME_OVERLAY` — BASE seçimleri + candidate-independent soft stress exposure overlay.",
        "4. `REGIME_ADAPTIVE` — BASE ile quality arasında rejime bağlı adaptive selection.",
        "5. `ROBUSTNESS / DATA` — cost stress + bootstrap + dönemler + holdout + data audit.",
        "",
        "Canlı `config.py`, `engine.py`, `portfolio.py`, `signals.py` ve state/order/NAV/trade dosyaları değiştirilmez.",
        "",
        "## 2. BASE referans sonucu",
        "",
        "| Metrik | BASE |",
        "|---|---:|",
        f"| CAGR | {fmt(base_sanity['CAGR'])}% |",
        f"| Max DD | {fmt(base_sanity['MaxDD'])}% |",
        f"| Profit Factor | {fmt(base_sanity['PF'])} |",
        f"| İşlem Win | {fmt(base_sanity['Win'])}% |",
        f"| En kötü 12 ay | {fmt(base_sanity['Worst12M'])}% |",
        f"| İşlem sayısı | {int(base_sanity['Trades'])} |",
        f"| Toplam çarpan | {fmt(base_sanity['Total'])}x |",
        "",
        f"Karar referansı maliyeti: **{base_key} RT**.",
        "",
        "## 3. Master karşılaştırma",
        "",
        "| Aday | CAGR | ΔCAGR | MaxDD | ΔDD | PF | ΔPF | Win | Worst12M | Holdout ΔCAGR | Stress | Bootstrap CI5 | Karar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in report_rows:
        lines.append(
            f"| {row['candidate']} | {fmt(row['CAGR'])}% | {fmt(row['CAGR_delta'])} pp | "
            f"{fmt(row['MaxDD'])}% | {fmt(row['MaxDD_delta'])} pp | {fmt(row['PF'])} | "
            f"{fmt(row['PF_delta'])} | {fmt(row['Win'])}% | {fmt(row['Worst12M'])}% | "
            f"{fmt(row['HoldoutDelta'])} pp | {row['StressGood']}/{len(costs)} | "
            f"{fmt(row['BootCI5'])} pp | {row['Decision']} |"
        )

    lines += [
        "",
        "### Öncelik sıralaması",
        "",
    ]

    for i, row in ranking_df.iterrows():
        lines.append(
            f"{i+1}. **{row['candidate']}** — araştırma skoru {row['RankScore']:.3f} — **{row['Decision']}**"
        )

    lines += [
        "",
        "## 4. Signal Quality",
        "",
        "Hipotez: BASE'in seçtiği adaylar içinde düşük volatilite ve pozitif-gün persistence bilgisi ek kalite bilgisi sağlayabilir mi?",
        "",
        f"Kalite blend'i önceden sabit: **{QUALITY_BLEND:.2f}**.",
        "Bu alan sonuç görüldükten sonra optimize edilmemiştir.",
        "",
    ]

    if "SIGNAL_QUALITY" in detail:
        q = detail["SIGNAL_QUALITY"]
        lines += [
            f"- Ana dönem CAGR farkı: **{fmt(report_rows[[r['candidate'] for r in report_rows].index('SIGNAL_QUALITY')]['CAGR_delta'])} pp**",
            f"- Bootstrap ortalama aylık fark: **{fmt(q['bootstrap']['mean'])} pp**",
            f"- Bootstrap %5 alt sınırı: **{fmt(q['bootstrap']['ci5'])} pp**",
            f"- Final holdout CAGR farkı: **{fmt(q['blocks'][-1]['cagr_delta'] if q['blocks'] else np.nan)} pp**",
        ]

    lines += [
        "",
        "## 5. Adaptive Risk Allocation",
        "",
        "Hipotez: BASE hisse seçimlerini değiştirmeden, BIST/ABD sermaye ağırlıkları inverse-vol + son 63 günlük drawdown riskine göre ayarlanırsa risk/getiri iyileşebilir mi?",
        "",
        "Drawdown multiplier: 0% DD'de 1.00, -20% DD'de 0.80, -50% veya daha kötüde 0.50 ile sınırlandırılır.",
        "",
    ]

    if "ADAPTIVE_RISK" in detail:
        a = detail["ADAPTIVE_RISK"]
        row = next(r for r in report_rows if r["candidate"] == "ADAPTIVE_RISK")
        lines += [
            f"- CAGR farkı: **{fmt(row['CAGR_delta'])} pp**",
            f"- MaxDD farkı: **{fmt(row['MaxDD_delta'])} pp**",
            f"- PF farkı: **{fmt(row['PF_delta'])}**",
            f"- Final holdout CAGR farkı: **{fmt(row['HoldoutDelta'])} pp**",
            f"- Stress-good: **{row['StressGood']}/{len(costs)}**",
        ]

    lines += [
        "",
        "## 6. Regime Risk Overlay",
        "",
        "Hipotez: rejim kötüleştiğinde sinyali kapatmak yerine maruziyeti soft olarak azaltmak drawdown kontrolünde işe yarar mı?",
        "",
        f"Soft maruziyet: **{OVERLAY_EXPOSURE:.2f}**.",
        f"Trigger stress >= **{OVERLAY_TRIGGER:.2f}**, release stress <= **{OVERLAY_RELEASE:.2f}**.",
        "Overlay sinyal üretimini değiştirmez; mevcut pozisyon boyutunu küçültür.",
        "",
    ]

    if "REGIME_OVERLAY" in detail:
        ro = detail["REGIME_OVERLAY"]
        row = next(r for r in report_rows if r["candidate"] == "REGIME_OVERLAY")
        lines += [
            f"- CAGR farkı: **{fmt(row['CAGR_delta'])} pp**",
            f"- MaxDD farkı: **{fmt(row['MaxDD_delta'])} pp**",
            f"- PF farkı: **{fmt(row['PF_delta'])}**",
            f"- Final holdout CAGR farkı: **{fmt(row['HoldoutDelta'])} pp**",
            f"- Stress-good: **{row['StressGood']}/{len(costs)}**",
        ]

    lines += [
        "",
        "## 7. Regime-Specific Adaptive Selection",
        "",
        "Hipotez: normal rejimde BASE çekirdeği baskın, stres arttığında quality component ağırlığı kontrollü biçimde yükselirse seçim kalitesi iyileşebilir mi?",
        "",
        f"Quality ağırlığı: **{REGIME_QUALITY_MIN:.2f} → {REGIME_QUALITY_MAX:.2f}**.",
        "Rejim ağırlığı yalnızca mevcut güne kadar görülen BIST+ABD cross-sectional stress bilgisiyle hesaplanır.",
        "",
    ]

    if "REGIME_ADAPTIVE" in detail:
        rr = detail["REGIME_ADAPTIVE"]
        row = next(r for r in report_rows if r["candidate"] == "REGIME_ADAPTIVE")
        lines += [
            f"- CAGR farkı: **{fmt(row['CAGR_delta'])} pp**",
            f"- MaxDD farkı: **{fmt(row['MaxDD_delta'])} pp**",
            f"- PF farkı: **{fmt(row['PF_delta'])}**",
            f"- Final holdout CAGR farkı: **{fmt(row['HoldoutDelta'])} pp**",
            f"- Stress-good: **{row['StressGood']}/{len(costs)}**",
        ]

    lines += [
        "",
        "## 8. Robustness / Data Validation",
        "",
        "### Maliyet stresleri",
        "",
        "| RT maliyet | BASE CAGR | SIGNAL_QUALITY | ADAPTIVE_RISK | REGIME_OVERLAY | REGIME_ADAPTIVE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for cost in costs:
        key = f"{cost:.4f}"
        mm = metrics_by_cost[key]
        lines.append(
            f"| {cost:.2f}% | {fmt(mm['BASE']['CAGR_pct'])}% | "
            f"{fmt(mm['SIGNAL_QUALITY']['CAGR_pct'])}% | "
            f"{fmt(mm['ADAPTIVE_RISK']['CAGR_pct'])}% | "
            f"{fmt(mm['REGIME_OVERLAY']['CAGR_pct'])}% | "
            f"{fmt(mm['REGIME_ADAPTIVE']['CAGR_pct'])}% |"
        )

    lines += [
        "",
        "Maliyet stresinde adayların sinyal/weight yolu yeniden optimize edilmez; her adayın aynı tanımlı kuralları farklı maliyet varsayımlarında sabit tutulur.",
        "",
        "### Bootstrap",
        "",
        "| Aday | Ort. aylık Δ | Medyan Δ | CI5 | CI95 | P(Δ≤0) | Pozitif ay |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in report_rows:
        b = detail[row["candidate"]]["bootstrap"]
        lines.append(
            f"| {row['candidate']} | {fmt(b['mean'])} pp | {fmt(b['median'])} pp | "
            f"{fmt(b['ci5'])} pp | {fmt(b['ci95'])} pp | {fmt(b['p_le0'])}% | "
            f"{fmt(b['positive_pct'])}% |"
        )

    lines += [
        "",
        "### Zaman blokları / final holdout",
        "",
    ]

    for candidate in [r["candidate"] for r in report_rows]:
        lines += [
            f"#### {candidate}",
            "",
            "| Dönem | BASE CAGR | Aday CAGR | ΔCAGR | BASE DD | Aday DD |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for b in detail[candidate]["blocks"]:
            lines.append(
                f"| {b['label']} | {fmt(b['base']['cagr'])}% | {fmt(b['test']['cagr'])}% | "
                f"{fmt(b['cagr_delta'])} pp | {fmt(b['base']['dd'])}% | {fmt(b['test']['dd'])}% |"
            )
        lines.append("")

    lines += [
        "### Veri denetimi",
        "",
        f"- BIST evreni: **{data_stats['bist_tickers']}** hisse",
        f"- ABD evreni: **{data_stats['us_tickers']}** hisse",
        f"- BIST veri satırı: **{data_stats['bist_rows']}**",
        f"- ABD veri satırı: **{data_stats['us_rows']}**",
        f"- BIST tarihleri: **{data_stats['bist_first']} → {data_stats['bist_last']}**",
        f"- ABD tarihleri: **{data_stats['us_first']} → {data_stats['us_last']}**",
        f"- USD/TRY tarihleri: **{data_stats['fx_first']} → {data_stats['fx_last']}**",
        "- ABD evreninde historical S&P membership kullanılmıyorsa survivorship bias sınırlaması devam eder.",
        "- Araştırma, gerçek `engine.MarketData` + `portfolio.process_day` mekaniklerini kullanır.",
        "- Canlı state/orders/NAV/trades dosyalarına yazılmaz.",
        "",
        "## 9. Nihai araştırma kararı",
        "",
    ]

    for i, row in ranking_df.iterrows():
        candidate = row["candidate"]
        lines += [
            f"### {i+1}. {candidate}",
            "",
            f"**{row['Decision']}**",
            "",
            row["Reason"],
            "",
        ]

    lines += [
        "### Sonuç",
        "",
    ]

    strong = ranking_df[ranking_df["Decision"].str.contains("DEVELOP FURTHER", na=False)]
    second = ranking_df[ranking_df["Decision"] == "SECOND LOOK"]

    if not strong.empty:
        lines.append(
            "Bu taramada derinleştirmeye değer görünen aday(lar): "
            + ", ".join(str(x) for x in strong["candidate"].tolist())
            + "."
        )
    else:
        lines.append(
            "Bu taramada BASE'i değiştirmeye değer yeterince güçlü bir aday çıkmadı."
        )

    if not second.empty:
        lines.append(
            "Risk tarafında ikinci bakışa değer aday(lar): "
            + ", ".join(str(x) for x in second["candidate"].tolist())
            + "."
        )

    lines += [
        "",
        "Bu rapor canlı sistemi değiştirmez. Bir adayın daha sonra geliştirilmesi ancak burada olumlu sonuç alan aday için ayrı, dondurulmuş final OOS doğrulamasından sonra yapılmalıdır.",
        "",
        "## 10. Araştırma kuralları",
        "",
        "- Tek bir 'magic' optimum aranmaz.",
        "- Adaylar birbirinin sonucundan türetilmez.",
        "- Final holdout sonucu parametre ayarlamak için kullanılmaz.",
        "- Maliyet stresleri sonuçlara göre seçilmez; önceden sabittir.",
        "- Rejim bilgisi point-in-time ve aday-independent olacak şekilde hesaplanır.",
        "- BASE üretim referansı olarak korunur.",
        "",
    ]

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== GMQ MASTER IMPROVEMENT RESEARCH ===")
    print(f"BASE: CAGR={fmt(base_m['CAGR_pct'])}% | DD={fmt(base_m['MaxDD_pct'])}% | PF={fmt(base_m['ProfitFactor'])} | Win={fmt(base_m['TradeWin_pct'])}%")
    for row in ranking_df.itertuples(index=False):
        print(
            f"{row.candidate}: CAGRΔ={fmt(row.CAGR_delta)} pp | "
            f"DDΔ={fmt(row.MaxDD_delta)} pp | PFΔ={fmt(row.PF_delta)} | "
            f"HoldoutΔ={fmt(row.HoldoutDelta)} pp | "
            f"Stress={row.StressGood}/{len(costs)} | {row.Decision}"
        )
    print(f"Rapor: {out_path}")
    print("Canlı state/orders/NAV/trades dosyalarına yazılmadı.")


def parse_costs(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("En az bir maliyet seviyesi gerekir.")
    return vals


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GMQ Master Improvement Research"
    )
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument(
        "--costs",
        default=",".join(str(x) for x in DEFAULT_COSTS),
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_REPORT),
    )
    args = ap.parse_args()

    costs = parse_costs(args.costs)
    main_research(
        years=args.years,
        costs=costs,
        out_path=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"❌ HATA: {exc}", file=sys.stderr)
        raise SystemExit(1)
