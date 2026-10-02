"""Portföy düzeyi: iki kolu (BIST TL, ABD $) tek bir TL portföyü olarak yönetir.

Canlı çalışmada da geçmiş testte de aynı `process_day` kullanılır.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import engine as E

OTHER = {"bist": "us", "us": "bist"}


def fx_at(fx: pd.Series, d):
    if fx is None or len(fx) == 0:
        return float("nan")
    s = fx[fx.index <= pd.Timestamp(d)]
    return float(s.iloc[-1]) if len(s) else float(fx.iloc[0])


# TCMB politika faizi (yaklaşık tarihçe, geçmiş test için) — canlıda config.CASH_RATE kullanılır
_TR_POLICY = [("2013-01-01", 5.5), ("2014-01-29", 10), ("2014-05-22", 9.5), ("2014-06-24", 8.75), ("2014-07-17", 8.25),
              ("2015-01-20", 7.75), ("2015-02-24", 7.5), ("2017-01-15", 10), ("2017-04-26", 12.25), ("2018-05-23", 16.5),
              ("2018-06-07", 17.75), ("2018-09-13", 24), ("2019-07-25", 19.75), ("2019-09-12", 16.5), ("2019-10-24", 14),
              ("2019-12-12", 12), ("2020-01-16", 11.25), ("2020-02-19", 10.75), ("2020-03-17", 9.75), ("2020-04-22", 8.75),
              ("2020-05-21", 8.25), ("2020-09-24", 10.25), ("2020-11-19", 15), ("2020-12-24", 17), ("2021-03-18", 19),
              ("2021-09-23", 18), ("2021-10-21", 16), ("2021-11-18", 15), ("2021-12-16", 14), ("2022-08-18", 13),
              ("2022-09-22", 12), ("2022-10-20", 10.5), ("2022-11-24", 9), ("2023-02-23", 8.5), ("2023-06-22", 15),
              ("2023-07-20", 17.5), ("2023-08-24", 25), ("2023-09-21", 30), ("2023-10-26", 35), ("2023-11-23", 40),
              ("2023-12-21", 42.5), ("2024-01-25", 45), ("2024-03-21", 50), ("2024-12-26", 47.5), ("2025-01-23", 45),
              ("2025-03-06", 42.5), ("2025-04-17", 46), ("2025-07-24", 43), ("2025-09-11", 40.5), ("2025-10-23", 39.5),
              ("2025-12-11", 38), ("2026-03-01", 37)]
HIST_RATES = {"bist": None, "us": None}   # geçmiş testte doldurulur: pd.Series (yıllık oran)


def cash_rate(mk, d):
    s = HIST_RATES.get(mk)
    if s is not None and len(s):
        v = s[s.index <= pd.Timestamp(d)]
        return float(v.iloc[-1]) if len(v) else float(s.iloc[0])
    return float(C.CASH_RATE.get(mk, 0.0))


def set_hist_rates(us_tbill=None):
    """Geçmiş test: TL için politika faizi*0.85, $ için 3 aylık hazine bonosu (IRX)."""
    s = pd.Series({pd.Timestamp(a): b * 0.85 / 100 for a, b in _TR_POLICY}).sort_index()
    HIST_RATES["bist"] = s
    HIST_RATES["us"] = (us_tbill / 100).dropna() if us_tbill is not None else pd.Series({pd.Timestamp("2013-01-01"): 0.02})


def new_state(capital_tl, fx_now):
    w = dict(C.FIXED_WEIGHTS) if getattr(C, "FIXED_WEIGHTS", None) else dict(C.RP_DEFAULT)
    return {
        "version": C.ENGINE_VERSION,
        "created": None,
        "markets": {
            "bist": E.new_market_state("bist", capital_tl * w["bist"]),
            "us": E.new_market_state("us", capital_tl * w["us"] / fx_now),
        },
        "pf": {"capital_tl": capital_tl, "peak_tl": capital_tl, "insurance": False, "dd": 0.0,
               "weights": w, "weights_month": None, "reduce_pending": {"bist": False, "us": False},
               "fx": fx_now, "transfers": [], "nav_tl": [], "shadow": {"bist": [], "us": []},
               "initial_split": {"bist_tl": capital_tl * w["bist"], "us_usd": capital_tl * w["us"] / fx_now}},
    }


def market_value_local(st):
    if st.get("nav_post") is not None:
        return float(st["nav_post"])
    if st["nav_hist"]:
        return float(st["nav_hist"][-1][1])
    return float(st["cash"])


def total_tl(state, fx):
    b = market_value_local(state["markets"]["bist"])
    u = market_value_local(state["markets"]["us"])
    return b + u * fx, b, u * fx


def _to_tl_returns(ret_local: pd.Series, mk, fx: pd.Series):
    if mk == "bist" or ret_local is None or len(ret_local) == 0:
        return ret_local
    f = fx.reindex(ret_local.index.union(fx.index)).ffill().reindex(ret_local.index)
    fr = f.pct_change().fillna(0)
    return (1 + ret_local) * (1 + fr) - 1


def process_day(state, mk, md: E.MarketData, i: int, fx: pd.Series, shadow_fn=None):
    """Bir kolun bir işlem gününü işler; portföy düzeyindeki ağırlık/sigorta/aktarım kararlarını uygular."""
    pf = state["pf"]
    st = state["markets"][mk]
    oth = state["markets"][OTHER[mk]]
    d = md.dates[i]
    f = fx_at(fx, d)
    if np.isfinite(f):
        pf["fx"] = f
    f = pf["fx"]

    # ayın ilk işlem gününde risk paritesi ağırlıkları
    month = f"{d.year}-{d.month:02d}"
    if pf.get("weights_month") != month and shadow_fn is not None:
        rb = shadow_fn("bist", d)
        ru = shadow_fn("us", d)
        if rb is not None and ru is not None and len(rb) and len(ru):
            idx = rb.index.union(ru.index)
            pf["weights"] = E.rp_weights(rb.reindex(idx).dropna(), ru.reindex(idx).dropna())
        if getattr(C, "FIXED_WEIGHTS", None):
            pf["weights"] = dict(C.FIXED_WEIGHTS)
        pf["weights_month"] = month

    # hedef değer
    nav_local = E.market_nav(md, i, st) if st["n"] >= 0 else st["cash"]
    other_tl = market_value_local(oth) * (f if OTHER[mk] == "us" else 1.0)
    me_tl = nav_local * (f if mk == "us" else 1.0)
    tot = me_tl + other_tl
    target_tl = tot * pf["weights"][mk]
    target_local = target_tl / (f if mk == "us" else 1.0)
    ctx = {"target_local": target_local,
           "exposure": C.INS_EXPOSURE if pf.get("insurance") else 1.0,
           "reduce": bool(pf["reduce_pending"].get(mk)),
           "cash_rate": cash_rate(mk, d)}
    ev = E.step_market(md, i, st, ctx)
    pf["reduce_pending"][mk] = False
    if pf.setdefault("restore_pending", {}).get(mk) and C.INS_RESTORE_NOW:
        E.restore_positions(md, i, st, ev, target_local)
        pf["restore_pending"][mk] = False

    # aktarım: bu kolun dilim bütçesi nakitten büyükse diğer kolun boştaki nakdi aktarılır
    ev["transfer"] = None
    if ev.get("shortfall", 0) > 0:
        need_tl = ev["shortfall"] * (f if mk == "us" else 1.0)
        reserved = sum(o.get("amount", 0) for o in oth["pending"] if o["side"] == "buy")
        idle_local = max(oth["cash"] - reserved, 0.0)
        idle_tl = idle_local * (f if OTHER[mk] == "us" else 1.0)
        amt_tl = min(need_tl, idle_tl)
        if amt_tl > 0.01 * tot:
            oth["cash"] -= amt_tl / (f if OTHER[mk] == "us" else 1.0)
            st["cash"] += amt_tl / (f if mk == "us" else 1.0)
            # bekleyen alımları yeni nakitle yeniden ölçekle
            buys = [o for o in st["pending"] if o["side"] == "buy"]
            ev["transfer"] = {"date": str(d.date()), "from": OTHER[mk], "to": mk, "tl": round(amt_tl, 2),
                              "usd": round(amt_tl / f, 2)}
            pf["transfers"].append(ev["transfer"])
            pf["transfers"] = pf["transfers"][-200:]
            missing = sum(max(o.get("full", o["amount"]) - o["amount"], 0) for o in buys)
            add_local = amt_tl / (f if mk == "us" else 1.0)
            if missing > 0:
                frac = min(1.0, add_local / missing)
                for o in buys:
                    o["amount"] += frac * max(o.get("full", o["amount"]) - o["amount"], 0)

    # kolun nakit akışından arındırılmış performans endeksi (aktarımlar performans sayılmaz)
    nav_step = float(st["nav_hist"][-1][1])
    tin = 0.0
    if ev.get("transfer"):
        tr = ev["transfer"]
        tin = (tr["tl"] if mk == "bist" else tr["usd"])
        if tr["to"] != mk:
            tin = -tin
    prev = st.get("nav_post")
    if prev and prev > 0:
        st["perf"] = st.get("perf", 1.0) * nav_step / prev
    else:
        st["perf"] = st.get("perf", 1.0)
    st["perf_peak"] = max(st.get("perf_peak", st["perf"]), st["perf"])
    st["nav_post"] = nav_step + tin
    if ev.get("transfer"):
        o_tin = -(ev["transfer"]["tl"] if OTHER[mk] == "bist" else ev["transfer"]["usd"])
        if ev["transfer"]["to"] == OTHER[mk]:
            o_tin = -o_tin
        if oth.get("nav_post") is not None:
            oth["nav_post"] = oth["nav_post"] + o_tin

    # sigorta
    tot_after, _, _ = total_tl(state, f)
    act = E.insurance_update(pf, tot_after)
    ev["insurance"] = act
    if act == "trigger":
        E.reduce_positions(st, ev)
        pf["reduce_pending"][OTHER[mk]] = True
    elif act == "release":
        ev["notes"].append("Portföy sigortası kapandı: pozisyonlar tam boyuta tamamlanıyor")
        if C.INS_RESTORE_NOW:
            E.restore_positions(md, i, st, ev, target_local)
            pf.setdefault("restore_pending", {})[OTHER[mk]] = True
            pf["reduce_pending"][OTHER[mk]] = False
    ev["weights"] = dict(pf["weights"])
    ev["total_tl"] = tot_after
    ev["dd"] = pf.get("dd", 0.0)
    ev["fx"] = f
    pf["nav_tl"].append([str(d.date()), mk, round(tot_after, 2), round(f, 4)])
    return ev


# ------------------------------------------------------------------ geçmiş test
def self_financed_returns(md: E.MarketData):
    """Kolun kendi kendini finanse eden tam geçmişi (risk paritesi için gölge seri)."""
    st = E.new_market_state(md.mk, 1.0)
    vals = []
    for i in range(len(md.dates)):
        nav_prev = E.market_nav(md, i, st) if st["n"] >= 0 else 1.0
        E.step_market(md, i, st, {"target_local": nav_prev, "exposure": 1.0})
        vals.append(st["nav_hist"][-1][1])
    s = pd.Series(vals, index=md.dates)
    return s.pct_change().fillna(0), st


def run_backtest(mdb: E.MarketData, mdu: E.MarketData, fx: pd.Series, start, end=None, capital=100_000.0):
    rb, _ = self_financed_returns(mdb)
    ru_usd, _ = self_financed_returns(mdu)
    ru = _to_tl_returns(ru_usd, "us", fx)

    def shadow(mk, d):
        s = rb if mk == "bist" else ru
        return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)

    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end else max(mdb.dates[-1], mdu.dates[-1])
    state = new_state(capital, fx_at(fx, start))
    days = sorted(set(d for d in mdb.dates + mdu.dates if start <= d <= end))
    for d in days:
        if d in mdb.didx:
            process_day(state, "bist", mdb, mdb.didx[d], fx, shadow)
        if d in mdu.didx:
            process_day(state, "us", mdu, mdu.didx[d], fx, shadow)
    nav = pd.DataFrame(state["pf"]["nav_tl"], columns=["date", "mk", "total_tl", "fx"])
    nav["date"] = pd.to_datetime(nav["date"])
    daily = nav.groupby("date")["total_tl"].last()
    return state, daily, rb, ru
