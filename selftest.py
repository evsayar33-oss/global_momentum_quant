"""Öz test: sentetik veriyle (geçici klasörde; ./data'ya asla yazmaz) motorun temel doğruluğunu sınar.

  python selftest.py
Kontroller:
  1) Fiyatlar sabit + maliyet 0 iken portföy değeri değişmez (para yaratılmaz/kaybolmaz).
  2) Her işlem döngüsü en fazla 21 işlem günü sürer.
  3) Felaket stopu %25 düşüşte tetiklenir, ertesi açılışta satılır.
  4) BIST'te düzeltilmemiş 10:1 bölünme stop tetiklemez, değer korunur.
  5) Canlı akış (gün gün, durum dosyasına yazıp okuyarak) ile tek seferlik geçmiş test aynı sonucu verir.
"""
import copy
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import config as C
import engine as E
import portfolio as P


def synth(n_t=60, n_d=420, seed=0, flat=False, split=None, crash=None):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_d)
    tick = [f"T{i:03d}" for i in range(n_t)]
    drift = rng.normal(0.0004, 0.0006, n_t)
    r = rng.normal(drift, 0.018, (n_d, n_t)) if not flat else np.zeros((n_d, n_t))
    r = np.clip(r, -0.09, 0.09)
    c = 50 * np.exp(np.cumsum(np.log1p(r), axis=0))
    if split is not None:
        j, d0 = split
        c[d0:, j] /= 10
    if crash is not None:
        j, d0 = crash
        c[d0:, j] *= 0.6
    o = c * (1 + (rng.normal(0, 0.002, c.shape) if not flat else 0))
    if split is not None:
        o[split[1]:, split[0]] = c[split[1]:, split[0]]
    if crash is not None:
        o[crash[1], crash[0]] = c[crash[1], crash[0]]
    h = np.maximum(c, o) * 1.01
    l = np.minimum(c, o) * 0.99
    v = rng.integers(1e5, 1e6, c.shape).astype(float)
    mk = lambda a: pd.DataFrame(a, index=dates, columns=tick)
    return {"o": mk(o), "h": mk(h), "l": mk(l), "c": mk(c), "v": mk(v)}


def fx_series(dates, flat=True):
    return pd.Series(30.0 if flat else np.linspace(30, 40, len(dates)), index=dates)


def check(name, cond, detail=""):
    print(("✅" if cond else "❌") + f" {name} {detail}")
    return bool(cond)


def main():
    tmp = tempfile.mkdtemp(prefix="gmq_selftest_")
    old_dirs = (C.DATA_DIR, C.STATE_FILE, C.TRADES_FILE, C.NAV_FILE, C.AUDIT_FILE, C.LOG_FILE)
    C.DATA_DIR = tmp
    C.STATE_FILE = os.path.join(tmp, "state.json")
    C.TRADES_FILE = os.path.join(tmp, "trades.csv")
    C.NAV_FILE = os.path.join(tmp, "nav.csv")
    C.AUDIT_FILE = os.path.join(tmp, "audit.json")
    C.LOG_FILE = os.path.join(tmp, "messages.log")
    ok = True
    saved_fw = getattr(C, "FIXED_WEIGHTS", None)
    C.FIXED_WEIGHTS = None          # testler iki kolu birlikte sınar
    try:
        # 1) muhasebe: sabit fiyat, sıfır maliyet
        saved = {m: C.MARKETS[m]["cost_rt_pct"] for m in C.MARKETS}
        saved_rate = dict(C.CASH_RATE)
        C.CASH_RATE = {"bist": 0.0, "us": 0.0}
        for m in C.MARKETS:
            C.MARKETS[m]["cost_rt_pct"] = 0.0
        Wb, Wu = synth(flat=True), synth(flat=True, seed=1)
        # sabit fiyatta sinyaller NaN olur; sıralama için küçük deterministik eğim ver
        for W in (Wb, Wu):
            for k in ("o", "h", "l", "c"):
                W[k] = W[k] * (1 + np.arange(W[k].shape[1]) * 1e-6)
        mdb, mdu = E.MarketData("bist", Wb, [["low_max", 1]]), E.MarketData("us", Wu, [["low_max", 1]])
        fx = fx_series(mdb.dates)
        state, daily, _, _ = P.run_backtest(mdb, mdu, fx, mdb.dates[300], capital=100000)
        ok &= check("muhasebe (sabit fiyat)", abs(daily.iloc[-1] / 100000 - 1) < 1e-6, f"son değer {daily.iloc[-1]:.4f}")
        for m in C.MARKETS:
            C.MARKETS[m]["cost_rt_pct"] = saved[m]
        C.CASH_RATE = saved_rate

        # 2) süre
        Wb = synth(seed=3)
        Wu = synth(seed=4)
        mdb, mdu = E.MarketData("bist", Wb, [["mom_12_1", 1]]), E.MarketData("us", Wu, [["mom_12_1", 1]])
        fx = fx_series(mdb.dates, flat=False)
        state, daily, _, _ = P.run_backtest(mdb, mdu, fx, mdb.dates[300], capital=100000)
        T = pd.DataFrame(state["markets"]["bist"]["trades"] + state["markets"]["us"]["trades"])
        didx = {str(d.date()): i for i, d in enumerate(mdb.dates)}
        span = (T["exit_date"].map(didx) - T["entry_date"].map(didx)).max()
        ok &= check("işlem süresi <= 22 gün (21 gün + ertesi açılış)", span <= 22, f"en uzun {span}")

        # 3) felaket stopu (ABD: tek günde %30 düşüş; BIST'te ±%10 sınır nedeniyle bu bölünme sayılır)
        Wc = synth(seed=3)
        base = float(Wc["c"].iloc[370, 5])
        Wc["c"].iloc[371:, 5] = base * 0.70
        Wc["o"].iloc[371:, 5] = base * 0.70
        Wc["o"].iloc[372:, 5] = base * 0.68
        mdc = E.MarketData("us", Wc, [["mom_12_1", 1]])
        st = E.new_market_state("us", 0.0)
        st["positions"].append({"id": "x", "t": "T005", "tranche": 0, "qty": 1.0, "entry_px": base, "entry_raw": base,
                                "entry_date": str(mdc.dates[370].date())})
        st["n"] = 1000
        E.step_market(mdc, 371, st, {"target_local": 0, "exposure": 1})
        ok &= check("stop emri üretildi (%30 düşüş)", any(o["reason"] == "felaket stopu" for o in st["pending"]))
        E.step_market(mdc, 372, st, {"target_local": 0, "exposure": 1})
        tr = st["trades"][-1] if st["trades"] else {}
        ok &= check("stop ertesi açılışta satıldı", not st["positions"] and tr.get("reason") == "felaket stopu",
                    f"getiri %{tr.get('ret_pct')}")
        # BIST: kademeli düşüş (her gün -%9) stopu tetikler
        Wg = synth(seed=3)
        b0 = float(Wg["c"].iloc[370, 5])
        for kk, i in enumerate(range(371, 375)):
            Wg["c"].iloc[i:, 5] = b0 * 0.91 ** (kk + 1)
            Wg["o"].iloc[i:, 5] = b0 * 0.91 ** (kk + 1)
        mdg = E.MarketData("bist", Wg, [["mom_12_1", 1]])
        st = E.new_market_state("bist", 0.0)
        st["positions"].append({"id": "z", "t": "T005", "tranche": 0, "qty": 1.0, "entry_px": b0, "entry_raw": b0,
                                "entry_date": str(mdg.dates[370].date())})
        st["n"] = 1000
        for i in range(371, 376):
            E.step_market(mdg, i, st, {"target_local": 0, "exposure": 1})
        ok &= check("BIST kademeli düşüşte stop", st["trades"] and st["trades"][-1]["reason"] == "felaket stopu",
                    f"getiri %{st['trades'][-1]['ret_pct'] if st['trades'] else None}")

        # 4) bölünme
        Wd = synth(seed=5, split=(7, 360))
        mdd = E.MarketData("bist", Wd, [["mom_12_1", 1]])
        st = E.new_market_state("bist", 0.0)
        pre = float(mdd.c[358, 7])
        st["positions"].append({"id": "y", "t": "T007", "tranche": 0, "qty": 10.0, "entry_px": pre, "entry_raw": pre,
                                "entry_date": str(mdd.dates[358].date()), "ref_date": str(mdd.dates[358].date()), "ref_px": pre})
        st["n"] = 1001
        for i in (359, 360, 361):
            E.step_market(mdd, i, st, {"target_local": 0, "exposure": 1})
        p = next(p for p in st["positions"] if p["t"] == "T007")
        val_after = p["qty"] * mdd.c[361, 7]
        ok &= check("bölünmede stop yok ve değer korunuyor", not st["pending"] and abs(val_after / (10 * pre) - 1) < 0.15,
                    f"değer oranı {val_after / (10 * pre):.3f}")

        # 5) canlı akış = geçmiş test
        Wb, Wu = synth(seed=8), synth(seed=9)
        specs = {"bist": [["mom_12_1", 1]], "us": [["mom_12_1", 1]]}
        mdb, mdu = E.MarketData("bist", Wb, specs["bist"]), E.MarketData("us", Wu, specs["us"])
        fx = fx_series(mdb.dates, flat=False)
        start = mdb.dates[320]
        ref_state, ref_daily, rb, ru = P.run_backtest(mdb, mdu, fx, start, capital=100000)
        # canlı: her gün durum JSON'a yazılıp okunur
        state = P.new_state(100000, float(fx[fx.index <= start].iloc[-1]))

        def shadow(mk, d):
            s = rb if mk == "bist" else P._to_tl_returns(ru * 0 + ru, "us", fx)
            s = rb if mk == "bist" else ru
            return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)
        for d in [d for d in mdb.dates if d >= start]:
            for mk, md in (("bist", mdb), ("us", mdu)):
                P.process_day(state, mk, md, md.didx[d], fx, shadow)
                state = json.loads(json.dumps(state, default=str))
        tot, _, _ = P.total_tl(state, float(fx.iloc[-1]))
        ok &= check("canlı akış = geçmiş test", abs(tot / ref_daily.iloc[-1] - 1) < 1e-6,
                    f"canlı {tot:,.2f} / test {ref_daily.iloc[-1]:,.2f}")
    finally:
        C.FIXED_WEIGHTS = saved_fw
        C.DATA_DIR, C.STATE_FILE, C.TRADES_FILE, C.NAV_FILE, C.AUDIT_FILE, C.LOG_FILE = old_dirs
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nSONUÇ:", "TÜM TESTLER GEÇTİ ✅" if ok else "HATA VAR ❌")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
