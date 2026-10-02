"""Çekirdek motor: 4 dilimli kademeli giriş + felaket stopu + kollar arası risk paritesi + portföy sigortası.

Aynı kod hem canlı çalışmada (her akşam bir gün ilerler) hem de geçmiş testte (13 yıl boyunca gün gün ilerler)
kullanılır; böylece canlı sistem test edilenle birebir aynı kurallarla çalışır.

Zamanlama (t = sinyal günü, kapanıştan sonra):
  * t akşamı: emirler üretilir (AL / SAT / TUT), Telegram'a gider.
  * t+1 açılışı: emirler açılış fiyatından gerçekleşmiş sayılır (+ yarım maliyet).
  * Dilim k, başlangıçtan itibaren k*5. işlem gününde ilk kez kurulur, sonra her 21 işlem gününde yenilenir.
  * Felaket stopu: kapanış <= giriş*(1-0.25) ise ertesi açılışta satılır, para dilim yenilenene kadar nakitte bekler.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

import config as C
import signals as SG


# ------------------------------------------------------------------ piyasa verisi (hızlı erişim)
class MarketData:
    def __init__(self, mk, W, spec=None):
        cfg = C.MARKETS[mk]
        self.mk = mk
        self.cfg = cfg
        self.spec = spec or cfg["spec"]
        c = W["c"]
        # işlem günleri: hisselerin en az %30'unun verisi olan günler
        cover = c.notna().mean(axis=1)
        days = cover[cover >= 0.3].index
        W = {k: v.reindex(days) for k, v in W.items()}
        self.dates = list(pd.DatetimeIndex(days))
        self.didx = {d: i for i, d in enumerate(self.dates)}
        self.tickers = list(W["c"].columns)
        self.tidx = {t: j for j, t in enumerate(self.tickers)}
        # BIST: Yahoo'nun düzeltmediği bedelsiz/bölünmeler (günlük %25+ düşüş veya %30+ sıçrama; BIST tavan/taban
        # sınırı ±%10 olduğu için gerçek hareket olamaz) -> sinyaller için geriye dönük düzeltilmiş seri,
        # açık pozisyonlar için olay günü adet/fiyat ölçekleme.
        c_raw = W["c"]
        self.event = None
        Wadj = W
        if mk == "bist":
            ratio = c_raw / c_raw.ffill().shift(1)
            evm = (ratio < 0.75) | (ratio > 1.30)
            f = ratio.where(evm, 1.0).fillna(1.0)
            cum = f.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
            Wadj = {k: (W[k] * cum if k in ("o", "h", "l", "c") else W[k] / cum) for k in W}
            self.event = f.values.astype(float)
        sc, U = SG.score(Wadj, self.spec, cfg["max_move"], cfg.get("liq_min_pct", C.LIQ_MIN_PCT))
        self.o = W["o"].values.astype(float)
        self.c = W["c"].values.astype(float)
        self.c_ff = W["c"].ffill().values.astype(float)
        self.o_adj = Wadj["o"].values.astype(float)
        self.c_adj = Wadj["c"].ffill().values.astype(float)
        self.score = sc.values
        self.U = U.values
        self.fc = cfg["cost_rt_pct"] / 200.0      # tek yön maliyet oranı

    def picks(self, i, n=C.N_PICKS):
        row = self.score[i]
        m = self.U[i] & np.isfinite(row)
        if m.sum() < 30:
            return []
        idx = np.where(m)[0]
        order = idx[np.argsort(-row[idx], kind="stable")][:n]
        return [self.tickers[j] for j in order]

    def universe(self, i):
        return [self.tickers[j] for j in np.where(self.U[i])[0]]

    def open_px(self, i, t):
        j = self.tidx.get(t)
        if j is None:
            return float("nan")
        return self.o[i, j]

    def close_px(self, i, t):
        j = self.tidx.get(t)
        if j is None:
            return float("nan")
        return self.c_ff[i, j]


def new_market_state(mk, cash):
    return {"n": -1, "last_date": None, "cash": float(cash), "positions": [], "pending": [],
            "cycles": [], "closed_cycles": [], "nav_hist": [], "trades": [], "peak": float(cash),
            "paused": False, "spec_name": None, "events": []}


def _pos_value(md, i, st):
    v = 0.0
    for p in st["positions"]:
        px = md.close_px(i, p["t"])
        if not np.isfinite(px):
            px = p.get("last_px", p["entry_raw"])
        p["last_px"] = float(px)
        v += p["qty"] * px
    return v


def market_nav(md, i, st):
    return st["cash"] + _pos_value(md, i, st)


# ------------------------------------------------------------------ bir günlük adım
def step_market(md: MarketData, i: int, st: dict, ctx: dict):
    """i: md.dates içindeki gün indisi. ctx: {'target_local': kolun hedef değeri (yerel para),
    'exposure': sigorta katsayısı, 'reduce': bool (sigorta yeni tetiklendi)}.  Olayları st['events']'e yazar."""
    d = md.dates[i]
    ev = {"date": str(d.date()), "fills": [], "orders": [], "stops": [], "notes": []}
    st["n"] += 1
    n = st["n"]
    st["last_date"] = str(d.date())

    # 0) bölünme / bedelsiz düzeltmeleri
    for p in st["positions"]:
        j = md.tidx.get(p["t"])
        if j is None:
            continue
        f = 1.0
        if md.event is not None and md.event[i, j] != 1.0:          # bugün düzeltilmemiş bölünme olayı
            f = float(md.event[i, j])
        else:                                                       # Yahoo geçmişi sonradan düzeltti mi?
            ie = md.didx.get(pd.Timestamp(p.get("ref_date", p["entry_date"])))
            if ie is not None and p.get("ref_px"):
                cur = md.c[ie, j]
                if np.isfinite(cur) and cur > 0 and abs(cur / p["ref_px"] - 1) > 0.05:
                    f = float(cur / p["ref_px"])
        if f != 1.0 and np.isfinite(f) and f > 0:
            p["qty"] /= f
            p["entry_px"] *= f
            p["entry_raw"] *= f
            # referansı bugünün (olay sonrası) kapanışına taşı: geçmiş sonradan düzeltilse de çift ölçekleme olmaz
            if np.isfinite(md.c[i, j]) and md.c[i, j] > 0:
                p["ref_date"], p["ref_px"] = str(d.date()), float(md.c[i, j])
            for o in st["pending"]:
                if o.get("pos_id") == p["id"] and o["side"] == "sell":
                    o["qty"] /= f
            ev["notes"].append(f"{p['t']}: bölünme/bedelsiz düzeltmesi uygulandı (oran {f:.3f})")

    # 0b) nakit faizi (TL: para piyasası/mevduat, $: hazine bonosu) — günlük
    if ctx.get("cash_rate", 0) and st["cash"] > 0:
        st["cash"] *= (1 + ctx["cash_rate"]) ** (1 / 252)

    # 1) bekleyen emirleri açılışta gerçekleştir (önce satışlar, sonra alışlar)
    keep = []
    sells = [o for o in st["pending"] if o["side"] == "sell"]
    buys = [o for o in st["pending"] if o["side"] == "buy"]
    for o in sells:
        pos = next((p for p in st["positions"] if p["id"] == o["pos_id"]), None)
        if pos is None:
            continue
        px = md.open_px(i, o["t"])
        if not np.isfinite(px) or px <= 0:
            o["wait"] = o.get("wait", 0) + 1
            if o["wait"] < 5:
                keep.append(o)
                continue
            px = md.close_px(i, o["t"])      # 5 gün işlem görmezse son kapanıştan çık
            if not np.isfinite(px):
                px = pos.get("last_px", pos["entry_raw"])
        qty = min(o["qty"], pos["qty"])
        fill = px * (1 - md.fc)
        st["cash"] += qty * fill
        pos["qty"] -= qty
        ret = (fill / pos["entry_px"] - 1) * 100
        tr = {"market": md.mk, "ticker": o["t"], "tranche": pos["tranche"], "entry_date": pos["entry_date"],
              "entry_px": round(pos["entry_raw"], 4), "exit_date": str(d.date()), "exit_px": round(px, 4),
              "qty": qty, "ret_pct": round(ret, 2), "pnl": round(qty * (fill - pos["entry_px"]), 2),
              "reason": o["reason"]}
        st["trades"].append(tr)
        ev["fills"].append(tr)
        if pos["qty"] <= 1e-9:
            st["positions"] = [p for p in st["positions"] if p["id"] != pos["id"]]
    for o in buys:
        px = md.open_px(i, o["t"])
        if not np.isfinite(px) or px <= 0:
            ev["notes"].append(f"{o['t']} açılışta işlem görmedi, alım iptal")
            continue
        amt = min(o["amount"], st["cash"])
        if amt <= 0:
            continue
        fill = px * (1 + md.fc)
        qty = amt / fill
        st["cash"] -= amt
        ex = next((p for p in st["positions"] if p["t"] == o["t"] and p["tranche"] == o["tranche"]), None)
        if ex is not None:     # mevcut pozisyona ekleme (yeniden boyutlandırma): ortalama maliyet
            tot = ex["qty"] + qty
            ex["entry_px"] = (ex["entry_px"] * ex["qty"] + fill * qty) / tot
            ex["entry_raw"] = (ex["entry_raw"] * ex["qty"] + px * qty) / tot
            ex["qty"] = tot
        else:
            pid = f"{md.mk}-{st['n']}-{o['t']}-{o['tranche']}"
            st["positions"].append({"id": pid, "t": o["t"], "tranche": o["tranche"], "qty": qty,
                                    "entry_px": fill, "entry_raw": px, "entry_date": str(d.date()),
                                    "ref_date": str(d.date()), "ref_px": float(md.c[i, md.tidx[o["t"]]]),
                                    "cycle_end": o.get("cycle_end")})
        ev["fills"].append({"market": md.mk, "ticker": o["t"], "side": "AL", "px": round(px, 4), "qty": qty,
                            "tranche": o["tranche"]})
    st["pending"] = keep

    # 1b) kapanan dilim döngüleri için denetim istatistiği (rastgele evrene karşı)
    for cy in list(st["cycles"]):
        if cy.get("exit_i") == i:
            ent, ex = cy["entry_i"], i
            def gross(names):
                r = []
                for t in names:
                    j = md.tidx.get(t)
                    if j is None or ex >= len(md.dates):
                        continue
                    a, b = md.o_adj[ent, j], md.o_adj[ex, j]
                    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
                        a, b = md.c_adj[ent, j], md.c_adj[ex, j]
                    if np.isfinite(a) and np.isfinite(b) and a > 0:
                        r.append(b / a - 1)
                return float(np.mean(r)) if r else float("nan")
            pr, ur = gross(cy["picks"]), gross(cy["universe"])
            st["closed_cycles"].append({"k": cy["k"], "start": cy["start"], "end": str(d.date()),
                                        "picks_ret": round(pr * 100, 3), "univ_ret": round(ur * 100, 3),
                                        "excess": round((pr - ur) * 100, 3)})
            st["cycles"].remove(cy)

    # 2) piyasa değeri
    nav = market_nav(md, i, st)

    # 3) felaket stopu
    pend_ids = {o.get("pos_id") for o in st["pending"]}
    for p in st["positions"]:
        if p["id"] in pend_ids:
            continue
        px = md.c[i, md.tidx[p["t"]]] if p["t"] in md.tidx else float("nan")
        if np.isfinite(px) and px <= p["entry_raw"] * (1 - C.CAT_STOP):
            st["pending"].append({"side": "sell", "t": p["t"], "qty": p["qty"], "pos_id": p["id"],
                                  "reason": "felaket stopu", "tranche": p["tranche"]})
            ev["stops"].append({"ticker": p["t"], "close": round(float(px), 4), "entry": round(p["entry_raw"], 4),
                                "chg_pct": round((px / p["entry_raw"] - 1) * 100, 1)})

    # 4) sigorta yeni tetiklendiyse tüm pozisyonları yarıya indir
    if ctx.get("reduce"):
        reduce_positions(st, ev)

    # 5) dilim yenileme
    shortfall = 0.0
    for k in range(C.N_TRANCHES):
        start = k * C.TRANCHE_STEP
        if n < start or (n - start) % C.HOLD_DAYS != 0:
            continue
        old = [p for p in st["positions"] if p["tranche"] == k]
        pend_ids = {o.get("pos_id") for o in st["pending"]}
        picks = [] if st.get("paused") else md.picks(i)
        budget = max(ctx["target_local"], 0) / C.N_TRANCHES * ctx.get("exposure", 1.0)
        per = budget / C.N_PICKS if picks else 0.0
        orders = []
        est_cash = st["cash"] - sum(o.get("amount", 0) for o in st["pending"] if o["side"] == "buy")
        for p in old:
            if p["id"] in pend_ids:
                continue
            val = p["qty"] * p.get("last_px", p["entry_raw"])
            if p["t"] not in picks:
                st["pending"].append({"side": "sell", "t": p["t"], "qty": p["qty"], "pos_id": p["id"],
                                      "reason": "süre doldu", "tranche": k})
                orders.append({"side": "SAT", "ticker": p["t"], "qty": p["qty"], "reason": "süre doldu (21 gün)",
                               "px": p.get("last_px"), "ret_pct": round((p.get("last_px", p["entry_raw"]) / p["entry_raw"] - 1) * 100, 1)})
                est_cash += val * (1 - md.fc)
        held = {p["t"]: p for p in old if p["id"] not in pend_ids}
        # 1 aylık işlem kuralı: listede kalan hisse için bu döngü kapanmış sayılır (maliyetsiz), yeni döngü
        # bugünkü kapanıştan başlar; felaket stopu da yeni girişe göre hesaplanır.
        for t in picks:
            if t in held:
                p = held[t]
                lp = p.get("last_px", p["entry_raw"])
                tr = {"market": md.mk, "ticker": t, "tranche": k, "entry_date": p["entry_date"],
                      "entry_px": round(p["entry_raw"], 4), "exit_date": str(d.date()), "exit_px": round(lp, 4),
                      "qty": p["qty"], "ret_pct": round((lp / p["entry_px"] - 1) * 100, 2),
                      "pnl": round(p["qty"] * (lp - p["entry_px"]), 2), "reason": "döngü yenilendi (devam)"}
                st["trades"].append(tr)
                p["entry_px"] = lp
                p["entry_raw"] = lp
                p["entry_date"] = str(d.date())
                p["cycle_end"] = n + C.HOLD_DAYS
                p["ref_date"] = str(d.date())
                p["ref_px"] = float(md.c[i, md.tidx[t]]) if np.isfinite(md.c[i, md.tidx[t]]) else lp
        buys = []
        for t in picks:
            if t in held:
                p = held[t]
                val = p["qty"] * p.get("last_px", p["entry_raw"])
                diff = per - val
                if abs(diff) <= max(0.2 * per, C.MIN_TRADE.get(md.mk, 0)):
                    orders.append({"side": "TUT", "ticker": t, "qty": p["qty"], "px": p.get("last_px")})
                    continue
                if diff < 0:
                    q = -diff / p.get("last_px", p["entry_raw"])
                    st["pending"].append({"side": "sell", "t": t, "qty": q, "pos_id": p["id"],
                                          "reason": "yeniden boyutlandırma", "tranche": k})
                    orders.append({"side": "AZALT", "ticker": t, "qty": q, "px": p.get("last_px")})
                    est_cash += -diff * (1 - md.fc)
                else:
                    buys.append([t, diff, "ARTIR"])
            else:
                buys.append([t, per, "AL"])
        need = sum(b[1] for b in buys)
        scale = 1.0
        if need > est_cash + 1e-9:
            scale = max(est_cash, 0) / need if need > 0 else 0
            shortfall += need - max(est_cash, 0)
        cycle_end = n + C.HOLD_DAYS
        for t, amt, tag in buys:
            a = amt * scale
            if a <= 0:
                continue
            st["pending"].append({"side": "buy", "t": t, "amount": a, "full": amt, "tranche": k, "cycle_end": cycle_end})
            px = md.c_ff[i, md.tidx[t]]
            orders.append({"side": tag, "ticker": t, "amount": round(a, 2), "px": float(px),
                           "qty_est": a / px if px > 0 else None,
                           "stop": round(float(px) * (1 - C.CAT_STOP), 4)})
        # denetim için döngü kaydı (giriş ertesi gün, çıkış 21 gün sonra ertesi gün)
        if picks:
            st["cycles"].append({"k": k, "start": str(d.date()), "picks": picks, "universe": md.universe(i),
                                 "entry_i": i + 1, "exit_i": i + 1 + C.HOLD_DAYS})
        ev["orders"].append({"tranche": k, "budget": round(budget, 2), "scale": round(scale, 3),
                             "orders": orders, "exit_date_est": cycle_end})
    ev["shortfall"] = round(shortfall, 2)

    st["nav_hist"].append([str(d.date()), round(nav, 4), round(st["cash"], 4)])
    st["peak"] = max(st.get("peak", nav), nav)
    ev["nav"] = nav
    st["events"].append(ev)
    st["events"] = st["events"][-30:]
    return ev


def reduce_positions(st, ev):
    """Sigorta: bekleyen satışı olmayan tüm pozisyonları ertesi açılışta INS_EXPOSURE boyutuna indir."""
    pend_ids = {o.get("pos_id") for o in st["pending"]}
    lst = []
    for p in st["positions"]:
        if p["id"] in pend_ids:
            continue
        q = p["qty"] * (1 - C.INS_EXPOSURE)
        st["pending"].append({"side": "sell", "t": p["t"], "qty": q, "pos_id": p["id"],
                              "reason": "sigorta (küçültme)", "tranche": p["tranche"]})
        lst.append({"side": "AZALT", "ticker": p["t"], "qty": q, "px": p.get("last_px"), "reason": "sigorta"})
    ev.setdefault("orders", []).append({"tranche": "sigorta", "budget": 0, "scale": 1, "orders": lst, "exit_date_est": None})
    ev.setdefault("notes", []).append("Portföy sigortası devrede: tüm pozisyonlar küçültülüyor")


def restore_positions(md, i, st, ev, target_local):
    """Sigorta kapanınca: yarıya inmiş pozisyonları dilim hedefine tamamlayacak alımlar (nakit yettiğince)."""
    per = max(target_local, 0) / C.N_TRANCHES / C.N_PICKS
    pend_ids = {o.get("pos_id") for o in st["pending"]}
    buys = []
    for p in st["positions"]:
        if p["id"] in pend_ids:
            continue
        val = p["qty"] * p.get("last_px", p["entry_raw"])
        if val < 0.8 * per:
            buys.append((p, per - val))
    need = sum(b for _, b in buys)
    cash = st["cash"] - sum(o.get("amount", 0) for o in st["pending"] if o["side"] == "buy")
    scale = min(1.0, max(cash, 0) / need) if need > 0 else 0
    lst = []
    for p, amt in buys:
        a = amt * scale
        if a > 0:
            st["pending"].append({"side": "buy", "t": p["t"], "amount": a, "full": a, "tranche": p["tranche"]})
            lst.append({"side": "ARTIR", "ticker": p["t"], "amount": round(a, 2), "px": p.get("last_px"),
                        "stop": round(p["entry_raw"] * (1 - C.CAT_STOP), 4)})
    ev.setdefault("orders", []).append({"tranche": "sigorta-bitti", "budget": 0, "scale": round(scale, 3),
                                        "orders": lst, "exit_date_est": None})


# ------------------------------------------------------------------ portföy düzeyi yardımcıları
def rp_weights(ret_bist_tl: pd.Series, ret_us_tl: pd.Series):
    """63 günlük oynaklığın tersi ile BIST/ABD ağırlığı (sınırlı)."""
    df = pd.concat([ret_bist_tl, ret_us_tl], axis=1).dropna().tail(C.RP_WINDOW)
    if len(df) < 40:
        return dict(C.RP_DEFAULT)
    vol = df.std()
    if (vol <= 0).any() or not np.isfinite(vol).all():
        return dict(C.RP_DEFAULT)
    iv = 1 / vol
    wb = float(iv.iloc[0] / iv.sum())
    wb = min(max(wb, C.RP_BOUNDS[0]), C.RP_BOUNDS[1])
    return {"bist": wb, "us": 1 - wb}


def insurance_update(pf: dict, total_tl: float):
    """Sigorta durumunu günceller. Dönüş: 'trigger' | 'release' | None."""
    pf["peak_tl"] = max(pf.get("peak_tl", total_tl), total_tl)
    dd = total_tl / pf["peak_tl"] - 1
    pf["dd"] = dd
    if not getattr(C, "INS_ENABLED", True):
        if pf.get("insurance"):
            pf["insurance"] = False
            return "release"
        return None
    if not pf.get("insurance") and dd < -C.INS_TRIGGER:
        pf["insurance"] = True
        return "trigger"
    if pf.get("insurance") and dd > -C.INS_RELEASE:
        pf["insurance"] = False
        return "release"
    return None


# ------------------------------------------------------------------ gölge (sanal) kol testi
def shadow_returns(md: MarketData, last_n=260):
    """Kolun kendi kendini finanse eden sanal kopyasını son last_n gün çalıştırır -> günlük getiri serisi.
    Risk paritesi ağırlıkları canlı geçmişten bağımsız olarak bununla hesaplanır."""
    i0 = max(0, len(md.dates) - last_n)
    st = new_market_state(md.mk, 1.0)
    navs = []
    for i in range(i0, len(md.dates)):
        nav_prev = market_nav(md, i, st) if st["n"] >= 0 else 1.0
        step_market(md, i, st, {"target_local": nav_prev, "exposure": 1.0})
        navs.append((md.dates[i], st["nav_hist"][-1][1]))
    s = pd.Series([v for _, v in navs], index=[d for d, _ in navs])
    return s.pct_change().dropna()
