"""Global Momentum Quant — Streamlit paneli (repo'daki data/ ve backtest_ref/ dosyalarını okur)."""
import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import config as C

st.set_page_config(page_title="Global Momentum Quant", page_icon="🌍", layout="wide")


@st.cache_data(ttl=300)
def load():
    out = {}
    if os.path.exists(C.STATE_FILE):
        with open(C.STATE_FILE, encoding="utf-8") as f:
            out["state"] = json.load(f)
    for k, p in (("nav", C.NAV_FILE), ("trades", C.TRADES_FILE)):
        if os.path.exists(p):
            out[k] = pd.read_csv(p)
    if os.path.exists(C.AUDIT_FILE):
        with open(C.AUDIT_FILE, encoding="utf-8") as f:
            out["audit"] = json.load(f)
    rp = os.path.join(C.REF_DIR, "equity.csv")
    if os.path.exists(rp):
        out["ref_eq"] = pd.read_csv(rp, parse_dates=["date"])
    sp = os.path.join(C.REF_DIR, "stats.json")
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            out["ref"] = json.load(f)
    return out


def tl(x):
    return f"{x:,.0f} TL".replace(",", ".")


D = load()
st.title("🌍 Global Momentum Quant")
st.caption("BIST momentum + ABD kalıntı momentum · 4 dilimli kademeli giriş · işlemler en fazla 21 işlem günü · "
           "felaket stopu %25 · risk paritesi · bot emir dosyası")

tabs = st.tabs(["Özet", "Açık pozisyonlar", "İşlemler", "Performans", "Öz denetim", "13 yıllık test"])

# ------------------------------------------------------------------ özet
with tabs[0]:
    S = D.get("state")
    if not S:
        st.info("Sistem henüz ilk kez çalışmadı. GitHub Actions ilk çalışmadan sonra burası dolacak.")
    else:
        pf = S["pf"]
        fx = pf.get("fx", 1)
        b = S["markets"]["bist"]
        u = S["markets"]["us"]
        bv = b["nav_hist"][-1][1] if b["nav_hist"] else b["cash"]
        uv = u["nav_hist"][-1][1] if u["nav_hist"] else u["cash"]
        tot = bv + uv * fx
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam portföy", tl(tot), f"{(tot / pf['capital_tl'] - 1) * 100:+.1f}% başlangıçtan")
        c2.metric("Zirveden", f"{pf.get('dd', 0) * 100:.1f}%")
        c3.metric("Sigorta", "DEVREDE" if pf.get("insurance") else "kapalı")
        c4.metric("USD/TRY", f"{fx:.2f}")
        c1, c2, c3 = st.columns(3)
        w = pf.get("weights", {})
        c1.metric("BIST kolu", tl(bv), f"hedef %{w.get('bist', 0) * 100:.0f} · gerçek %{bv / tot * 100:.0f}")
        c2.metric("ABD kolu", f"{uv:,.0f} $".replace(",", "."), f"hedef %{w.get('us', 0) * 100:.0f} · gerçek %{uv * fx / tot * 100:.0f}")
        c3.metric("Aktif stratejiler", f"{b.get('spec_name')} / {u.get('spec_name')}")
        for mk, s in (("bist", b), ("us", u)):
            if s.get("paused"):
                st.warning(f"{C.MARKETS[mk]['name']}: öz denetim nedeniyle yeni alımlar durduruldu.")
        st.subheader("Son mesajlar")
        if os.path.exists(C.LOG_FILE):
            with open(C.LOG_FILE, encoding="utf-8") as f:
                txt = f.read()[-6000:]
            st.text(txt.replace("<b>", "").replace("</b>", "").replace("<u>", "").replace("</u>", ""))

# ------------------------------------------------------------------ pozisyonlar
with tabs[1]:
    S = D.get("state")
    if S:
        for mk in ("bist", "us"):
            s = S["markets"][mk]
            ccy = C.MARKETS[mk]["ccy"]
            st.subheader(f"{C.MARKETS[mk]['name']} — {len(s['positions'])} pozisyon")
            if s["positions"]:
                rows = []
                n = s["n"]
                for p in s["positions"]:
                    lp = p.get("last_px", p["entry_raw"])
                    rows.append({"Hisse": p["t"], "Dilim": p["tranche"] + 1, "Giriş tarihi": p["entry_date"],
                                 f"Giriş ({ccy})": round(p["entry_raw"], 2), f"Son ({ccy})": round(lp, 2),
                                 "K/Z %": round((lp / p["entry_px"] - 1) * 100, 1),
                                 f"Felaket stopu ({ccy})": round(p["entry_raw"] * (1 - C.CAT_STOP), 2),
                                 "Adet": round(p["qty"], 2), f"Değer ({ccy})": round(p["qty"] * lp, 0),
                                 "Kalan gün (≈)": max((p.get("cycle_end") or n) - n, 0)})
                st.dataframe(pd.DataFrame(rows).sort_values("Dilim"), use_container_width=True, hide_index=True)
            if s.get("pending"):
                st.caption("Yarın açılışta bekleyen emirler: " + ", ".join(
                    f"{'AL' if o['side'] == 'buy' else 'SAT'} {o['t']}" for o in s["pending"]))

# ------------------------------------------------------------------ işlemler
with tabs[2]:
    T = D.get("trades")
    if T is None or T.empty:
        st.info("Henüz kapanan işlem yok.")
    else:
        for mk in ("bist", "us"):
            g = T[T.market == mk]
            if g.empty:
                continue
            w = g.ret_pct[g.ret_pct > 0]
            l = g.ret_pct[g.ret_pct <= 0]
            c = st.columns(5)
            c[0].metric(f"{C.MARKETS[mk]['name']} işlem", len(g))
            c[1].metric("Kârla kapanan", f"%{(g.ret_pct > 0).mean() * 100:.0f}")
            c[2].metric("İşlem başı net", f"%{g.ret_pct.mean():+.2f}")
            c[3].metric("Ort. kazanç / kayıp", f"%{w.mean() if len(w) else 0:+.1f} / %{l.mean() if len(l) else 0:+.1f}")
            c[4].metric("Kâr faktörü", f"{w.sum() / -l.sum():.2f}" if len(l) and l.sum() < 0 else "-")
        st.dataframe(T.iloc[::-1], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ performans
with tabs[3]:
    N = D.get("nav")
    if N is None or N.empty:
        st.info("Canlı öz sermaye eğrisi ilk çalışmalardan sonra oluşacak.")
    else:
        N["date"] = pd.to_datetime(N["date"])
        s = N.groupby("date")["total_tl"].last()
        dd = s / s.cummax() - 1
        df = pd.DataFrame({"date": s.index, "Portföy (TL)": s.values, "Düşüş %": dd.values * 100})
        st.altair_chart(alt.Chart(df).mark_line().encode(x="date:T", y=alt.Y("Portföy (TL):Q", scale=alt.Scale(type="log"))), use_container_width=True)
        st.altair_chart(alt.Chart(df).mark_area(opacity=0.4, color="#c0392b").encode(x="date:T", y="Düşüş %:Q"), use_container_width=True)
        yt = s.resample("YE").last()
        first = s.iloc[0]
        rows = []
        prev = first
        for d, v in yt.items():
            rows.append({"Yıl": d.year, "Getiri %": round((v / prev - 1) * 100, 1)})
            prev = v
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ------------------------------------------------------------------ öz denetim
with tabs[4]:
    S = D.get("state")
    if S:
        for mk in ("bist", "us"):
            s = S["markets"][mk]
            st.subheader(C.MARKETS[mk]["name"])
            st.write("Canlı denetim:", s.get("audit_status", "-"))
            cc = pd.DataFrame(s.get("closed_cycles", []))
            if not cc.empty:
                st.dataframe(cc.iloc[::-1], hide_index=True, use_container_width=True)
    A = D.get("audit")
    if A:
        for mk, rec in A.items():
            if rec.get("history"):
                h = rec["history"][-1]
                st.subheader(f"Derin denetim — {mk} ({h['date']}, aktif: {h['active']})")
                st.dataframe(pd.DataFrame(h["table"]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ test referansı
with tabs[5]:
    R = D.get("ref")
    E = D.get("ref_eq")
    if R:
        st.markdown(R.get("aciklama", ""))
        st.dataframe(pd.DataFrame(R["ozet"]), hide_index=True, use_container_width=True)
        if R.get("yillik"):
            st.dataframe(pd.DataFrame(R["yillik"]), hide_index=True, use_container_width=True)
        if R.get("islem"):
            st.dataframe(pd.DataFrame(R["islem"]), hide_index=True, use_container_width=True)
    if E is not None:
        m = E.melt("date", var_name="Seri", value_name="Değer")
        st.altair_chart(alt.Chart(m).mark_line().encode(x="date:T", y=alt.Y("Değer:Q", scale=alt.Scale(type="log")), color="Seri:N"),
                        use_container_width=True)
