"""Global Momentum Quant — Streamlit paneli (repo'daki data/ ve backtest_ref/ dosyalarını okur)."""
import json
import os
import re

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


def active_markets(S=None):
    fw = getattr(C, "FIXED_WEIGHTS", None)
    out = []
    for mk in ("bist", "us"):
        has_pos = bool(S and (S["markets"][mk].get("positions") or S["markets"][mk].get("pending")))
        if not fw or fw.get(mk, 0) > 0 or has_pos:
            out.append(mk)
    return out


def mval(s):
    if s.get("nav_post") is not None:
        return float(s["nav_post"])
    return s["nav_hist"][-1][1] if s["nav_hist"] else s["cash"]


def tl(x):
    return f"{x:,.0f} TL".replace(",", ".")


D = load()
st.title("🌍 Global Momentum Quant")
st.caption("ABD büyük şirketler · kalıntı + 12 ay momentum · 4 dilimli kademeli giriş · işlemler en fazla 21 işlem günü · "
           "felaket stopu %25 · bot emir dosyası")

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
        bv, uv = mval(b), mval(u)
        tot = bv + uv * fx
        AM = active_markets(S)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam portföy", tl(tot), f"{(tot / pf['capital_tl'] - 1) * 100:+.1f}% başlangıçtan")
        c2.metric("Zirveden", f"{pf.get('dd', 0) * 100:.1f}%")
        c3.metric("Sigorta", "DEVREDE" if pf.get("insurance") else "kapalı")
        c4.metric("USD/TRY", f"{fx:.2f}")
        w = pf.get("weights", {})
        cols = st.columns(len(AM) + 1)
        for j, mk in enumerate(AM):
            s = S["markets"][mk]
            v = bv if mk == "bist" else uv
            vtl = v if mk == "bist" else v * fx
            label = tl(v) if mk == "bist" else f"{v:,.0f} $".replace(",", ".")
            cols[j].metric(f"{C.MARKETS[mk]['name']} kolu", label,
                           f"hedef %{w.get(mk, 0) * 100:.0f} · gerçek %{vtl / tot * 100:.0f}" if tot else None)
        cols[-1].metric("Aktif strateji", " / ".join(str(S["markets"][mk].get("spec_name")) for mk in AM))
        for mk in AM:
            if S["markets"][mk].get("paused"):
                st.warning(f"{C.MARKETS[mk]['name']}: öz denetim nedeniyle yeni alımlar durduruldu.")
        st.subheader("Son mesajlar (en yeni üstte)")
        if os.path.exists(C.LOG_FILE):
            with open(C.LOG_FILE, encoding="utf-8") as f:
                txt = f.read()
            blocks = [b_.strip() for b_ in txt.split("===== ") if b_.strip()]
            for b_ in blocks[::-1][:8]:
                head, _, body = b_.partition("=====")
                clean = re.sub(r"<[^>]+>", "", body).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                with st.expander(head.strip(), expanded=False):
                    st.text(clean.strip())

# ------------------------------------------------------------------ pozisyonlar
with tabs[1]:
    S = D.get("state")
    if S:
        for mk in active_markets(S):
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
        for mk in active_markets(D.get("state")):
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
        S_ = D.get("state") or {}
        if S_.get("created"):
            N = N[N["date"] >= pd.Timestamp(S_["created"]) - pd.Timedelta(days=5)]
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
        for mk in active_markets(S):
            s = S["markets"][mk]
            st.subheader(C.MARKETS[mk]["name"])
            st.write("Canlı denetim:", s.get("audit_status", "-"))
            cc = pd.DataFrame(s.get("closed_cycles", []))
            if not cc.empty:
                st.dataframe(cc.iloc[::-1], hide_index=True, use_container_width=True)
    A = D.get("audit")
    if A:
        for mk, rec in A.items():
            if mk not in active_markets(D.get("state")):
                continue
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
