"""Öz denetim — yavaş ve kanıta dayalı uyum.

1) Canlı denetim (her çalışmada): kapanan her dilimin getirisi aynı günlerde evrenin ortalamasıyla
   (rastgele seçim beklentisi) karşılaştırılır. Son 24 dilimde fark anlamlı biçimde negatifse
   (t < -2) o kolda yeni alımlar durur ve haber verilir; t > -0.5'e dönünce yeniden başlar.
2) Aylık derin denetim (python auditor.py --deep): son 8 yıllık veriyle önceden kayıtlı rakip stratejiler
   aynı kurallarla test edilir. Bir rakip son 36 ayda mevcut stratejiyi t>2 farkla geçerse ve bu
   3 ay üst üste tekrarlanırsa (ve son değişiklikten 1 yıl geçtiyse) strateji değiştirilir.
   Tek bir iyi ay hiçbir şeyi değiştirmez.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config as C


def tstat(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return float("nan"), len(x), float(np.mean(x)) if len(x) else float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))), len(x), float(x.mean())


def live_audit(state, mk):
    """Dönüş: (durum metni, not listesi)."""
    st = state["markets"][mk]
    cyc = st.get("closed_cycles", [])[-24:]
    t, n, m = tstat([c["excess"] for c in cyc])
    notes = []
    status = "veri birikiyor"
    if n >= C.AUDIT_MIN_TRANCHES and np.isfinite(t):
        status = f"son {n} dilim: rastgeleye göre %{m:+.2f}/dilim (t={t:.1f})"
        if t < C.AUDIT_PAUSE_T and not st.get("paused"):
            st["paused"] = True
            st["paused_since"] = datetime.now().strftime("%Y-%m-%d")
            notes.append(f"Öz denetim: {C.MARKETS[mk]['name']} kolu son {n} dilimde rastgele seçimden anlamlı "
                         f"derecede kötü (t={t:.1f}). Yeni alımlar durduruldu; aylık derin denetim rakipleri inceleyecek.")
        elif st.get("paused") and t > -0.5:
            st["paused"] = False
            notes.append(f"Öz denetim: {C.MARKETS[mk]['name']} kolu toparlandı (t={t:.1f}); yeni alımlar yeniden başlıyor.")
        elif t < C.AUDIT_WARN_T:
            notes.append(f"Öz denetim uyarısı: {C.MARKETS[mk]['name']} kolu son {n} dilimde zayıf (t={t:.1f}). İzleniyor.")
    # kol düşüşü (aktarımlardan arındırılmış performans endeksiyle)
    perf, peak = st.get("perf"), st.get("perf_peak")
    if perf and peak and perf / peak - 1 < -C.AUDIT_PAUSE_DD:
        notes.append(f"Uyarı: {C.MARKETS[mk]['name']} kolunun performansı zirvesinden %{(perf / peak - 1) * 100:.0f} aşağıda "
                     f"(13 yıllık testteki en kötü düşüşün ötesinde). Strateji derin denetimde inceleniyor.")
    # aynı uyarıyı her gün tekrarlama: 5 işlem gününde bir
    warned = st.setdefault("warned", {})
    out = []
    for n_ in notes:
        key = n_.split("(")[0][:60]
        last = warned.get(key, -999)
        if st["n"] - last >= 5:
            out.append(n_)
            warned[key] = st["n"]
    notes = out
    st["audit_status"] = status
    return status, notes


# ------------------------------------------------------------------ derin denetim
def evaluate(md, hold=C.HOLD_DAYS, n=C.N_PICKS):
    """Her 21 günde (4 kaydırmalı) en iyi n hisse: net getiri ve evrene göre fazla getiri serileri."""
    fc = md.cfg["cost_rt_pct"] / 100
    rows = []
    T = len(md.dates)
    for off in range(0, hold, C.TRANCHE_STEP):
        for i in range(off + 260, T - hold - 1, hold):
            picks = md.picks(i, n)
            if not picks:
                continue
            ent, ex = i + 1, i + 1 + hold
            a, b = md.o_adj[ent], md.o_adj[ex]
            r = b / a - 1
            u = md.U[i] & np.isfinite(r)
            if u.sum() < 30:
                continue
            pj = [md.tidx[t] for t in picks]
            pr = np.nanmean(r[pj])
            ur = np.nanmean(r[u])
            rows.append({"date": md.dates[i], "off": off, "net": (pr - fc) * 100, "excess": (pr - ur) * 100 - fc * 100})
    return pd.DataFrame(rows)


def deep_audit(mk, W, state):
    import engine as E
    st = state["markets"][mk]
    cur = st.get("spec_name") or list(C.CANDIDATES[mk])[0]
    res = {}
    for name, spec in C.CANDIDATES[mk].items():
        md = E.MarketData(mk, W, spec)
        res[name] = evaluate(md)
    end = max(df["date"].max() for df in res.values() if len(df))
    cut = end - pd.DateOffset(months=36)
    table = []
    base = res[cur]
    b0 = base[(base.off == 0) & (base.date >= cut)].set_index("date")["excess"]
    for name, df in res.items():
        if df.empty:
            continue
        rec = df[df.date >= cut]
        t_all, n_all, m_all = tstat(df[df.off == 0]["excess"])
        t_36, n_36, m_36 = tstat(rec[rec.off == 0]["excess"])
        c0 = rec[rec.off == 0].set_index("date")["excess"]
        j = pd.concat([c0, b0], axis=1).dropna()
        td, _, md_ = tstat((j.iloc[:, 0] - j.iloc[:, 1]).values) if len(j) else (float("nan"), 0, float("nan"))
        table.append({"strateji": name, "aktif": name == cur, "tüm_fazla": round(m_all, 2), "tüm_t": round(t_all, 2),
                      "36ay_fazla": round(m_36, 2), "36ay_t": round(t_36, 2), "36ay_net": round(rec["net"].mean(), 2),
                      "aktife_göre_t": round(td, 2) if name != cur else None})
    tab = pd.DataFrame(table)
    # değişim kararı
    au = _load_audit()
    mrec = au.setdefault(mk, {"streak": {}, "last_switch": None, "history": []})
    best = None
    for r in table:
        if r["aktif"]:
            continue
        ok = (r["aktife_göre_t"] is not None and r["aktife_göre_t"] > C.SWITCH_T_MARGIN and r["tüm_t"] > 2 and r["tüm_fazla"] > 0)
        mrec["streak"][r["strateji"]] = mrec["streak"].get(r["strateji"], 0) + 1 if ok else 0
        if ok and mrec["streak"][r["strateji"]] >= C.SWITCH_CONFIRM_MONTHS:
            if best is None or r["aktife_göre_t"] > best["aktife_göre_t"]:
                best = r
    notes = []
    switched = None
    if best is not None:
        last = mrec.get("last_switch")
        if not last or (pd.Timestamp.now() - pd.Timestamp(last)).days >= C.SWITCH_COOLDOWN_DAYS:
            st["spec_name"] = best["strateji"]
            st["paused"] = False
            mrec["last_switch"] = datetime.now().strftime("%Y-%m-%d")
            switched = best["strateji"]
            notes.append(f"Derin denetim: {C.MARKETS[mk]['name']} stratejisi {cur} → {best['strateji']} olarak değişti "
                         f"(son 36 ayda t={best['aktife_göre_t']:.1f} fark, 3 ay üst üste doğrulandı). "
                         "Açık pozisyonlar kendi süresinde kapanır, yeni dilimler yeni stratejiyle kurulur.")
    mrec["history"].append({"date": datetime.now().strftime("%Y-%m-%d"), "active": st.get("spec_name") or cur,
                            "table": table, "switched": switched})
    mrec["history"] = mrec["history"][-36:]
    _save_audit(au)
    return tab, notes


def _load_audit():
    try:
        with open(C.AUDIT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_audit(au):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    with open(C.AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(au, f, ensure_ascii=False, indent=1, default=str)
