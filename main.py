"""Giriş noktası.

  python main.py bist          # BIST kapanışından sonra (18:45 TR)
  python main.py us            # ABD kapanışından sonra (≈00:45 TR)
  python main.py auto          # GitHub Actions: zamanlamaya göre pazarı seçer
  python main.py audit         # aylık derin denetim (her iki pazar)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
import data as DA
import engine as E
import portfolio as P
import telegram_bot as TG
import auditor as AU
import orders as OR


# ------------------------------------------------------------------ durum dosyaları
def load_state():
    if os.path.exists(C.STATE_FILE):
        with open(C.STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_state(state):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    # işlemler ve kapanan döngüler CSV'ye taşınır, durum dosyası küçük kalır
    for mk, st in state["markets"].items():
        if st.get("trades"):
            df = pd.DataFrame(st["trades"])
            df.to_csv(C.TRADES_FILE, mode="a", header=not os.path.exists(C.TRADES_FILE), index=False)
            st["trades"] = []
        st["closed_cycles"] = st.get("closed_cycles", [])[-60:]
        st["nav_hist"] = st.get("nav_hist", [])[-500:]
    pf = state["pf"]
    if pf.get("nav_tl"):
        df = pd.DataFrame(pf["nav_tl"], columns=["date", "market", "total_tl", "usdtry"])
        df.to_csv(C.NAV_FILE, mode="a", header=not os.path.exists(C.NAV_FILE), index=False)
        pf["nav_tl"] = []
    tmp = C.STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, C.STATE_FILE)


def default_spec(mk):
    return list(C.CANDIDATES[mk])[0]


def pick_market_auto():
    sched = os.environ.get("GH_SCHEDULE", "")
    if sched:
        hour = int(sched.split()[1])
        return "bist" if hour < 19 else "us"
    h = datetime.utcnow().hour
    return "bist" if 12 <= h < 19 else "us"


# ------------------------------------------------------------------ günlük çalışma
def run_market(mk):
    fx = DA.get_fx(3)
    state = load_state()
    first = state is None
    if first:
        if not len(fx):
            print("Kur alınamadı; ilk kurulum ertelendi.")
            return
        state = P.new_state(C.CAPITAL_TL, float(fx.iloc[-1]))
        state["created"] = datetime.now().strftime("%Y-%m-%d")
        for m in state["markets"]:
            state["markets"][m]["spec_name"] = default_spec(m)
        TG.send(TG.welcome_message(state))
    st = state["markets"][mk]
    st.setdefault("spec_name", default_spec(mk))

    # ayda bir evren güncellemesi
    um = state["pf"].setdefault("universe_month", {})
    month = datetime.now().strftime("%Y-%m")
    if um.get(mk) != month:
        DA.refresh_universe(mk)
        um[mk] = month

    panel = DA.get_panel(mk)
    health = DA.data_health(panel)
    print("veri sağlığı:", health)
    if panel.empty:
        TG.send(f"⚠️ {C.MARKETS[mk]['name']}: veri indirilemedi, bugün işlem yapılmadı. Bir sonraki çalışmada tekrar denenecek.")
        save_state(state)
        return
    W = DA.to_wide(panel)
    spec = C.CANDIDATES[mk].get(st["spec_name"]) or C.MARKETS[mk]["spec"]
    md = E.MarketData(mk, W, spec)

    # işlenecek günler
    last_ok = len(md.dates) - 1
    if not health["ok"] and str(md.dates[-1].date()) == health["last_date"]:
        last_ok -= 1                         # bugünkü veri eksik: bir sonraki çalışmayı bekle
        print("Son gün verisi eksik, bekleniyor.")
    if st["last_date"] is None:
        if C.START_DATE:
            cand = [i for i, d in enumerate(md.dates) if d >= pd.Timestamp(C.START_DATE)]
            todo = [i for i in cand if i <= last_ok][:1]
        else:
            todo = [last_ok] if last_ok >= 0 else []
    else:
        todo = [i for i, d in enumerate(md.dates) if d > pd.Timestamp(st["last_date"]) and i <= last_ok]
    if not todo:
        print("Yeni işlem günü yok.")
        save_state(state)
        return

    # risk paritesi için gölge seri (en fazla 15 günde bir yenilenir)
    sh = state["pf"].setdefault("shadow", {"bist": [], "us": []})
    if not sh.get(mk) or (pd.Timestamp(md.dates[-1]) - pd.Timestamp(sh[mk][-1][0])).days > 15:
        r = E.shadow_returns(md)
        r = P._to_tl_returns(r, mk, fx)
        sh[mk] = [[str(d.date()), float(v)] for d, v in r.tail(100).items()]

    def shadow_fn(m, d):
        lst = sh.get(m) or []
        if not lst:
            return None
        s = pd.Series([v for _, v in lst], index=pd.to_datetime([a for a, _ in lst]))
        return s[s.index < pd.Timestamp(d)]

    evs = []
    for i in todo:
        evs.append(P.process_day(state, mk, md, i, fx, shadow_fn))
    ev = evs[-1]

    status, notes = AU.live_audit(state, mk)
    if len(todo) > 1 and not first:
        notes.insert(0, f"{len(todo)} işlem günü birlikte işlendi (önceki çalışmalar kaçmış). Arada üretilen emirler "
                        "kağıt üzerinde gerçekleşmiş sayıldı; lütfen panelden pozisyonlarını kontrol et.")
    send = bool([o for b in ev.get("orders", []) for o in b["orders"]] or ev.get("stops") or ev.get("transfer")
                or ev.get("notes") or notes or ev.get("insurance") or pd.Timestamp(ev["date"]).weekday() == 4 or first)
    doc = OR.build(mk, ev, state)
    OR.write(mk, doc)
    msg = TG.market_message(mk, ev, state, notes)
    msg += f"\n🧪 Öz denetim: {status}"
    lines = OR.plain_lines(doc)
    bot_msg = None
    if lines:
        import html as _h
        bot_msg = "🤖 <b>Bot emirleri</b> (data/orders_" + mk + ".json)\n<pre>" + _h.escape("\n".join(lines[:60])) + "</pre>"
    if send:
        TG.send(msg)
        if bot_msg:
            TG.send(bot_msg)
    else:
        print(msg)
    save_state(state)


def run_audit():
    state = load_state()
    if state is None:
        print("Durum yok; önce günlük çalışma.")
        return
    lines = ["🔬 <b>Aylık derin denetim</b> (son 8 yıl, aynı kurallar, rastgeleye göre)"]
    for mk in ("bist", "us"):
        try:
            panel = DA.get_panel(mk, years=C.AUDIT_YEARS, force_full=True)
            W = DA.to_wide(panel)
            tab, notes = AU.deep_audit(mk, W, state)
            lines.append(f"\n<b>{C.MARKETS[mk]['name']}</b> (aktif: {state['markets'][mk].get('spec_name')})")
            for _, r in tab.iterrows():
                mark = "✅" if r["aktif"] else "▫️"
                vs = "" if r["aktif"] or r["aktife_göre_t"] is None or pd.isna(r["aktife_göre_t"]) else f", aktife göre t={r['aktife_göre_t']:.1f}"
                lines.append(f"{mark} {r['strateji']}: tüm dönem %{r['tüm_fazla']:+.2f}/dilim (t={r['tüm_t']:.1f}), "
                             f"son 36 ay %{r['36ay_fazla']:+.2f} (t={r['36ay_t']:.1f}{vs})")
            lines.append("Kural: rakip, aktif stratejiyi son 36 ayda t>2 farkla 3 ay üst üste geçerse değişir (yılda en fazla 1 kez).")
            lines += [f"ℹ️ {n}" for n in notes]
        except Exception as exc:
            lines.append(f"⚠️ {mk} denetimi tamamlanamadı: {exc}")
    TG.send("\n".join(lines))
    save_state(state)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "auto"
    if arg == "audit":
        run_audit()
    else:
        run_market(pick_market_auto() if arg == "auto" else arg)
