"""Telegram mesajları (Türkçe, HTML biçimli)."""
from __future__ import annotations

import html
import os
import time

import config as C


def _fmt(x, nd=0):
    try:
        s = f"{x:,.{nd}f}"
    except Exception:
        return str(x)
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def qty(q, mk):
    if q is None:
        return "-"
    return _fmt(q, 0) if mk == "bist" else _fmt(q, 2)


def money(x, ccy):
    return f"{_fmt(x, 0)} {ccy}" if ccy == "TL" else f"{_fmt(x, 0)} $"


def px(x, ccy):
    if x is None:
        return "-"
    nd = 2 if x >= 1 else 4
    return f"{_fmt(x, nd)} {ccy}" if ccy == "TL" else f"{_fmt(x, nd)} $"


def send(text):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    with open(C.LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M')} =====\n{text}\n")
    if not C.TELEGRAM_TOKEN or not C.TELEGRAM_CHAT:
        print("(Telegram bilgisi yok, mesaj yalnızca günlüğe yazıldı)\n" + text)
        return False
    import requests
    ok = True
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > 3800:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur)
    for p in parts:
        for a in range(3):
            try:
                r = requests.post(f"https://api.telegram.org/bot{C.TELEGRAM_TOKEN}/sendMessage",
                                  data={"chat_id": C.TELEGRAM_CHAT, "text": p, "parse_mode": "HTML",
                                        "disable_web_page_preview": "true"}, timeout=20)
                if r.status_code == 200:
                    break
                print("Telegram hata:", r.status_code, r.text[:200])
            except Exception as exc:
                print("Telegram hata:", exc)
            time.sleep(2)
        else:
            ok = False
    return ok


def market_message(mk, ev, state, extra_notes=None, fx=None):
    cfg = C.MARKETS[mk]
    ccy = cfg["ccy"]
    flag = "🇹🇷" if mk == "bist" else "🇺🇸"
    st = state["markets"][mk]
    pf = state["pf"]
    fx = fx or pf.get("fx", 1.0)
    L = [f"{flag} <b>{cfg['name']}</b> — {ev['date']} kapanışı"]
    acts = [o for blk in ev.get("orders", []) for o in blk["orders"]]
    if acts or ev.get("stops"):
        L.append("\n📌 <b>Yarın AÇILIŞTA yapılacaklar</b>")
    for blk in ev.get("orders", []):
        k = blk["tranche"]
        if not blk["orders"]:
            continue
        title = f"Dilim {k + 1}" if isinstance(k, int) else ("Sigorta" if k == "sigorta" else "Sigorta bitti")
        extra = ""
        if isinstance(k, int):
            extra = f" (bütçe {money(blk['budget'], ccy)}, 21 işlem günü tutulur)"
            if blk.get("scale", 1) < 0.999:
                extra += f" ⚠️ nakit yetersiz, %{blk['scale'] * 100:.0f} ölçekli"
        L.append(f"<u>{title}</u>{extra}")
        for o in blk["orders"]:
            t = html.escape(o["ticker"])
            if o["side"] == "SAT":
                L.append(f"🔴 SAT <b>{t}</b> — tümü (~{qty(o['qty'], mk)} adet), {o.get('reason', '')}, döngü getirisi %{o.get('ret_pct', 0):+.1f}")
            elif o["side"] == "AZALT":
                L.append(f"🟠 AZALT <b>{t}</b> — ~{qty(o['qty'], mk)} adet sat ({o.get('reason', 'yeniden boyutlandırma')})")
            elif o["side"] in ("AL", "ARTIR"):
                q = o.get("qty_est") or (o["amount"] / o["px"] if o.get("px") else 0)
                if mk == "bist":
                    q = max(round(q), 0)
                tl = f" (≈ {money(o['amount'] * fx, 'TL')})" if mk == "us" else ""
                L.append(f"🟢 {o['side']} <b>{t}</b> — {money(o['amount'], ccy)}{tl} ≈ {qty(q, mk)} adet "
                         f"(son fiyat {px(o['px'], ccy)}, felaket stopu {px(o.get('stop'), ccy)})")
            elif o["side"] == "TUT":
                L.append(f"⚪ TUT <b>{t}</b> — listede kalmaya devam ediyor")
    for s in ev.get("stops", []):
        L.append(f"⛔ <b>FELAKET STOPU</b> {html.escape(s['ticker'])}: kapanış {px(s['close'], ccy)} "
                 f"(giriş {px(s['entry'], ccy)}, %{s['chg_pct']:+.1f}) → yarın açılışta SAT")
    if ev.get("transfer"):
        tr = ev["transfer"]
        src = "BIST (TL) hesabından" if tr["from"] == "bist" else "ABD ($) hesabından"
        dst = "ABD ($) hesabına" if tr["to"] == "us" else "BIST (TL) hesabına"
        L.append(f"🔁 <b>AKTARIM</b>: {src} {dst} {money(tr['tl'], 'TL')} (≈ {money(tr['usd'], '$')}) aktar")
    for n in ev.get("notes", []) + (extra_notes or []):
        L.append(f"ℹ️ {html.escape(n)}")
    # durum
    nav = ev.get("nav", 0)
    pos_val = nav - st["cash"]
    L.append("\n💼 <b>Durum</b>")
    L.append(f"{cfg['name']} kolu: {money(nav, ccy)} | açık pozisyon {len(st['positions'])} | nakit %{(st['cash'] / nav * 100 if nav else 0):.0f}")
    w = pf.get("weights", {})
    L.append(f"🌍 Toplam portföy: <b>{money(ev.get('total_tl', 0), 'TL')}</b> (zirveden %{ev.get('dd', 0) * 100:+.1f}) | "
             f"hedef BIST %{w.get('bist', 0) * 100:.0f} / ABD %{w.get('us', 0) * 100:.0f} | USD/TRY {_fmt(fx, 2)}")
    ins = f"🛡️ Sigorta: <b>DEVREDE</b> (pozisyonlar %{C.INS_EXPOSURE * 100:.0f} boyutta, zirveden %{C.INS_RELEASE * 100:.0f} içine dönünce tam boyut)" if pf.get("insurance") else \
        f"🛡️ Sigorta: kapalı (zirveden %{C.INS_TRIGGER * 100:.0f} düşüşte devreye girer)"
    if not getattr(C, "INS_ENABLED", True):
        ins = "🛡️ Portföy sigortası kapalı (her hissede %25 felaket stopu var)"
    L.append(ins)
    if st.get("paused"):
        L.append("⏸️ <b>Bu kolda yeni alımlar DURDURULDU</b> (öz denetim). Mevcut pozisyonlar kurallara göre kapanır.")
    return "\n".join(L)


def welcome_message(state):
    s = state["pf"]["initial_split"]
    return ("🚀 <b>Global Momentum Quant başladı</b>\n"
            f"Başlangıç sermayesi: {money(state['pf']['capital_tl'], 'TL')}\n"
            f"• BIST (TL) hesabı: {money(s['bist_tl'], 'TL')}\n"
            f"• ABD ($) hesabı: {money(s['us_usd'], '$')}\n"
            "Her kol 4 dilimde kurulur: ilk dilim yarın, sonrakiler 5'er işlem günü arayla. "
            "Her işlem en fazla 21 işlem günü (≈1 ay) tutulur. Felaket stopu: girişin %25 altı. "
            + ("Portföy sigortası: kapalı." if not getattr(C, "INS_ENABLED", True) else f"Portföy sigortası: toplam değer zirveden %{C.INS_TRIGGER * 100:.0f} düşerse pozisyonlar %{C.INS_EXPOSURE * 100:.0f} boyuta iner, toparlanınca tam boyuta döner."))
