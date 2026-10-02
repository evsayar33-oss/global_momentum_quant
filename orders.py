"""Bot için makine-okur emir dosyası.

Her çalışmada yazılır:
  data/orders_bist.json , data/orders_us.json   -> o pazarın EN SON emir listesi (her zaman güncel, boş olabilir)
  data/orders_history.jsonl                      -> tüm emirlerin satır satır geçmişi

Şema (her emir):
  id            benzersiz kimlik (aynı emir iki kez işlenmesin diye)
  action        BUY | SELL | REDUCE | INCREASE | HOLD
  symbol        borsa kodu (THYAO, AAPL)      yahoo_symbol: THYAO.IS / AAPL
  quantity      adet (BIST: tam sayı; ABD: 4 ondalık kesirli)
  sell_all      SELL için true -> pozisyonun tamamını sat
  amount        BUY/INCREASE için tutar (para birimi: currency)
  ref_price     sinyal günü kapanışı (limit değil, bilgi amaçlı)
  stop_price    felaket stopu seviyesi (kapanış bunun altına inerse sistem SELL üretir)
  order_type    MARKET_ON_OPEN  (ertesi işlem günü açılışta piyasa emri)
  execute_date  emrin uygulanacağı gün (tahmini; tatil olursa bir sonraki işlem günü)
  exit_date_est pozisyonun en geç kapanacağı gün (tahmini)
  reason        expiry | catastrophe_stop | resize | new_entry | rollover | insurance
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

REASON = {"süre doldu": "expiry", "süre doldu (21 gün)": "expiry", "felaket stopu": "catastrophe_stop",
          "yeniden boyutlandırma": "resize", "sigorta": "insurance", "sigorta (küçültme)": "insurance"}


def _q(q, mk):
    if q is None or not np.isfinite(q):
        return None
    return int(np.floor(q + 1e-9)) if mk == "bist" else round(float(q), 4)


def build(mk, ev, state):
    cfg = C.MARKETS[mk]
    st = state["markets"][mk]
    sig = pd.Timestamp(ev["date"])
    exe = (sig + pd.offsets.BDay(1)).date().isoformat()
    exit_est = (sig + pd.offsets.BDay(C.HOLD_DAYS + 1)).date().isoformat()
    sym = lambda t: f"{t}{cfg['suffix']}"
    out = []

    def add(action, t, **kw):
        oid = f"{mk}-{ev['date']}-{kw.get('tranche', 'x')}-{t}-{action}"
        out.append({"id": oid, "market": mk, "action": action, "symbol": t, "yahoo_symbol": sym(t),
                    "order_type": "MARKET_ON_OPEN", "execute_date": exe, "currency": "TRY" if mk == "bist" else "USD",
                    **kw})

    for blk in ev.get("orders", []):
        k = blk["tranche"]
        for o in blk["orders"]:
            side = o["side"]
            if side == "SAT":
                add("SELL", o["ticker"], tranche=k, quantity=_q(o.get("qty"), mk), sell_all=True,
                    ref_price=o.get("px"), reason="expiry")
            elif side == "AZALT":
                add("REDUCE", o["ticker"], tranche=k, quantity=_q(o.get("qty"), mk), sell_all=False,
                    ref_price=o.get("px"), reason=REASON.get(o.get("reason", ""), "resize"))
            elif side in ("AL", "ARTIR"):
                px = o.get("px")
                qty = o["amount"] / px if px else None
                add("BUY" if side == "AL" else "INCREASE", o["ticker"], tranche=k, quantity=_q(qty, mk),
                    amount=round(o["amount"], 2), ref_price=px, stop_price=o.get("stop"),
                    exit_date_est=exit_est if isinstance(k, int) else None,
                    reason="new_entry" if side == "AL" else "resize")
            elif side == "TUT":
                add("HOLD", o["ticker"], tranche=k, quantity=_q(o.get("qty"), mk), ref_price=o.get("px"),
                    exit_date_est=exit_est, reason="rollover")
    # felaket stopu satışları
    stop_t = {s["ticker"] for s in ev.get("stops", [])}
    for p in st["pending"]:
        if p["side"] == "sell" and p.get("reason") == "felaket stopu" and p["t"] in stop_t:
            add("SELL", p["t"], tranche=p.get("tranche"), quantity=_q(p["qty"], mk), sell_all=True,
                ref_price=next((s["close"] for s in ev["stops"] if s["ticker"] == p["t"]), None),
                reason="catastrophe_stop")
    positions = []
    n = st.get("n", 0)
    for p in st["positions"]:
        positions.append({"symbol": p["t"], "yahoo_symbol": sym(p["t"]), "tranche": p["tranche"],
                          "quantity": _q(p["qty"], mk), "entry_price": round(p["entry_raw"], 4),
                          "last_price": round(p.get("last_px", p["entry_raw"]), 4),
                          "stop_price": round(p["entry_raw"] * (1 - C.CAT_STOP), 4), "entry_date": p["entry_date"],
                          "days_left": max((p.get("cycle_end") or n) - n, 0)})
    tr = ev.get("transfer")
    return {
        "schema": "gmq.orders.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "market": mk,
        "signal_date": ev["date"],
        "execute_date": exe,
        "currency": "TRY" if mk == "bist" else "USD",
        "usdtry": round(float(ev.get("fx", state["pf"].get("fx", 0))), 4),
        "orders": out,
        "transfer": ({"from": tr["from"], "to": tr["to"], "amount_try": tr["tl"], "amount_usd": tr["usd"]} if tr else None),
        "portfolio": {"total_try": round(float(ev.get("total_tl", 0)), 2), "market_value": round(float(ev.get("nav", 0)), 2),
                      "cash": round(float(st["cash"]), 2), "weights": state["pf"].get("weights"),
                      "paused": bool(st.get("paused")), "strategy": st.get("spec_name")},
        "positions": positions,
    }


def write(mk, doc):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    path = os.path.join(C.DATA_DIR, f"orders_{mk}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, default=float)
    if doc["orders"] or doc["transfer"]:
        with open(os.path.join(C.DATA_DIR, "orders_history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({k: doc[k] for k in ("market", "signal_date", "execute_date", "orders", "transfer")},
                               ensure_ascii=False, default=float) + "\n")
    return path


def plain_lines(doc):
    """Telegram'a eklenen, botun da okuyabileceği sabit biçimli satırlar."""
    L = []
    for o in doc["orders"]:
        if o["action"] == "HOLD":
            continue
        q = "ALL" if o.get("sell_all") else o.get("quantity")
        s = f"{o['action']} {o['yahoo_symbol']} QTY={q}"
        if o.get("amount") is not None:
            s += f" AMT={o['amount']}"
        if o.get("stop_price") is not None:
            s += f" STOP={o['stop_price']}"
        s += f" {o['order_type']} {o['execute_date']}"
        L.append(s)
    if doc.get("transfer"):
        t = doc["transfer"]
        L.append(f"TRANSFER {t['from'].upper()}->{t['to'].upper()} TRY={t['amount_try']} USD={t['amount_usd']}")
    return L
