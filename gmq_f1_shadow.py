"""F1 gölge kaydı — canlı sistemi ETKİLEMEZ, yalnızca kayıt tutar (protokol v1.4 §H, gölge aşaması).

Ne yapar: Her ABD çalışmasından sonra data/orders_us.json'daki yeni ALIM emirleri için
"F1 filtresi olsaydı bu hisse elenir miydi?" sorusunu SEC verisiyle cevaplar ve tek satır kaydeder.
F1 kuralı: son bilançodaki kazanç sürprizi (SUE, ≤91 gün taze) ABD evreninin en kötü %20'sindeyse → ELENİRDİ.

Dosya boyutu (sabit / çok küçük):
  data/f1_shadow/sue_cache.json : hisse başına yalnızca son 2 SUE değeri (~500 hisse, ≈60–80 KB, büyümez)
  data/f1_shadow/log.csv        : alım başına 1 satır (≈90 bayt) → yılda ≈ 50–100 KB
Hata olursa sessizce çıkar (workflow'da continue-on-error), canlı emirler zaten üretilmiştir.

Değerlendirme (12–18 ay sonra): log.csv ile data/trades.csv birleştirilir;
f1_flag=1 olan alımların getirisi f1_flag=0 olanlardan anlamlı düşük mü?
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np

import gmq_sec_fundamentals as SF

BASE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(BASE, "data", "f1_shadow")
CACHE = os.path.join(DIR, "sue_cache.json")
LOG = os.path.join(DIR, "log.csv")
ORDERS = os.path.join(BASE, "data", "orders_us.json")
UNIVERSE = os.path.join(BASE, "data", "universe_us.json")
MAX_FETCH = int(os.environ.get("F1_MAX_FETCH", "80"))      # çalışma başına en fazla SEC isteği
RECHECK_DAYS = 7                                           # bir hisseyi en erken kaç günde bir yeniden kontrol et
NEW_QTR_DAYS = 75                                          # son bilinen bilanço bundan eskiyse yeni çeyrek beklenir
COLS = ["signal_date", "ticker", "tranche", "ref_price", "sue", "sue_pct", "f1_flag", "coverage", "logged_at"]


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _fresh_sue(entry, on_date):
    """on_date günü kapanışında bilinen taze SUE (dosyalamadan sonraki gün, ≤91 gün)."""
    best = None
    for known, v in entry.get("sue", []):
        k = datetime.strptime(known, "%Y-%m-%d").date() + timedelta(days=1)
        if k <= on_date and (on_date - k).days <= SF.SUE_FRESH_DAYS:
            if best is None or k > best[0]:
                best = (k, v)
    return None if best is None else float(best[1])


def main():
    orders = _load(ORDERS, {})
    buys = [o for o in orders.get("orders", []) if str(o.get("action", "")).upper() == "BUY"]
    sig = orders.get("signal_date")
    if not buys or not sig:
        print("F1 gölge: yeni ABD alımı yok.")
        return 0
    os.makedirs(DIR, exist_ok=True)
    done = set()
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as f:
            done = {(r["signal_date"], r["ticker"]) for r in csv.DictReader(f)}
    buys = [o for o in buys if (sig, o["symbol"]) not in done]
    if not buys:
        print("F1 gölge: bu sinyal günü zaten kaydedilmiş.")
        return 0

    uni = _load(UNIVERSE, {})
    members = sorted({str(t).upper() for t in (uni.get("members") or uni.get("tickers") or [])})
    cache = _load(CACHE, {})
    today = datetime.now().date()
    sig_d = datetime.strptime(sig, "%Y-%m-%d").date()

    def needs(t):
        e = cache.get(t)
        if not e:
            return True
        if (today - datetime.strptime(e["checked"], "%Y-%m-%d").date()).days < RECHECK_DAYS:
            return False
        last = max([k for k, _ in e.get("sue", [])], default=None)
        return last is None or (today - datetime.strptime(last, "%Y-%m-%d").date()).days > NEW_QTR_DAYS

    picks = [o["symbol"] for o in buys]
    queue = [t for t in picks if needs(t)] + [t for t in members if t not in picks and needs(t)]
    limit = MAX_FETCH if len(cache) >= 100 else 600            # ilk kurulum: önbellek bir kerede dolar
    queue = queue[:limit]
    if queue:
        import requests
        s = requests.Session()
        cmap = SF.ticker_cik_map(s)
        if not cmap:
            print(f"F1 gölge: SEC erişimi yok ({SF.LAST_ERROR}); bu çalışma atlandı.")
            return 0
        for t in queue:
            cik = cmap.get(t)
            sue = []
            if cik is not None:
                facts = SF._get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", s)
                time.sleep(0.12)
                if facts is not None:
                    ser = SF.sue_series(SF.parse_eps_quarters(facts))
                    sue = [[str(r.known.date()), round(float(r.sue), 4)] for r in ser.tail(2).itertuples()]
            cache[t] = {"checked": str(today), "sue": sue}
        cache = {t: v for t, v in cache.items() if t in set(members) | set(picks)}     # çıkan hisseleri temizle
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, separators=(",", ":"), sort_keys=True)

    vals = {t: _fresh_sue(cache.get(t, {}), sig_d) for t in members}
    have = np.array([v for v in vals.values() if v is not None])
    coverage = len(have) / max(len(members), 1)
    rows = []
    for o in buys:
        t = o["symbol"]
        v = _fresh_sue(cache.get(t, {}), sig_d)
        pct = float((have < v).mean() + 0.5 * (have == v).mean()) if (v is not None and len(have) >= 50) else None
        flag = "" if (pct is None or coverage < 0.5) else int(pct <= 0.20)      # kapsama yetersizse karar yazılmaz
        rows.append({"signal_date": sig, "ticker": t, "tranche": o.get("tranche"), "ref_price": round(float(o.get("ref_price", 0)), 4),
                     "sue": "" if v is None else round(v, 3), "sue_pct": "" if pct is None else round(pct, 3),
                     "f1_flag": flag, "coverage": round(coverage, 3), "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerows(rows)
    n_flag = sum(1 for r in rows if r["f1_flag"] == 1)
    print(f"F1 gölge: {len(rows)} alım kaydedildi · F1 elerdi: {n_flag} · SUE kapsaması %{coverage * 100:.0f} · "
          f"SEC isteği: {len(queue)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                     # gölge kayıt asla canlı akışı bozmaz
        print(f"F1 gölge: hata, atlandı ({exc})")
        sys.exit(0)
