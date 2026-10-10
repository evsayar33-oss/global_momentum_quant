"""SEC EDGAR (ücretsiz, resmi) — zaman açısından doğru (point-in-time) temel veriler.

Kaynak: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json  (anahtar gerekmez; SEC kuralı: User-Agent + ≤10 istek/sn)
Kural: her değer, SEC'e İLK dosyalandığı tarihten (filed) SONRAKİ işlem gününden itibaren "bilinir".
       Sonradan yapılan düzeltmeler (restatement) aynı dönem için daha geç tarihli kayıt olarak gelir ve YOK SAYILIR.

Üretilen iki sinyal (ABD):
  * SUE (standartlaştırılmış kazanç sürprizi, mevsimsel rastgele yürüyüş): (HBK_q − HBK_{q−4}) / std(son 8 fark)
    Bilanço sonrası kayma (PEAD) literatürü: Bernard & Thomas (1989). Yalnızca dosyalamadan sonraki 91 gün "taze" sayılır.
  * GP/A (brüt kârlılık / toplam varlık): Novy-Marx (2013). Son yıllık rapor, 15 ay geçerli.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

UA = os.environ.get("SEC_USER_AGENT", "GMQ Research gmq-research@users.noreply.github.com")
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasic")
GP_TAGS = ("GrossProfit",)
REV_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
            "RevenueFromContractWithCustomerIncludingAssessedTax")
COST_TAGS = ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold")
SUE_FRESH_DAYS = 91
GPA_FRESH_DAYS = 456


def _get(url, session, retries=3):
    for a in range(retries):
        try:
            r = session.get(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(1.5 * (a + 1))
    return None


def ticker_cik_map(session):
    js = _get("https://www.sec.gov/files/company_tickers.json", session)
    out = {}
    for v in (js or {}).values():
        out[str(v["ticker"]).upper().replace(".", "-")] = int(v["cik_str"])
    return out


def _entries(facts, tag, unit_pred):
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag)
    if not node:
        return []
    out = []
    for unit, lst in node.get("units", {}).items():
        if unit_pred(unit):
            out += lst
    return out


def _dur(e):
    try:
        return (pd.Timestamp(e["end"]) - pd.Timestamp(e["start"])).days
    except Exception:
        return None


def _first_filed(entries, key):
    """Aynı dönem (key) için İLK dosyalanan değer (sonraki düzeltmeler yok sayılır)."""
    best = {}
    for e in entries:
        k = key(e)
        if k is None or e.get("filed") is None or e.get("val") is None:
            continue
        if k not in best or e["filed"] < best[k]["filed"]:
            best[k] = e
    return best


def parse_eps_quarters(facts):
    ents = []
    for tag in EPS_TAGS:
        ents = _entries(facts, tag, lambda u: "shares" in u.lower())
        if ents:
            break
    if not ents:
        return pd.DataFrame(columns=["end", "val", "filed"])
    q = _first_filed([e for e in ents if (_dur(e) or 0) in range(75, 106)], key=lambda e: e["end"])
    fy = _first_filed([e for e in ents if (_dur(e) or 0) in range(340, 390)], key=lambda e: e["end"])
    rows = [{"end": pd.Timestamp(k), "val": float(v["val"]), "filed": pd.Timestamp(v["filed"])} for k, v in q.items()]
    # Q4 genellikle yalnızca yıllık raporda: Q4 = FY − (aynı mali yıldaki 3 çeyrek), FY dosyalama tarihinde bilinir
    qdf = pd.DataFrame(rows)
    for k, v in fy.items():
        e = pd.Timestamp(k)
        if len(qdf) and (abs((qdf["end"] - e).dt.days) <= 10).any():
            continue
        inside = qdf[(qdf["end"] < e) & (qdf["end"] > e - pd.Timedelta(days=370))] if len(qdf) else qdf
        if len(inside) == 3:
            rows.append({"end": e, "val": float(v["val"]) - float(inside["val"].sum()), "filed": pd.Timestamp(v["filed"])})
    out = pd.DataFrame(rows)
    return out.sort_values("end").reset_index(drop=True) if len(out) else pd.DataFrame(columns=["end", "val", "filed"])


def sue_series(q):
    """Her çeyrek için SUE ve bilinme tarihi."""
    if q is None or len(q) < 6:
        return pd.DataFrame(columns=["known", "sue"])
    q = q.sort_values("end").reset_index(drop=True)
    diffs = []
    for i, r in q.iterrows():
        prev = q[(abs((q["end"] - (r["end"] - pd.Timedelta(days=365))).dt.days) <= 20)]
        d = r["val"] - prev["val"].iloc[0] if len(prev) else np.nan
        diffs.append(d)
    q["d"] = diffs
    out = []
    for i in range(len(q)):
        hist = q["d"].iloc[max(0, i - 8):i].dropna()
        if len(hist) >= 4 and np.isfinite(q["d"].iloc[i]):
            sd = hist.std(ddof=1)
            if sd > 0:
                # bilinme: dosyalama ile önceki dönemin dosyalaması arasından geç olan (Q4 türetimi için)
                out.append({"known": q["filed"].iloc[i], "sue": float(np.clip(q["d"].iloc[i] / sd, -10, 10))})
    return pd.DataFrame(out)


def gpa_series(facts):
    def fy_map(tags):
        for tag in tags:
            ents = _entries(facts, tag, lambda u: u == "USD")
            m = _first_filed([e for e in ents if (_dur(e) or 0) in range(340, 390)], key=lambda e: e["end"])
            if m:
                return m
        return {}
    gp = fy_map(GP_TAGS)
    if not gp:
        rev, cost = fy_map(REV_TAGS), fy_map(COST_TAGS)
        gp = {k: {"val": float(rev[k]["val"]) - float(cost[k]["val"]), "filed": max(rev[k]["filed"], cost[k]["filed"])}
              for k in rev if k in cost}
    assets = _first_filed([e for e in _entries(facts, "Assets", lambda u: u == "USD") if e.get("end")],
                          key=lambda e: e["end"])
    out = []
    for k, v in gp.items():
        a = assets.get(k)
        if a and float(a["val"]) > 0:
            out.append({"known": pd.Timestamp(max(v["filed"], a["filed"])), "gpa": float(v["val"]) / float(a["val"])})
    return pd.DataFrame(out)


def download_all(tickers, cache_dir="data_cache/sec_parsed", log=print):
    """Her hisse için ayrıştırılmış (küçük) SUE ve GP/A tablosu. Ham JSON saklanmaz."""
    import requests
    os.makedirs(cache_dir, exist_ok=True)
    s = requests.Session()
    cmap = ticker_cik_map(s)
    if not cmap:
        raise RuntimeError("SEC ticker→CIK listesi alınamadı (User-Agent / ağ)")
    res, miss = {}, []
    for k, t in enumerate(tickers):
        cik = cmap.get(t)
        if cik is None:
            miss.append(t)
            continue
        path = os.path.join(cache_dir, f"{t}.json")
        if os.path.exists(path):
            d = json.load(open(path))
        else:
            facts = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", s)
            time.sleep(0.12)                                   # SEC sınırı: ≤10 istek/sn
            if facts is None:
                miss.append(t)
                continue
            su = sue_series(parse_eps_quarters(facts))
            gp = gpa_series(facts)
            d = {"sue": [[str(r.known.date()), r.sue] for r in su.itertuples()],
                 "gpa": [[str(r.known.date()), r.gpa] for r in gp.itertuples()]}
            json.dump(d, open(path, "w"))
        res[t] = d
        if k % 50 == 0:
            log(f"   SEC {k}/{len(tickers)}")
    return res, miss


def to_panels(res, dates, tickers):
    """Tarih x hisse: o gün kapanışında bilinen en son (taze) SUE ve GP/A. Bilinme = dosyalamadan SONRAKİ gün."""
    idx = pd.DatetimeIndex(dates)
    sue = pd.DataFrame(np.nan, index=idx, columns=tickers)
    gpa = pd.DataFrame(np.nan, index=idx, columns=tickers)
    for t in tickers:
        d = res.get(t)
        if not d:
            continue
        for key, frame, fresh in (("sue", sue, SUE_FRESH_DAYS), ("gpa", gpa, GPA_FRESH_DAYS)):
            if not d.get(key):
                continue
            s = pd.DataFrame(d[key], columns=["known", "v"])
            s["known"] = pd.to_datetime(s["known"]) + pd.Timedelta(days=1)       # dosyalama günü kullanılmaz
            s = s.sort_values("known").drop_duplicates("known", keep="last").set_index("known")["v"]
            kd = pd.Series(s.index, index=s.index)
            v = s.reindex(idx.union(s.index)).ffill().reindex(idx)
            last = kd.reindex(idx.union(kd.index)).ffill().reindex(idx)
            age = (pd.Series(idx, index=idx) - last).dt.days
            frame[t] = v.where(age <= fresh)
    return sue, gpa


def synthetic_panels(dates, tickers, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(dates)
    res = {}
    for t in tickers:
        ends = pd.date_range(idx[0] - pd.Timedelta(days=800), idx[-1], freq="QE")
        sue = [[str((e + pd.Timedelta(days=int(rng.integers(25, 45)))).date()), float(rng.normal())] for e in ends]
        gpa = [[str((e + pd.Timedelta(days=60)).date()), float(rng.uniform(0, 0.6))] for e in ends[3::4]]
        res[t] = {"sue": sue, "gpa": gpa}
    return res
