"""Ücretsiz veri katmanı (yfinance + Wikipedia + TradingView tarayıcı).

* Günlük OHLCV: Yahoo 'Close' bölünmeye göre düzeltilmiştir (auto_adjust=False), araştırmayla aynı.
* Önbellek: data_cache/ (git'e girmez; GitHub Actions cache ile günler arasında korunur).
  Önbellek yoksa / 6 günden eskiyse / pazartesiyse tam indirme, değilse son 1 ay indirilip birleştirilir.
  Örtüşen günlerde fiyat %2'den fazla farklıysa (bölünme / bedelsiz) o hissenin tamamı yeniden indirilir.
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

CACHE_DIR = os.path.join(C.BASE, "data_cache")
COLS = ["open", "high", "low", "close", "volume"]


def _yf():
    import yfinance as yf
    return yf


def norm_us(t):
    return str(t).upper().strip().replace(".", "-").replace("/", "-")


# ------------------------------------------------------------------ evren
def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_universe(mk):
    u = _read_json(os.path.join(C.BASE, C.MARKETS[mk]["universe_file"]))
    if mk == "us":
        t = u.get("members") or u.get("tickers") or []
        return sorted({norm_us(x) for x in t})
    return sorted({str(x).upper().strip() for x in (u.get("tickers") or [])})


def refresh_universe(mk):
    """Ayda bir: ABD için Wikipedia S&P 500 üyeleri, BIST için TradingView tarayıcısı (yeni halka arzlar)."""
    path = os.path.join(C.BASE, C.MARKETS[mk]["universe_file"])
    u = _read_json(path)
    try:
        import requests
        if mk == "us":
            html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                                headers={"User-Agent": "Mozilla/5.0 (research bot)"}, timeout=25).text
            for t in pd.read_html(io.StringIO(html)):
                cols = [str(c).lower() for c in t.columns]
                if "symbol" in cols:
                    t.columns = cols
                    mem = [norm_us(x) for x in t["symbol"].dropna()]
                    sec_col = next((c for c in cols if "gics sector" in c), None)
                    if len(mem) > 400:
                        u["members"] = sorted(mem)
                        u["tickers"] = sorted(set(u.get("tickers") or []) | set(mem))
                        if sec_col:
                            sec = dict(u.get("sectors") or {})
                            sec.update(dict(zip(mem, t[sec_col])))
                            u["sectors"] = sec
                        break
        else:
            payload = {"filter": [{"left": "type", "operation": "equal", "right": "stock"}],
                       "columns": ["name"], "range": [0, 1000]}
            r = requests.post("https://scanner.tradingview.com/turkey/scan", json=payload, timeout=20,
                              headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.tradingview.com/"})
            names = [it["d"][0] for it in r.json().get("data", []) if it.get("d")]
            if len(names) > 300:
                u["tickers"] = sorted(set(u.get("tickers") or []) | {str(x).upper() for x in names})
        u["updated"] = datetime.now().strftime("%Y-%m-%d")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(u, f, ensure_ascii=False, indent=1)
        return True
    except Exception as exc:
        print(f"Evren güncellenemedi ({mk}): {exc}")
        return False


# ------------------------------------------------------------------ indirme
def _extract(data, sym, single):
    sub = None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            lv0 = set(map(str, data.columns.get_level_values(0)))
            lv1 = set(map(str, data.columns.get_level_values(1)))
            if sym in lv0:
                sub = data[sym]
            elif sym in lv1:
                sub = data.xs(sym, axis=1, level=1)
        elif single:
            sub = data
    except Exception:
        sub = None
    if sub is None or len(sub) == 0:
        return None
    sub = sub.copy()
    if isinstance(sub.columns, pd.MultiIndex):
        sub.columns = sub.columns.get_level_values(-1)
    sub.columns = [str(c).lower() for c in sub.columns]
    if not set(COLS).issubset(sub.columns):
        return None
    sub = sub[COLS].dropna(subset=["close"])
    return sub if len(sub) else None


def download(tickers, suffix, start=None, period=None, retries=2):
    yf = _yf()
    rows = []
    tickers = [t for t in tickers if t]
    for k in range(0, len(tickers), C.DOWNLOAD_CHUNK):
        chunk = tickers[k:k + C.DOWNLOAD_CHUNK]
        syms = [f"{t}{suffix}" for t in chunk]
        data = None
        for a in range(retries + 1):
            try:
                kw = dict(interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
                if start:
                    kw["start"] = start
                else:
                    kw["period"] = period or "1mo"
                data = yf.download(syms, **kw)
                if data is not None and not data.empty:
                    break
            except Exception as exc:
                print(f"yfinance hata ({a + 1}): {exc}")
            time.sleep(2 + 3 * a)
        if data is None or data.empty:
            continue
        single = len(syms) == 1
        for t, s in zip(chunk, syms):
            sub = _extract(data, s, single)
            if sub is None:
                continue
            sub = sub.reset_index()
            sub = sub.rename(columns={sub.columns[0]: "tarih"})
            sub["ticker"] = t
            rows.append(sub[["tarih", "ticker"] + COLS])
        time.sleep(0.5)
    if not rows:
        return pd.DataFrame(columns=["tarih", "ticker"] + COLS)
    out = pd.concat(rows, ignore_index=True)
    out["tarih"] = pd.to_datetime(out["tarih"]).dt.tz_localize(None).dt.normalize()
    out = out[(out["close"] > 0) & (out["high"] >= out["low"])]
    return out


def _cache_path(mk):
    return os.path.join(CACHE_DIR, f"ohlcv_{mk}.csv.gz")


def get_panel(mk, years=C.HISTORY_YEARS, force_full=False):
    """Uzun format panel döndürür (tarih, ticker, OHLCV)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cfg = C.MARKETS[mk]
    tickers = load_universe(mk)
    start = (pd.Timestamp.now() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    path = _cache_path(mk)
    cached = None
    if os.path.exists(path) and not force_full:
        try:
            cached = pd.read_csv(path, parse_dates=["tarih"])
        except Exception:
            cached = None
    full = (cached is None or cached.empty or
            (pd.Timestamp.now() - cached["tarih"].max()).days > 6 or
            pd.Timestamp.now().weekday() == 0 or
            cached["tarih"].min() > pd.Timestamp(start) + pd.Timedelta(days=30))
    if full:
        print(f"📥 {mk}: tam indirme ({len(tickers)} hisse, {years} yıl)")
        panel = download(tickers, cfg["suffix"], start=start)
    else:
        print(f"📥 {mk}: artımlı indirme (son 1 ay)")
        new = download(tickers, cfg["suffix"], period="1mo")
        # bölünme kontrolü
        m = new.merge(cached, on=["tarih", "ticker"], suffixes=("", "_o"))
        bad = m[(m["close"] / m["close_o"] - 1).abs() > 0.02]["ticker"].unique().tolist()
        missing = sorted(set(tickers) - set(cached["ticker"]))
        redo = sorted(set(bad) | set(missing))
        if redo:
            print(f"   ↻ yeniden indirilecek {len(redo)} hisse (bölünme/yeni)")
            fresh = download(redo, cfg["suffix"], start=start)
            cached = cached[~cached["ticker"].isin(redo)]
            new = pd.concat([new[~new["ticker"].isin(redo)], fresh])
        panel = pd.concat([cached[~cached.set_index(["tarih", "ticker"]).index.isin(
            new.set_index(["tarih", "ticker"]).index)], new])
    panel = panel[panel["tarih"] >= pd.Timestamp(start)].drop_duplicates(["tarih", "ticker"], keep="last")
    panel = panel.sort_values(["tarih", "ticker"])
    if len(panel):
        panel.to_csv(path, index=False, compression="gzip")
    return panel


def to_wide(panel):
    W = {}
    for k, col in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close"), ("v", "volume")):
        W[k] = panel.pivot(index="tarih", columns="ticker", values=col).sort_index()
    cols = W["c"].columns
    return {k: v.reindex(columns=cols) for k, v in W.items()}


def get_fx(years=2):
    """USD/TRY günlük kapanış serisi."""
    yf = _yf()
    for a in range(3):
        try:
            d = yf.download(C.FX_TICKER, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
            if d is not None and len(d):
                s = d["Close"]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                s = s[(s > 1) & (s < 1000)].dropna()
                if len(s):
                    return s
        except Exception as exc:
            print(f"Kur indirilemedi ({a + 1}): {exc}")
        time.sleep(3)
    return pd.Series(dtype=float)


def data_health(panel):
    """Son günün veri kapsamı, önceki 20 günün medyanına göre. %80 altıysa o gün 'henüz tam gelmedi' sayılır."""
    if panel is None or panel.empty:
        return {"ok": False, "ratio": 0.0, "last_date": None}
    cnt = panel.groupby("tarih")["ticker"].nunique().sort_index()
    last = cnt.index[-1]
    med = float(cnt.iloc[-21:-1].median()) if len(cnt) > 5 else float(cnt.iloc[-1])
    ratio = float(cnt.iloc[-1]) / max(med, 1.0)
    return {"ok": ratio >= 0.8, "ratio": round(ratio, 3), "last_date": str(last.date()), "count": int(cnt.iloc[-1])}
