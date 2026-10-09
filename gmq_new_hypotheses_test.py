"""GMQ — Kapsamlı Yeni Hipotezler Testi (protokol v1.1, PROTOKOL_YENI_HIPOTEZLER.md).

Tek çalıştırma:  python gmq_new_hypotheses_test.py --years 14 --out-dir gmq_new_hypotheses_output
Yerel duman testi: python gmq_new_hypotheses_test.py --synthetic --out-dir smoke_new_hypotheses

İlkeler (protokolden):
  * Üretim dosyalarına dokunulmaz; GMQ'nun KENDİ motoru (engine.step_market + portfolio.process_day) ile
    TAM PORTFÖY yeniden oynatımı yapılır (emir, açılış fill'i, nakit, dilim, felaket stopu, risk paritesi).
  * Zaman bölmesi: eğitim | doğrulama | KİLİTLİ test. Aday taraması test başlangıcında DURDURULUR; test verisi
    yalnızca doğrulamada seçilen tek aday için ve bir kez işlenir.
  * v1.1 ekleri: (1) güç analizi / en küçük ölçülebilir etki, (2) Holm + White Reality Check,
    (3) plasebo (karıştırılmış/rastgele özellik) kontrolü, (4) hayatta kalma yanlılığının ölçülüp raporlanması
    (+ S&P 500 için Wikipedia değişiklik tablosundan geçmiş üyelik).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import engine as E
import portfolio as P

PROTOCOL_VERSION = "1.2"
MODE = "all"                # "all": karma portföy (v1.1) · "us": yalnızca ABD kolu, USD bazlı ölçüm (v1.2)
US_ONLY = {"bist": 0.0, "us": 1.0}
SEED = 1701
BOOT_B = 2000
BLOCK_MONTHS = 3
TEST_MONTHS = 36            # v1.1 madde 1: kilitli pencere mümkün olduğunca geniş (≈3 yıl)
VAL_MONTHS = 30
EMBARGO_CAL_DAYS = 32        # 21 işlem günü tutma + 1 gün giriş gecikmesi ≈ 32 takvim günü
WARMUP_IDX = 300             # sinyallerin oturması için ilk işlem günü indeksi (selftest ile aynı)
POOL_EXTRA = 10              # filtrelerde sıralı listeden kaç yedek isim bakılır
FILTER_PCT = 0.80            # filtre: kesitsel yüzdelik >= 0.80 olan aday elenir (önceden sabit)
GATE_MULT = 0.5              # piyasa kapısı: yeni dilim bütçesi yarıya iner (önceden sabit)
HOLM_ALPHA = 0.20            # doğrulamada seçim için Holm-düzeltilmiş tek yönlü p (kilitli test ikinci, bağımsız sınavdır)
PLACEBO_PCTL = 0.90          # aday doğrulama farkı, kendi tipindeki plasebo dağılımının %90'ını aşmalı
MIN_TRADE_SHARE = 0.70       # işlem sayısını yapay azaltma yasağı: BASE'in en az %70'i
MIN_TEST_TRADES = 300
MIN_TEST_TRADES_MKT = 100
SUBGROUP_MIN = 50


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ====================================================================== veri
def sp500_membership_history():
    """Wikipedia: güncel üyeler + 'Selected changes' tablosu -> geriye doğru üyelik. Başarısızsa None."""
    try:
        import io
        import requests
        html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                            headers={"User-Agent": "Mozilla/5.0 (GMQ research)"}, timeout=30).text
        tables = pd.read_html(io.StringIO(html))
        cur = None
        changes = None
        for t in tables:
            cols = [" ".join(map(str, c)).lower() if isinstance(c, tuple) else str(c).lower() for c in t.columns]
            if cur is None and any(c == "symbol" or c.startswith("symbol") for c in cols) and len(t) > 400:
                sc = t.columns[[i for i, c in enumerate(cols) if c.startswith("symbol")][0]]
                cur = {str(x).upper().replace(".", "-").strip() for x in t[sc].dropna()}
            elif changes is None and any("added" in c for c in cols) and any("removed" in c for c in cols):
                t = t.copy()
                t.columns = cols
                dc = next(c for c in cols if "date" in c)
                ac = next(c for c in cols if "added" in c and "ticker" in c)
                rc = next(c for c in cols if "removed" in c and "ticker" in c)
                ch = pd.DataFrame({"date": pd.to_datetime(t[dc], errors="coerce"),
                                   "added": t[ac].astype(str).str.upper().str.replace(".", "-", regex=False).str.strip(),
                                   "removed": t[rc].astype(str).str.upper().str.replace(".", "-", regex=False).str.strip()})
                ch = ch.dropna(subset=["date"])
                ch = ch.replace({"NAN": None, "": None})
                changes = ch.sort_values("date")
        if not cur or changes is None or len(changes) < 50:
            return None
        return {"current": sorted(cur), "changes": changes}
    except Exception as exc:
        log(f"⚠️ S&P üyelik geçmişi alınamadı: {exc}")
        return None


def membership_mask(hist, dates, tickers):
    cur = set(hist["current"])
    ch = hist["changes"]
    rows = []
    mem = set(cur)
    # tarihten geriye: d gününde üyelik = bugünkü küme, d'den SONRA olan değişiklikler geri alınmış
    events = ch.sort_values("date", ascending=False).to_dict("records")
    k = 0
    for d in sorted(dates, reverse=True):
        while k < len(events) and events[k]["date"] > d:
            e = events[k]
            if e["added"]:
                mem.discard(e["added"])
            if e["removed"]:
                mem.add(e["removed"])
            k += 1
        rows.append((d, frozenset(mem)))
    rows.reverse()
    tset = {t: j for j, t in enumerate(tickers)}
    arr = np.zeros((len(rows), len(tickers)), dtype=bool)
    for i, (_, s) in enumerate(rows):
        js = [tset[t] for t in s if t in tset]
        arr[i, js] = True
    return pd.DataFrame(arr, index=pd.DatetimeIndex([d for d, _ in rows]), columns=tickers)


def load_real(years):
    import data as DA
    log("📥 Veri indiriliyor (yfinance)…")
    panels = {mk: DA.get_panel(mk, years=years, force_full=True) for mk in ("bist", "us")}
    hist = sp500_membership_history()
    removed = []
    if hist is not None:
        start = pd.Timestamp.now() - pd.DateOffset(years=years)
        removed = sorted({t for t in hist["changes"].loc[hist["changes"]["date"] >= start, "removed"].dropna()}
                         - set(panels["us"]["ticker"].unique()))
        if removed:
            log(f"📥 Endeksten çıkmış {len(removed)} ABD hissesi deneniyor (hayatta kalma yanlılığını azaltmak için)")
            extra = DA.download(removed, "", start=start.strftime("%Y-%m-%d"))
            if len(extra):
                panels["us"] = pd.concat([panels["us"], extra]).drop_duplicates(["tarih", "ticker"], keep="first")
    fx = DA.get_fx(years)
    irx = None
    try:
        d = DA._yf().download("^IRX", period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
        s = d["Close"]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        irx = s.dropna()
    except Exception as exc:
        log(f"⚠️ ^IRX alınamadı (nakit faizi varsayılan): {exc}")
    W = {mk: DA.to_wide(p) for mk, p in panels.items()}
    return W, fx, irx, panels, hist, removed


def load_synthetic(seed=SEED):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-02", "2026-09-30")
    W, panels = {}, {}
    for mk, n in (("bist", 140), ("us", 160)):
        tick = [f"{mk.upper()}{i:03d}" for i in range(n)]
        T = len(dates)
        mkt = rng.normal(0.0004, 0.011, T)
        alpha = np.cumsum(rng.normal(0, 0.00012, (T, n)), axis=0)       # kalıcı (momentumlu) sürüklenme
        beta = rng.uniform(0.6, 1.4, n)
        r = mkt[:, None] * beta + alpha * 0.02 + rng.normal(0, 0.017, (T, n))
        r = np.clip(r, -0.095, 0.095)
        c = 20 * np.exp(np.cumsum(np.log1p(r), axis=0))
        o = c * (1 + rng.normal(0, 0.004, c.shape))
        h = np.maximum(c, o) * (1 + np.abs(rng.normal(0, 0.006, c.shape)))
        l = np.minimum(c, o) * (1 - np.abs(rng.normal(0, 0.006, c.shape)))
        v = np.exp(rng.normal(13, 1, n))[None, :] * np.exp(rng.normal(0, 0.4, c.shape))
        late = rng.choice(n, 10, replace=False)                          # sonradan halka arz
        for j in late:
            k = rng.integers(300, 900)
            for a in (o, h, l, c, v):
                a[:k, j] = np.nan
        mkdf = lambda a: pd.DataFrame(a, index=dates, columns=tick)
        W[mk] = {"o": mkdf(o), "h": mkdf(h), "l": mkdf(l), "c": mkdf(c), "v": mkdf(v)}
        lp = W[mk]["c"].stack().rename("close").reset_index()
        lp.columns = ["tarih", "ticker", "close"]
        panels[mk] = lp
    fx = pd.Series(np.exp(np.cumsum(rng.normal(0.0006, 0.006, len(dates)))) * 3.0, index=dates)
    return W, fx, None, panels, None, []


# ====================================================================== özellikler (yalnızca t ve öncesi)
class Feat:
    """Bir kolun tüm aday özellikleri; md.dates x md.tickers hizalı numpy dizileri."""

    def __init__(self, md: E.MarketData, W, fx: pd.Series | None):
        idx, cols = pd.DatetimeIndex(md.dates), md.tickers
        Wr = {k: W[k].reindex(index=idx, columns=cols) for k in ("o", "h", "l", "c", "v")}
        c_raw = Wr["c"]
        c = pd.DataFrame(md.c_adj, index=idx, columns=cols)
        o = pd.DataFrame(md.o_adj, index=idx, columns=cols)
        h = (Wr["h"] / c_raw) * c
        l = (Wr["l"] / c_raw) * c
        r = c.pct_change(fill_method=None).where(c_raw.notna())
        r = r.where(r.abs() <= md.cfg["max_move"])
        U = pd.DataFrame(md.U, index=idx, columns=cols)
        tv = (c_raw * Wr["v"]).where(Wr["v"] > 0)                         # ciro: bölünmeden etkilenmez
        self.md, self.idx, self.cols = md, idx, cols
        sd21 = r.rolling(21, min_periods=15).std()
        sd63 = r.rolling(63, min_periods=40).std()
        sd126 = r.rolling(126, min_periods=80).std()
        self.vratio = (sd21 / sd126).values                                 # H3
        gap = np.log(o / c.shift(1))
        gsd = gap.rolling(63, min_periods=40).std()
        g1 = gsd.where(U).rank(axis=1, pct=True)
        g2 = (gsd / sd63).where(U).rank(axis=1, pct=True)
        self.gap_risk = ((g1 + g2) / 2).values                             # H5
        ltv = np.log(tv)
        med = ltv.rolling(252, min_periods=150).median()
        mad = (ltv - med).abs().rolling(252, min_periods=150).median() * 1.4826
        z21 = (ltv.rolling(21, min_periods=15).mean() - med) / mad.replace(0, np.nan)
        pers = (ltv > med).astype(float).where(ltv.notna()).rolling(10, min_periods=7).mean()
        self.part = ((z21.where(U).rank(axis=1, pct=True) + pers.where(U).rank(axis=1, pct=True)) / 2).values  # H4
        self.z21, self.pers = z21.values, pers.values
        # piyasa düzeyi (kolun eşit ağırlıklı evreni)
        mret = r.where(U).mean(axis=1).fillna(0.0)
        ind = (1 + mret).cumprod()
        ma200 = c.rolling(200, min_periods=150).mean()
        self.breadth = ((c > ma200).where(U & ma200.notna()).mean(axis=1)).values          # H1
        self.bear24 = (ind / ind.shift(504) - 1 < 0).values                               # H8
        self.mvol = (mret.rolling(126, min_periods=80).std() * np.sqrt(252)).values         # H8
        self.mret21 = (ind / ind.shift(21) - 1).values
        self.bull = (ind > ind.rolling(200, min_periods=150).mean()).values
        if fx is not None and len(fx):
            f = fx.reindex(idx.union(fx.index)).ffill().reindex(idx)
            self.fx21 = (f / f.shift(21) - 1).shift(1).values        # t-1: kur kapanışı BIST kapanışından sonra
        else:
            self.fx21 = np.full(len(idx), np.nan)
        # H6 için ek özellikler + etiket yolu
        self.shock1 = (r / sd63).values
        self.ret5 = (c / c.shift(5) - 1).values
        self.ret21 = (c / c.shift(21) - 1).values
        self.hi52 = (c / c.rolling(252, min_periods=200).max()).values
        trv = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()]).groupby(level=0).max()
        self.atr = trv.reindex(idx).rolling(14, min_periods=10).mean().values
        self.o, self.h, self.l, self.c = o.values, h.values, l.values, c.values
        self.sd63 = sd63.values

    def xs_pct(self, arr, i):
        row = (arr[i] if np.ndim(arr) == 2 else arr).astype(float).copy()
        m = self.md.U[i] & np.isfinite(row)
        out = np.full(len(row), np.nan)
        if m.sum() > 5:
            out[m] = pd.Series(row[m]).rank(pct=True).values
        return out

    def h6_matrix(self, i, js):
        """H6 özellikleri (i günü kapanışında bilinen)."""
        sc = self.md.score[i]
        scp = self.xs_pct(sc, i)
        cols = [self.xs_pct(self.vratio, i)[js], self.gap_risk[i, js], self.part[i, js], self.shock1[i, js],
                self.ret5[i, js], self.ret21[i, js], self.hi52[i, js], scp[js],
                np.full(len(js), self.breadth[i]), np.full(len(js), self.mret21[i]), np.full(len(js), self.mvol[i])]
        return np.column_stack(cols)

    def adverse_first(self, i, js, horizon=10, k=1.5):
        """Yarışan risk etiketi: girişten (i+1 açılış) sonra önce -k*ATR mi, +k*ATR mi? Hiçbiri: 10. gün kapanış < giriş."""
        n = len(self.idx)
        if i + horizon >= n:
            return np.full(len(js), np.nan)
        e = self.o[i + 1, js]
        a = self.atr[i, js]
        y = np.full(len(js), np.nan)
        done = ~(np.isfinite(e) & np.isfinite(a) & (a > 0))
        for d in range(1, horizon + 1):
            lo, hi = self.l[i + d, js], self.h[i + d, js]
            adv = ~done & np.isfinite(lo) & (lo <= e - k * a)
            fav = ~done & ~adv & np.isfinite(hi) & (hi >= e + k * a)
            y[adv], y[fav] = 1.0, 0.0
            done |= adv | fav
        rest = ~done
        y[rest] = (self.c[i + horizon, js][rest] < e[rest]).astype(float)
        return y


def ranked(md, i, row=None):
    row = md.score[i] if row is None else row
    m = md.U[i] & np.isfinite(row)
    if m.sum() < 30:
        return np.array([], dtype=int)
    idx = np.where(m)[0]
    return idx[np.argsort(-row[idx], kind="stable")]


def fill_filtered(md, order, flagged, n):
    out = [j for j in order[: n + POOL_EXTRA] if not flagged[j]]
    if len(out) < n:                                       # yedekler yetmezse sıradakilerle tamamla
        extra = [j for j in order[n + POOL_EXTRA:] if not flagged[j]]
        out += extra[: n - len(out)]
    if len(out) < n:                                       # hâlâ eksikse işaretli olanlardan (işlem sayısı korunur)
        out += [j for j in order if j not in out][: n - len(out)]
    return [md.tickers[j] for j in out[:n]]


# ====================================================================== politikalar
class Policy:
    def __init__(self, name, family, kind, desc, markets=("bist", "us"), pick=None, gate=None, spec=None,
                 volscale=False, ptype=None, weights=None):
        self.name, self.family, self.kind, self.desc = name, family, kind, desc
        self.markets = set(markets)
        self.pick, self.gate, self.spec, self.volscale = pick, gate, spec, volscale
        self.ptype = ptype or kind                      # plasebo eşleşme tipi: filter / rerank / gate
        self.weights = weights                          # None: moda göre varsayılan · "KARMA": üretim risk paritesi


def make_filter_pick(feat_by_mk, flag_fn):
    def pick(md, i):
        order = ranked(md, i)
        if len(order) == 0:
            return []
        flagged = flag_fn(feat_by_mk[md.mk], i)
        return fill_filtered(md, order, flagged, C.N_PICKS)
    return pick


def make_rerank_pick(feat_by_mk, comp_fn, w=0.25):
    def pick(md, i):
        f = feat_by_mk[md.mk]
        comp = comp_fn(f, i)
        sc = md.score[i]
        row = (1 - w) * sc + w * np.where(np.isfinite(comp), comp, 0.5)
        row = np.where(np.isfinite(sc), row, np.nan)
        order = ranked(md, i, row)
        return [md.tickers[j] for j in order[: C.N_PICKS]]
    return pick


def build_policies(feat, thr, h6):
    pol = [Policy("BASE", "BASE", "base", "Üretim mantığı (değişiklik yok)")]
    pol.append(Policy("H1_breadth_gate", "H1", "gate",
                      "Evrende 200g ortalama üstündeki hisse payı, eğitim dağılımının %20'sinin altındaysa yeni dilim bütçesi ×0.5",
                      gate=lambda mk, f, i: GATE_MULT if np.isfinite(f.breadth[i]) and f.breadth[i] < thr[mk]["breadth_q20"] else 1.0))
    pol.append(Policy("H2_bist_resid_mom", "H2", "spec",
                      "BIST skoruna piyasadan arındırılmış momentum (resid_mom) 4. bileşen olarak eklenir; ABD zaten resid_mom kullanıyor (değişmez)",
                      markets=("bist",), spec={"bist": C.MARKETS["bist"]["spec"] + [["resid_mom", 1]]}, ptype="rerank"))
    pol.append(Policy("H3_vol_transition_filter", "H3", "filter",
                      "21g/126g oynaklık oranı kesitsel %80 üstündeki (geç evre, dağınık oynaklık) aday elenir, sıradaki isimle doldurulur",
                      pick=make_filter_pick(feat, lambda f, i: f.xs_pct(f.vratio, i) >= FILTER_PCT)))
    pol.append(Policy("H4_participation_rerank", "H4", "rerank",
                      "Sıralama = 0.75×üretim skoru + 0.25×katılım (ciro robust z + süreklilik) yüzdeliği",
                      pick=make_rerank_pick(feat, lambda f, i: f.part[i])))
    pol.append(Policy("H5_gap_risk_filter", "H5", "filter",
                      "Gece boşluğu riski (63g gap oynaklığı + toplam oynaklığa oranı) kesitsel %80 üstü aday elenir",
                      pick=make_filter_pick(feat, lambda f, i: np.nan_to_num(f.gap_risk[i], nan=0.0) >= FILTER_PCT)))
    if h6:
        def flag_h6(f, i):
            mk = f.md.mk
            out = np.zeros(len(f.cols), dtype=bool)
            if mk not in h6:
                return out
            order = ranked(f.md, i)[: C.N_PICKS + POOL_EXTRA]
            if len(order) == 0:
                return out
            X = f.h6_matrix(i, order)
            p = h6[mk]["predict"](X)
            out[order] = p >= h6[mk]["thr"]
            return out
        pol.append(Policy("H6_adverse_first_model", "H6", "filter",
                          "Yarışan risk modeli (önce −1.5ATR mi +1.5ATR mi, 10g): eğitimde kurulan lojistik; tahmini risk eğitim %80 üstü aday elenir",
                          markets=tuple(h6.keys()), pick=make_filter_pick(feat, flag_h6)))

    def g7(mk, f, i):
        if mk != "bist":
            return 1.0
        x = f.fx21[i]
        return GATE_MULT if np.isfinite(x) and x >= thr["bist"]["fx21_q90"] else 1.0
    pol.append(Policy("H7_fx_shock_gate", "H7", "gate",
                      "BIST: USD/TRY 21g değişimi (t−1) eğitim %90 üstündeyse (kur şoku) yeni dilim bütçesi ×0.5; ABD'de uygulanabilir çapraz bağlam yok",
                      markets=("bist",), gate=g7))

    def g8(mk, f, i):
        return GATE_MULT if bool(f.bear24[i]) and np.isfinite(f.mvol[i]) and f.mvol[i] >= thr[mk]["mvol_q70"] else 1.0
    pol.append(Policy("H8_momentum_crash_state", "H8", "gate",
                      "Daniel–Moskowitz: piyasa 24 ay negatif VE 126g piyasa oynaklığı eğitim %70 üstü -> yeni dilim ×0.5",
                      gate=g8))
    pol.append(Policy("H9_vol_managed", "H9", "volscale",
                      "Barroso–Santa-Clara: mevcut kodda bulunan ama kapalı VOL_TARGET (üretim parametreleriyle, ayar yok)",
                      volscale=True, ptype="gate"))
    return pol


def placebo_policies(feat, n, gate_share):
    out = []
    for s in range(n):
        seed = SEED + 1000 + s

        def pf(f, i, seed=seed):
            rng = np.random.default_rng(seed * 100003 + i)
            return rng.random(len(f.cols)) >= FILTER_PCT
        out.append(Policy(f"PLACEBO_filter_{s}", "PLACEBO", "filter", "rastgele %20 eleme", pick=make_filter_pick(feat, pf)))

        def pr(f, i, seed=seed):
            rng = np.random.default_rng(seed * 100019 + i)
            return rng.random(len(f.cols))
        out.append(Policy(f"PLACEBO_rerank_{s}", "PLACEBO", "rerank", "rastgele 0.25 ağırlıklı bileşen",
                          pick=make_rerank_pick(feat, pr)))

        def pg(mk, f, i, seed=seed):
            d = f.idx[i]
            rng = np.random.default_rng(seed * 7919 + d.year * 13 + d.month + (0 if mk == "bist" else 500))
            return GATE_MULT if rng.random() < gate_share.get(mk, 0.15) else 1.0
        out.append(Policy(f"PLACEBO_gate_{s}", "PLACEBO", "gate", "rastgele ay bazlı kapı (aynı aktiflik payı)", gate=pg))
    return out


# ====================================================================== simülasyon
class World:
    def __init__(self, W, fx, irx, members=None):
        self.W, self.fx, self.irx = W, fx, irx
        P.set_hist_rates(irx)
        self.md = {"bist": E.MarketData("bist", W["bist"]),
                   "us": E.MarketData("us", W["us"], member=members)}
        self.members = members
        self.base_fc = {mk: self.md[mk].fc for mk in self.md}
        self.alt_md = {}
        rb, _ = P.self_financed_returns(self.md["bist"])
        ru_usd, _ = P.self_financed_returns(self.md["us"])
        P.VOLSCALE["bist"] = P.make_vol_scale(rb)
        P.VOLSCALE["us"] = P.make_vol_scale(ru_usd)
        self.rb, self.ru = rb, P._to_tl_returns(ru_usd, "us", fx)
        self.start = max(pd.Timestamp(self.md["bist"].dates[WARMUP_IDX]), pd.Timestamp(self.md["us"].dates[WARMUP_IDX]))
        self.end = max(self.md["bist"].dates[-1], self.md["us"].dates[-1])

    def shadow(self, mk, d):
        s = self.rb if mk == "bist" else self.ru
        return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)

    def md_for(self, pol, mk):
        if pol.spec and mk in pol.spec:
            key = (pol.name, mk)
            if key not in self.alt_md:
                self.alt_md[key] = E.MarketData(mk, self.W[mk], spec=pol.spec[mk],
                                                member=self.members if mk == "us" else None)
            return self.alt_md[key]
        return self.md[mk]

    def run(self, pol: Policy, feat, end, cost_mult=1.0):
        mds = {mk: self.md_for(pol, mk) for mk in ("bist", "us")}
        gate_log = {"bist": [0, 0], "us": [0, 0]}
        orig_step = E.step_market
        old_vt = C.VOL_TARGET
        old_w = C.FIXED_WEIGHTS
        try:
            for mk, md in mds.items():
                md.fc = self.base_fc[mk] * cost_mult
                if pol.pick and mk in pol.markets:
                    md.picks = (lambda md_: (lambda i, n=None: pol.pick(md_, i)))(md)
            if pol.gate:
                def wrapped(md, i, st, ctx):
                    if md.mk in pol.markets:
                        g = pol.gate(md.mk, feat[md.mk], i)
                        gate_log[md.mk][0] += 1
                        if g != 1.0:
                            gate_log[md.mk][1] += 1
                            ctx = dict(ctx)
                            ctx["exposure"] = ctx.get("exposure", 1.0) * g
                    return orig_step(md, i, st, ctx)
                E.step_market = wrapped
            C.VOL_TARGET = bool(pol.volscale)
            if pol.weights == "KARMA":
                C.FIXED_WEIGHTS = None
            elif MODE == "us":
                C.FIXED_WEIGHTS = dict(US_ONLY)
            state = P.new_state(100_000.0, P.fx_at(self.fx, self.start))
            days = sorted(set(d for d in mds["bist"].dates + mds["us"].dates if self.start <= d <= pd.Timestamp(end)))
            for d in days:
                if d in mds["bist"].didx:
                    P.process_day(state, "bist", mds["bist"], mds["bist"].didx[d], self.fx, self.shadow)
                if d in mds["us"].didx:
                    P.process_day(state, "us", mds["us"], mds["us"].didx[d], self.fx, self.shadow)
        finally:
            E.step_market = orig_step
            C.VOL_TARGET = old_vt
            C.FIXED_WEIGHTS = old_w
            for mk, md in mds.items():
                md.fc = self.base_fc[mk]
                if "picks" in md.__dict__:
                    del md.__dict__["picks"]
        nav = pd.DataFrame(state["pf"]["nav_tl"], columns=["date", "mk", "total_tl", "fx"])
        nav["date"] = pd.to_datetime(nav["date"])
        daily_tl = nav.groupby("date")["total_tl"].last().sort_index()
        fxd = self.fx.reindex(daily_tl.index.union(self.fx.index)).ffill().reindex(daily_tl.index)
        daily_usd = daily_tl / fxd
        daily = daily_usd if MODE == "us" else daily_tl
        tr = []
        for mk in ("bist", "us"):
            for t in state["markets"][mk]["trades"]:
                tr.append(dict(t, market=mk))
        trades = pd.DataFrame(tr)
        if MODE == "us" and len(trades):
            trades = trades[trades["market"] == "us"].reset_index(drop=True)
        if len(trades):
            trades["entry_date"] = pd.to_datetime(trades["entry_date"])
            trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        share = {mk: (g[1] / g[0] if g[0] else 0.0) for mk, g in gate_log.items()}
        return {"daily": daily, "daily_tl": daily_tl, "daily_usd": daily_usd, "trades": trades, "state": state,
                "gate_share": share}


# ====================================================================== ölçümler
def window_trades(tr, a, b):
    if tr is None or tr.empty:
        return tr
    return tr[(tr["entry_date"] >= a) & (tr["entry_date"] < b - pd.Timedelta(days=EMBARGO_CAL_DAYS))]


def nav_window(daily, a, b):
    s = daily[(daily.index >= a) & (daily.index < b)]
    return s


def metrics(daily_w, tr_w):
    out = {}
    r = daily_w.pct_change().dropna()
    if len(daily_w) > 20:
        yrs = max((daily_w.index[-1] - daily_w.index[0]).days / 365.25, 1e-6)
        out["cagr_pct"] = ((daily_w.iloc[-1] / daily_w.iloc[0]) ** (1 / yrs) - 1) * 100
        out["vol_pct"] = r.std() * math.sqrt(252) * 100
        out["sharpe"] = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else np.nan
        out["mdd_pct"] = (daily_w / daily_w.cummax() - 1).min() * 100
        q = r.quantile(0.05)
        out["cvar95_daily_pct"] = r[r <= q].mean() * 100
    x = tr_w["ret_pct"].astype(float) if tr_w is not None and len(tr_w) else pd.Series(dtype=float)
    out["n_trades"] = int(len(x))
    if len(x):
        pos, neg = x[x > 0].sum(), -x[x < 0].sum()
        out.update({"expectancy_pct": x.mean(), "win_rate_pct": (x > 0).mean() * 100,
                    "profit_factor": pos / neg if neg > 0 else np.nan,
                    "avg_win_pct": x[x > 0].mean() if (x > 0).any() else np.nan,
                    "avg_loss_pct": x[x < 0].mean() if (x < 0).any() else np.nan,
                    "tail_loss_share_pct": (x <= -10).mean() * 100,
                    "stop_share_pct": (tr_w["reason"] == "felaket stopu").mean() * 100})
        for mk in ("bist", "us"):
            y = x[tr_w["market"] == mk]
            out[f"n_{mk}"] = int(len(y))
            out[f"expectancy_{mk}_pct"] = y.mean() if len(y) else np.nan
            out[f"win_rate_{mk}_pct"] = (y > 0).mean() * 100 if len(y) else np.nan
    return out


def month_arrays(tr_w, months):
    if tr_w is None or tr_w.empty:
        z = np.zeros(len(months))
        return z, z.copy()
    m = tr_w["entry_date"].dt.to_period("M")
    g = tr_w.assign(m=m).groupby("m")["ret_pct"]
    s = g.sum().reindex(months, fill_value=0.0).values.astype(float)
    n = g.count().reindex(months, fill_value=0).values.astype(float)
    return s, n


def block_indices(nm, B, rng, block=BLOCK_MONTHS):
    nb = int(math.ceil(nm / block))
    starts = rng.integers(0, max(nm - block + 1, 1), size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :nm]
    return np.minimum(idx, nm - 1)


def paired_boot(base_arr, cand_arrs, months, B=BOOT_B, seed=SEED):
    """Aynı ay blokları ortak örneklenir -> beklenti farkı dağılımı (her aday) + White RC."""
    rng = np.random.default_rng(seed)
    nm = len(months)
    if nm < 6:
        return None
    ix = block_indices(nm, B, rng)
    bs, bn = base_arr
    b_star = bs[ix].sum(1) / np.maximum(bn[ix].sum(1), 1)
    b_obs = bs.sum() / max(bn.sum(), 1)
    res = {}
    for k, (cs, cn) in cand_arrs.items():
        c_star = cs[ix].sum(1) / np.maximum(cn[ix].sum(1), 1)
        d_obs = cs.sum() / max(cn.sum(), 1) - b_obs
        d_star = c_star - b_star
        res[k] = {"diff": d_obs, "se": float(np.std(d_star, ddof=1)), "p_one": float(np.mean(d_star <= 0)),
                  "ci_lo": float(np.quantile(d_star, 0.025)), "ci_hi": float(np.quantile(d_star, 0.975)),
                  "star": d_star}
    if res:
        keys = list(res)
        V = max(res[k]["diff"] / max(res[k]["se"], 1e-9) for k in keys)
        Vs = np.max(np.column_stack([(res[k]["star"] - res[k]["diff"]) / max(res[k]["se"], 1e-9) for k in keys]), axis=1)
        rc_p = float(np.mean(Vs >= V))
        for k in keys:
            res[k]["rc_p"] = rc_p
    return res


def holm(pvals: dict):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, run = {}, 0.0
    for r, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - r) * p))
        adj[k] = run
    return adj


def monthly_returns(daily):
    return daily.resample("ME").last().pct_change().dropna()


# ====================================================================== kalite kapısı
def quality_gate(W, fx, panels, world, members_hist, removed, out_rows, synthetic):
    ok = True

    def add(check, mk, value, status, note=""):
        out_rows.append({"check": check, "market": mk, "value": value, "status": status, "note": note})

    for mk in ("bist", "us"):
        p = panels[mk]
        add("rows", mk, int(len(p)), "INFO")
        add("tickers", mk, int(p["ticker"].nunique()), "INFO")
        add("date_range", mk, f"{p['tarih'].min().date()} → {p['tarih'].max().date()}", "INFO")
        dup = int(p.duplicated(["tarih", "ticker"]).sum())
        add("duplicate_date_ticker", mk, dup, "PASS" if dup == 0 else "FAIL")
        ok &= dup == 0
        Wm = W[mk]
        bad = ((Wm["h"] < Wm["l"]) | (Wm["c"] <= 0)).sum().sum()
        tot = Wm["c"].notna().sum().sum()
        frac = float(bad / max(tot, 1))
        add("invalid_ohlc_share", mk, round(frac, 6), "PASS" if frac < 0.01 else "FAIL")
        ok &= frac < 0.01
        mono = Wm["c"].index.is_monotonic_increasing
        add("date_order", mk, bool(mono), "PASS" if mono else "FAIL")
        ok &= bool(mono)
        # kapsama / hayatta kalma yanlılığı (v1.1 madde 4: ölç ve raporla)
        c = Wm["c"]
        last = c.apply(lambda s: s.last_valid_index())
        first = c.apply(lambda s: s.first_valid_index())
        endd = c.index[-1]
        stopped = int((last < endd - pd.Timedelta(days=15)).sum())
        late = int((first > c.index[0] + pd.Timedelta(days=30)).sum())
        add("tickers_stopped_trading", mk, stopped, "DISCLOSED",
            "veri bitmeden duran (delist/askıya alınmış) hisse sayısı; sıfıra yakınsa evren büyük ölçüde güncel listedir")
        add("tickers_listed_later", mk, late, "INFO")
        by_year = c.notna().any(axis=0)
        yr = c.groupby(c.index.year).apply(lambda x: int(x.notna().any().sum()))
        add("tickers_with_data_by_year", mk, json.dumps({int(k): int(v) for k, v in yr.items()}), "INFO")
    # FX
    sim_days = [d for d in world.md["bist"].dates + world.md["us"].dates if d >= world.start]
    fxr = fx.reindex(pd.DatetimeIndex(sorted(set(sim_days))).union(fx.index)).ffill(limit=5)
    cov = float(fxr.reindex(pd.DatetimeIndex(sorted(set(sim_days)))).notna().mean())
    add("fx_coverage_sim_days", "fx", round(cov, 4), "PASS" if cov >= 0.95 else "FAIL")
    ok &= cov >= 0.95
    # maliyetler
    for mk in ("bist", "us"):
        exp = C.MARKETS[mk]["cost_rt_pct"] / 200.0
        add("cost_one_way", mk, world.md[mk].fc, "PASS" if abs(world.md[mk].fc - exp) < 1e-12 else "FAIL",
            f"config cost_rt_pct={C.MARKETS[mk]['cost_rt_pct']}")
    # üyelik
    if not synthetic:
        if members_hist is not None:
            add("sp500_point_in_time_membership", "us", "Wikipedia değişiklik tablosu", "PASS",
                f"endeksten çıkmış {len(removed)} hisse denendi; Yahoo'da verisi olmayanlar hâlâ eksik (küçük iyimserlik payı)")
        else:
            add("sp500_point_in_time_membership", "us", "YOK", "DISCLOSED",
                "yalnızca güncel üyeler: ABD sonuçlarında hayatta kalma yanlılığı var; aday ve BASE aynı evrende kıyaslanır")
    add("bist_point_in_time_membership", "bist", "YOK", "DISCLOSED",
        "BIST için ücretsiz geçmiş liste yok; işlemden kalkmış hisseler eksik. Aday-BASE farkı aynı evrende ölçülür")
    return ok


def lookahead_test(W, world, feat, rows, n_dates=3):
    """Zaman uyumu: veriyi t'de kesip özellik/skor yeniden hesaplanır; t satırı tam veriyle aynı olmalı."""
    ok = True
    rng = np.random.default_rng(SEED)
    for mk in ("bist", "us"):
        md = world.md[mk]
        cand = list(range(WARMUP_IDX + 50, len(md.dates) - 30))
        for i in sorted(rng.choice(cand, size=min(n_dates, len(cand)), replace=False)):
            d = md.dates[i]
            Wt = {k: v.loc[:d] for k, v in W[mk].items()}
            try:
                mdt = E.MarketData(mk, Wt, member=world.members if mk == "us" else None)
                it = mdt.didx.get(d)
                if it is None:
                    rows.append({"check": "lookahead_score", "market": mk, "value": str(d.date()), "status": "SKIP"})
                    continue
                ft = Feat(mdt, Wt, world.fx[world.fx.index <= d] if world.fx is not None else None)
                pairs = [("score", md.score[i], mdt.score[it]), ("vratio", feat[mk].vratio[i], ft.vratio[it]),
                         ("gap_risk", feat[mk].gap_risk[i], ft.gap_risk[it]), ("part", feat[mk].part[i], ft.part[it])]
                jt = [md.tidx[t] for t in mdt.tickers]
                for nm, a, b in pairs:
                    a = np.asarray(a)[jt]
                    b = np.asarray(b)
                    both = np.isfinite(a) & np.isfinite(b)
                    mism = int((np.isfinite(a) != np.isfinite(b)).sum())
                    dif = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
                    good = dif < 1e-8 and mism == 0
                    ok &= good
                    rows.append({"check": f"lookahead_{nm}", "market": mk, "value": f"{d.date()} maxdiff={dif:.2e} nan_mismatch={mism}",
                                 "status": "PASS" if good else "FAIL"})
                for nm, a, b in (("breadth", feat[mk].breadth[i], ft.breadth[it]), ("mvol", feat[mk].mvol[i], ft.mvol[it])):
                    good = (np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-8
                    ok &= bool(good)
                    rows.append({"check": f"lookahead_{nm}", "market": mk, "value": f"{d.date()} {a:.6f} vs {b:.6f}",
                                 "status": "PASS" if good else "FAIL"})
            except Exception as exc:
                ok = False
                rows.append({"check": "lookahead_error", "market": mk, "value": str(exc)[:200], "status": "FAIL"})
    return ok


def reconcile(res, name):
    tr = res["trades"]
    st = res["state"]
    rows = []
    for mk in ("bist", "us"):
        t = tr[tr["market"] == mk] if len(tr) else tr
        n = int(len(t))
        bad_px = int(((t["entry_px"] <= 0) | (t["exit_px"] <= 0)).sum()) if n else 0
        bad_dt = int((t["exit_date"] < t["entry_date"]).sum()) if n else 0
        bad_ret = int((~np.isfinite(t["ret_pct"].astype(float))).sum()) if n else 0
        bad_qty = int((t["qty"].astype(float) <= 0).sum()) if n else 0
        m = st["markets"][mk]
        neg_cash = float(min(m["cash"], 0.0))
        open_pos = len(m["positions"])
        pend = len(m["pending"])
        unmatched = bad_px + bad_dt + bad_ret + bad_qty
        rows.append({"policy": name, "market": mk, "closed_trades": n, "open_positions_end": open_pos,
                     "pending_orders_end": pend, "invalid_price": bad_px, "exit_before_entry": bad_dt,
                     "non_finite_return": bad_ret, "non_positive_qty": bad_qty, "negative_cash_end": neg_cash,
                     "unmatched_or_invalid": unmatched,
                     "status": "PASS" if (unmatched <= max(1, 0.005 * n) and neg_cash > -1e-6) else "FAIL"})
    return rows


# ====================================================================== H6 modeli
def train_h6(feat, world, train_end, val_rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    out, diag = {}, []
    for mk in ("bist", "us"):
        f, md = feat[mk], world.md[mk]
        i0 = md.didx.get(min((d for d in md.dates if d >= world.start), default=md.dates[-1]), WARMUP_IDX)
        Xs, ys, Xv, yv = [], [], [], []
        for i in range(i0, len(md.dates) - 12, 5):
            d = md.dates[i]
            in_train = md.dates[i + 11] < train_end
            in_val = val_rng[0] <= d < val_rng[1] - pd.Timedelta(days=EMBARGO_CAL_DAYS)
            if not (in_train or in_val):
                continue
            order = ranked(md, i)[: C.N_PICKS + POOL_EXTRA]
            if len(order) == 0:
                continue
            X = f.h6_matrix(i, order)
            y = f.adverse_first(i, order)
            ok = np.isfinite(y)
            (Xs if in_train else Xv).append(X[ok])
            (ys if in_train else yv).append(y[ok])
        if not Xs or sum(len(a) for a in ys) < 500:
            diag.append({"market": mk, "status": "NOT TESTABLE", "note": "eğitim örneklemi yetersiz"})
            continue
        X, y = np.vstack(Xs), np.concatenate(ys)
        med = np.nanmedian(X, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xf = np.where(np.isfinite(X), X, med)
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-9
        model = LogisticRegression(C=0.1, max_iter=2000).fit((Xf - mu) / sd, y)

        def predict(Z, model=model, med=med, mu=mu, sd=sd):
            Z = np.where(np.isfinite(Z), Z, med)
            return model.predict_proba(np.clip((Z - mu) / sd, -6, 6))[:, 1]
        p_tr = predict(X)
        thr = float(np.quantile(p_tr, FILTER_PCT))
        auc_tr = float(roc_auc_score(y, p_tr)) if len(set(y)) > 1 else np.nan
        auc_v = np.nan
        if Xv:
            Xvv, yvv = np.vstack(Xv), np.concatenate(yv)
            if len(set(yvv)) > 1:
                auc_v = float(roc_auc_score(yvv, predict(Xvv)))
        out[mk] = {"predict": predict, "thr": thr}
        diag.append({"market": mk, "status": "TRAINED", "n_train": int(len(y)), "adverse_first_rate": float(y.mean()),
                     "auc_train": auc_tr, "auc_validation": auc_v, "threshold_p80": thr})
    return out, diag


# ====================================================================== ana akış
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fmt(x, d=2):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=14)
    ap.add_argument("--placebo-n", type=int, default=6)
    ap.add_argument("--boot", type=int, default=BOOT_B)
    ap.add_argument("--out-dir", default="gmq_new_hypotheses_output")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mode", choices=["all", "us"], default="all",
                    help="all: karma portföy (v1.1) · us: yalnızca ABD kolu + 'karma mı %100 ABD mi' sorusu, USD bazlı (v1.2)")
    ap.add_argument("--force-winner", default=None, help="YALNIZCA --synthetic ile: kilitli test kod yolunu sınamak için")
    a = ap.parse_args()
    global MODE
    MODE = a.mode
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    np.random.seed(SEED)
    manifest = {"protocol_version": PROTOCOL_VERSION, "mode": MODE,
                "currency_for_portfolio_metrics": "USD" if MODE == "us" else "TL", "seed": SEED, "synthetic": a.synthetic,
                "started": pd.Timestamp.now().isoformat(timespec="seconds"),
                "commit": os.environ.get("GITHUB_SHA") or _git_hash(), "engine_version": C.ENGINE_VERSION,
                "params": {"TEST_MONTHS": TEST_MONTHS, "VAL_MONTHS": VAL_MONTHS, "EMBARGO_CAL_DAYS": EMBARGO_CAL_DAYS,
                           "FILTER_PCT": FILTER_PCT, "GATE_MULT": GATE_MULT, "HOLM_ALPHA": HOLM_ALPHA,
                           "PLACEBO_PCTL": PLACEBO_PCTL, "MIN_TRADE_SHARE": MIN_TRADE_SHARE, "BOOT_B": a.boot,
                           "BLOCK_MONTHS": BLOCK_MONTHS, "placebo_n_per_type": a.placebo_n,
                           "cost_rt_pct": {mk: C.MARKETS[mk]["cost_rt_pct"] for mk in C.MARKETS},
                           "N_PICKS": C.N_PICKS, "HOLD_DAYS": C.HOLD_DAYS, "N_TRANCHES": C.N_TRANCHES}}
    dq_rows, decision, report = [], {}, []
    try:
        if a.synthetic:
            W, fx, irx, panels, hist, removed = load_synthetic()
        else:
            W, fx, irx, panels, hist, removed = load_real(a.years)
        if fx is None or fx.empty:
            raise RuntimeError("USD/TRY serisi boş")
        members = None
        if hist is not None:
            members = membership_mask(hist, pd.DatetimeIndex(W["us"]["c"].index), list(W["us"]["c"].columns))
        log("⚙️ Motor kuruluyor (BASE skorları + gölge seriler)…")
        world = World(W, fx, irx, members)
        feat = {mk: Feat(world.md[mk], W[mk], fx) for mk in ("bist", "us")}

        # ---- zaman bölmesi (dondurulur)
        end = pd.Timestamp(world.end)
        span_m = (end - world.start).days / 30.44
        if span_m >= 96:
            test_start = (end - pd.DateOffset(months=TEST_MONTHS)).normalize()
            val_start = (test_start - pd.DateOffset(months=VAL_MONTHS)).normalize()
        else:
            test_start = world.start + pd.Timedelta(days=int((end - world.start).days * 0.8))
            val_start = world.start + pd.Timedelta(days=int((end - world.start).days * 0.6))
        split = {"sim_start": str(world.start.date()), "train": [str(world.start.date()), str(val_start.date())],
                 "validation": [str(val_start.date()), str(test_start.date())], "locked_test": [str(test_start.date()), str(end.date())]}
        manifest["split"] = split
        log(f"🗓️ Bölme: {split}")

        # ---- kalite kapısı
        gate_ok = quality_gate(W, fx, panels, world, hist, removed, dq_rows, a.synthetic)
        log("🔍 Zaman uyumu (look-ahead) testi…")
        gate_ok &= lookahead_test(W, world, feat, dq_rows)
        log("🔁 BASE (ön-test) iki kez: belirlenimcilik…")
        base_pre = world.run(Policy("BASE", "BASE", "base", ""), feat, test_start - pd.Timedelta(days=1))
        base_pre2 = world.run(Policy("BASE", "BASE", "base", ""), feat, test_start - pd.Timedelta(days=1))
        det = bool(np.allclose(base_pre["daily"].values, base_pre2["daily"].values, rtol=0, atol=1e-6))
        dq_rows.append({"check": "engine_determinism", "market": "all", "value": det, "status": "PASS" if det else "FAIL"})
        gate_ok &= det
        rec = reconcile(base_pre, "BASE")
        gate_ok &= all(r["status"] == "PASS" for r in rec)
        # motor paritesi: backtest_ref (yalnızca test öncesi dönemde; kilitli pencereye bakılmaz)
        ref_path = Path(C.REF_DIR) / "equity.csv"
        if ref_path.exists() and not a.synthetic:
            ref = pd.read_csv(ref_path, parse_dates=["date"]).set_index("date")[
                "Sistem (TL)" if MODE == "all" else "%100 ABD büyükler (TL)"]
            ref = ref[(ref.index >= world.start) & (ref.index < test_start)]
            ours = base_pre["daily_tl"]
            if len(ref) > 50:
                yrs = (ref.index[-1] - ref.index[0]).days / 365.25
                ref_cagr = ((ref.iloc[-1] / ref.iloc[0]) ** (1 / yrs) - 1) * 100
                o2 = ours[(ours.index >= ref.index[0]) & (ours.index <= ref.index[-1])]
                our_cagr = ((o2.iloc[-1] / o2.iloc[0]) ** (1 / yrs) - 1) * 100
                diff = our_cagr - ref_cagr
                st_ = "PASS" if abs(diff) <= 5 else "WARN"
                dq_rows.append({"check": "engine_parity_vs_backtest_ref_pretest_cagr", "market": "all",
                                "value": f"bu çalışma {our_cagr:.1f}% vs referans {ref_cagr:.1f}% (fark {diff:+.1f} puan)",
                                "status": st_, "note": "fark: veri revizyonu / evren / üyelik farkı. Kilitli pencereye bakılmadı"})
        pd.DataFrame(dq_rows).to_csv(out / "data_quality_report.csv", index=False)
        if not gate_ok:
            decision = {"decision": "INCONCLUSIVE — DATA/REPLAY QUALITY GATE FAILED", "locked_window_opened": False}
            pd.DataFrame(rec).to_csv(out / "execution_reconciliation.csv", index=False)
            write_report(out, manifest, decision, ["Kalite kapısı geçilemedi; aday sıralaması yapılmadı. Ayrıntı: data_quality_report.csv"])
            finish(out, manifest)
            return

        # ---- eşikler (yalnızca eğitim penceresinden, dondurulur)
        thr = {}
        for mk in ("bist", "us"):
            f, md = feat[mk], world.md[mk]
            msk = np.array([(world.start <= d < val_start) for d in md.dates])
            thr[mk] = {"breadth_q20": float(np.nanquantile(f.breadth[msk], 0.20)),
                       "mvol_q70": float(np.nanquantile(f.mvol[msk], 0.70)),
                       "fx21_q90": float(np.nanquantile(f.fx21[msk], 0.90)) if np.isfinite(f.fx21[msk]).any() else np.inf}
        manifest["frozen_thresholds"] = thr
        log("🧠 H6 yarışan-risk modeli (yalnızca eğitim penceresi)…")
        h6, h6_diag = train_h6(feat, world, val_start, (val_start, test_start))
        pd.DataFrame(h6_diag).to_csv(out / "h6_model_diagnostics.csv", index=False)
        policies = build_policies(feat, thr, h6)
        alloc = None
        if MODE == "us":
            kept = [policies[0]]
            policies[0].desc = "Üretim mantığı, yalnızca ABD kolu (%100 ABD, risk paritesi yok)"
            for p in policies[1:]:
                p.markets &= {"us"}
                if p.markets:
                    kept.append(p)
            policies = kept
            alloc = Policy("KARMA_BIST_US", "Q0", "allocation", "Üretimdeki karma portföy (BIST + ABD, risk paritesi)",
                           weights="KARMA")

        # ---- güç analizi (v1.1 madde 1) — test öncesi BASE verisi ile
        months_pre = pd.period_range(world.start, val_start - pd.Timedelta(days=1), freq="M")
        trw = window_trades(base_pre["trades"], world.start, val_start)
        bs, bn = month_arrays(trw, months_pre)
        rng = np.random.default_rng(SEED)
        T = TEST_MONTHS if span_m >= 96 else max(6, int(span_m * 0.2))
        ix = block_indices(len(months_pre), a.boot, rng)[:, :T] if len(months_pre) >= T else block_indices(len(months_pre), a.boot, rng)
        e_star = bs[ix].sum(1) / np.maximum(bn[ix].sum(1), 1)
        se_T = float(np.std(e_star, ddof=1))
        base_e = float(bs.sum() / max(bn.sum(), 1))
        power_rows = []
        for rho in (0.0, 0.9, 0.99):
            mde = 2.486 * se_T * math.sqrt(2 * (1 - rho))
            power_rows.append({"assumption": f"genel referans: aday-BASE aylık korelasyonu ρ={rho}", "test_months": T,
                               "se_base_expectancy_pct": se_T, "mde_abs_pct": mde,
                               "mde_rel_to_base_pct": mde / abs(base_e) * 100 if base_e else np.nan,
                               "base_expectancy_train_pct": base_e})

        # ---- tarama (yalnız test başlangıcına kadar)
        results = {"BASE": base_pre}
        for pol in policies[1:]:
            log(f"▶ {pol.name}")
            results[pol.name] = world.run(pol, feat, test_start - pd.Timedelta(days=1))
        gate_share = {mk: float(np.mean([results[n]["gate_share"][mk] for n in ("H1_breadth_gate", "H8_momentum_crash_state")
                                         if n in results])) for mk in ("bist", "us")}
        gate_share = {mk: max(v, 0.05) for mk, v in gate_share.items()}
        plac = placebo_policies(feat, a.placebo_n, gate_share)
        if MODE == "us":
            for p in plac:
                p.markets &= {"us"}
            log(f"▶ {alloc.name} (Soru 0: karma mı %100 ABD mi)")
            results[alloc.name] = world.run(alloc, feat, test_start - pd.Timedelta(days=1))
        for pol in plac:
            log(f"▶ {pol.name}")
            results[pol.name] = world.run(pol, feat, test_start - pd.Timedelta(days=1))

        # ---- ölçümler (eğitim + doğrulama)
        windows = {"train": (world.start, val_start), "validation": (val_start, test_start)}
        rows, yearly = [], []
        arrs = {w: {} for w in windows}
        mon = {w: pd.period_range(a_, b_ - pd.Timedelta(days=1), freq="M") for w, (a_, b_) in windows.items()}
        for name, res in results.items():
            for w, (a_, b_) in windows.items():
                trw = window_trades(res["trades"], a_, b_)
                m = metrics(nav_window(res["daily"], a_, b_), trw)
                pol = next((p for p in policies + plac + ([alloc] if alloc else []) if p.name == name), None)
                rows.append({"candidate": name, "family": pol.family if pol else "BASE", "kind": pol.kind if pol else "base",
                             "window": w, **m, "gate_active_share_bist": res["gate_share"]["bist"],
                             "gate_active_share_us": res["gate_share"]["us"]})
                arrs[w][name] = month_arrays(trw, mon[w])
            tr = res["trades"]
            if len(tr):
                for y_, g in tr[(tr["entry_date"] < test_start - pd.Timedelta(days=EMBARGO_CAL_DAYS))].groupby(tr["entry_date"].dt.year):
                    yearly.append({"candidate": name, "year": int(y_), "n": len(g), "expectancy_pct": g["ret_pct"].mean(),
                                   "win_rate_pct": (g["ret_pct"] > 0).mean() * 100})
        cmp_df = pd.DataFrame(rows)
        cand_names = [p.name for p in policies[1:]]
        boots = {w: paired_boot(arrs[w]["BASE"], {k: arrs[w][k] for k in cand_names + [p.name for p in plac]},
                                mon[w], B=a.boot) for w in windows}
        val_p = {k: boots["validation"][k]["p_one"] for k in cand_names}
        val_holm = holm(val_p)
        # RC yalnızca gerçek adaylar arasında
        rc = paired_boot(arrs["validation"]["BASE"], {k: arrs["validation"][k] for k in cand_names}, mon["validation"], B=a.boot)
        rc_p = next(iter(rc.values()))["rc_p"] if rc else np.nan

        # plasebo dağılımı
        pl_rows = []
        pl_dist = {}
        for p in plac:
            typ = p.kind
            dv = boots["validation"][p.name]["diff"]
            pl_dist.setdefault(typ, []).append(dv)
            pl_rows.append({"placebo": p.name, "type": typ, "val_diff_expectancy_pct": dv,
                            "train_diff_expectancy_pct": boots["train"][p.name]["diff"],
                            "val_p_one": boots["validation"][p.name]["p_one"]})
        mt_rows = []
        base_val = cmp_df[(cmp_df.candidate == "BASE") & (cmp_df.window == "validation")].iloc[0]
        base_tr = cmp_df[(cmp_df.candidate == "BASE") & (cmp_df.window == "train")].iloc[0]
        survivors = []
        for p in policies[1:]:
            k = p.name
            v = cmp_df[(cmp_df.candidate == k) & (cmp_df.window == "validation")].iloc[0]
            bv, bt = boots["validation"][k], boots["train"][k]
            dist = np.array(pl_dist.get(p.ptype, []))
            pl_q = float(np.quantile(dist, PLACEBO_PCTL)) if len(dist) else np.nan
            pl_rank = float((dist < bv["diff"]).mean()) if len(dist) else np.nan
            mdd_ok = v.get("mdd_pct", 0) >= base_val.get("mdd_pct", 0) * 1.05
            n_ok = v["n_trades"] >= MIN_TRADE_SHARE * base_val["n_trades"]
            sharpe_ok = (v.get("sharpe", np.nan) >= base_val.get("sharpe", np.nan) - 0.05)
            crit = {"holm_p<=0.20": val_holm[k] <= HOLM_ALPHA, "train_diff>0": bt["diff"] > 0,
                    "beats_placebo_p90": (not np.isfinite(pl_q)) or bv["diff"] > pl_q,
                    "mdd_not_worse_5pct": bool(mdd_ok), "trades>=70%_base": bool(n_ok), "sharpe_not_lower": bool(sharpe_ok)}
            passed = all(crit.values())
            t_stat = bv["diff"] / max(bv["se"], 1e-9)
            mde_c = 2.486 * bv["se"] * math.sqrt(len(mon["validation"]) / T)
            mt_rows.append({"candidate": k, "family": p.family, "desc": p.desc, "markets": ",".join(sorted(p.markets)),
                            "train_diff_pct": bt["diff"], "val_diff_pct": bv["diff"], "val_ci95": f"[{bv['ci_lo']:.3f}, {bv['ci_hi']:.3f}]",
                            "val_t": t_stat, "val_p_one": bv["p_one"], "val_p_holm": val_holm[k], "white_rc_p_family": rc_p,
                            "placebo_p90_threshold": pl_q, "placebo_rank": pl_rank, "mde_test_abs_pct": mde_c,
                            **{f"crit_{ck}": cv for ck, cv in crit.items()}, "selected_for_locked_test": False})
            power_rows.append({"assumption": f"{k}: gözlenen eşleşik oynaklık", "test_months": T,
                               "se_base_expectancy_pct": bv["se"], "mde_abs_pct": mde_c,
                               "mde_rel_to_base_pct": mde_c / abs(base_e) * 100 if base_e else np.nan,
                               "base_expectancy_train_pct": base_e})
            if passed:
                survivors.append((t_stat, k))
        winner = max(survivors)[1] if survivors else None
        if a.force_winner:
            if not a.synthetic:
                raise RuntimeError("--force-winner yalnızca --synthetic ile kullanılabilir (kilitli pencere kuralı)")
            winner = a.force_winner
        for r in mt_rows:
            r["selected_for_locked_test"] = r["candidate"] == winner
        pd.DataFrame(mt_rows).to_csv(out / "multiple_testing.csv", index=False)
        pd.DataFrame(pl_rows).to_csv(out / "placebo_results.csv", index=False)
        pd.DataFrame(power_rows).to_csv(out / "power_analysis.csv", index=False)
        pd.DataFrame(yearly).to_csv(out / "yearly_stability_pretest.csv", index=False)
        cmp_df.to_csv(out / "candidate_comparison.csv", index=False)
        recs = list(rec)
        for k in cand_names:
            recs += reconcile(results[k], k)
        pd.DataFrame(recs).to_csv(out / "execution_reconciliation.csv", index=False)
        manifest["selected_candidate"] = winner
        manifest["white_rc_p"] = rc_p

        # ---- Soru 0 (yalnızca ABD modu): %100 ABD mi, karma mı? (USD, portföy düzeyi)
        q0 = None
        if alloc is not None:
            q0 = {"rows": [], "send": False}
            sh = {}
            for w in windows:
                for nm in ("BASE", alloc.name):
                    r_ = cmp_df[(cmp_df.candidate == nm) & (cmp_df.window == w)].iloc[0]
                    sh[(nm, w)] = r_
                    q0["rows"].append({"window": w, "policy": "%100 ABD" if nm == "BASE" else "KARMA (BIST+ABD)",
                                       "cagr_usd_pct": r_.get("cagr_pct"), "vol_usd_pct": r_.get("vol_pct"),
                                       "sharpe_usd": r_.get("sharpe"), "mdd_usd_pct": r_.get("mdd_pct")})
                a_, b_ = windows[w]
                mb_ = monthly_returns(nav_window(results["BASE"]["daily_usd"], a_, b_))
                mk_ = monthly_returns(nav_window(results[alloc.name]["daily_usd"], a_, b_))
                d_ = (mb_ - mk_).dropna()
                q0[f"{w}_monthly_diff_mean_pct"] = float(d_.mean() * 100) if len(d_) else np.nan
            q0["send"] = all(sh[("BASE", w)].get("sharpe", -9) >= sh[(alloc.name, w)].get("sharpe", 9) for w in windows)
            manifest["q0_sent_to_locked"] = q0["send"]

        # ---- kilitli test (yalnızca seçilen aday, bir kez)
        locked_rows, stress_rows, regime_rows = [], [], []
        if q0 is not None and q0["send"]:
            log("🔓 Kilitli test (Soru 0): %100 ABD vs KARMA")
            fb = world.run(policies[0], feat, end)
            fk = world.run(alloc, feat, end)
            mb0 = metrics(nav_window(fb["daily_usd"], test_start, end + pd.Timedelta(days=1)), None)
            mk0 = metrics(nav_window(fk["daily_usd"], test_start, end + pd.Timedelta(days=1)), None)
            rb_, rk_ = monthly_returns(nav_window(fb["daily_usd"], test_start, end + pd.Timedelta(days=1))), \
                monthly_returns(nav_window(fk["daily_usd"], test_start, end + pd.Timedelta(days=1)))
            dd_ = (rb_ - rk_).dropna().values
            rng0 = np.random.default_rng(SEED)
            if len(dd_) >= 6:
                ix0 = block_indices(len(dd_), a.boot, rng0)
                bm = dd_[ix0].mean(1) * 100
                ci0 = [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))]
            else:
                ci0 = [np.nan, np.nan]
            g0 = {"sharpe_us100>=karma": mb0.get("sharpe", -9) >= mk0.get("sharpe", 9),
                  "cagr_us100>=karma": mb0.get("cagr_pct", -9) >= mk0.get("cagr_pct", 9),
                  "mdd_not_worse_5pp": mb0.get("mdd_pct", -99) >= mk0.get("mdd_pct", 0) - 5.0}
            q0["locked"] = {"us100": mb0, "karma": mk0, "monthly_diff_mean_pct": float(dd_.mean() * 100) if len(dd_) else np.nan,
                            "monthly_diff_ci95": ci0, "gates": g0,
                            "verdict": ("%100 ABD tercih edilebilir (gölge çalışma ile)" if all(g0.values())
                                        else "Kanıt yetersiz: karma portföy korunur")}
            for nm, m_ in (("%100 ABD", mb0), ("KARMA (BIST+ABD)", mk0)):
                q0["rows"].append({"window": "locked_test", "policy": nm, "cagr_usd_pct": m_.get("cagr_pct"),
                                   "vol_usd_pct": m_.get("vol_pct"), "sharpe_usd": m_.get("sharpe"), "mdd_usd_pct": m_.get("mdd_pct")})
        elif q0 is not None:
            q0["locked"] = {"verdict": "Doğrulama ölçütü geçilmedi (eğitim VE doğrulamada %100 ABD Sharpe ≥ karma gerekir): karma portföy korunur; kilitli pencere bu soru için açılmadı"}
        if q0 is not None:
            pd.DataFrame(q0["rows"]).to_csv(out / "q0_us100_vs_karma.csv", index=False)
            manifest["q0"] = {k: v for k, v in q0.items() if k != "rows"}
        if winner is None:
            decision = {"decision": "FAIL — doğrulamada seçim ölçütlerini geçen aday yok (kilitli pencere AÇILMADI, gelecekteki test için korunuyor)",
                        "locked_window_opened": False}
        else:
            log(f"🔓 Kilitli test açılıyor: BASE vs {winner}")
            wpol = next(p for p in policies if p.name == winner)
            bpol = policies[0]
            full = {}
            for mult in (1.0, 1.5, 2.0):
                full[("BASE", mult)] = world.run(bpol, feat, end, cost_mult=mult)
                full[(winner, mult)] = world.run(wpol, feat, end, cost_mult=mult)
                for nm in ("BASE", winner):
                    m = metrics(nav_window(full[(nm, mult)]["daily"], test_start, end + pd.Timedelta(days=1)),
                                window_trades(full[(nm, mult)]["trades"], test_start, end + pd.Timedelta(days=EMBARGO_CAL_DAYS + 1)))
                    stress_rows.append({"cost_multiplier": mult, "policy": nm, **m})
            tb = full[("BASE", 1.0)]
            tw = full[(winner, 1.0)]
            te_end = end + pd.Timedelta(days=EMBARGO_CAL_DAYS + 1)
            tr_b = window_trades(tb["trades"], test_start, te_end)
            tr_w = window_trades(tw["trades"], test_start, te_end)
            mb = metrics(nav_window(tb["daily"], test_start, end + pd.Timedelta(days=1)), tr_b)
            mw = metrics(nav_window(tw["daily"], test_start, end + pd.Timedelta(days=1)), tr_w)
            mon_t = pd.period_range(test_start, end, freq="M")
            bt_ = paired_boot(month_arrays(tr_b, mon_t), {"W": month_arrays(tr_w, mon_t)}, mon_t, B=a.boot)["W"]
            mrb, mrw = monthly_returns(nav_window(tb["daily"], test_start, end + pd.Timedelta(days=1))), \
                monthly_returns(nav_window(tw["daily"], test_start, end + pd.Timedelta(days=1)))
            locked_rows = [{"policy": "BASE", **mb}, {"policy": winner, **mw},
                           {"policy": "FARK (aday − BASE)", "expectancy_pct": bt_["diff"],
                            "win_rate_pct": mw.get("win_rate_pct", np.nan) - mb.get("win_rate_pct", np.nan),
                            "cagr_pct": mw.get("cagr_pct", np.nan) - mb.get("cagr_pct", np.nan),
                            "sharpe": mw.get("sharpe", np.nan) - mb.get("sharpe", np.nan),
                            "mdd_pct": mw.get("mdd_pct", np.nan) - mb.get("mdd_pct", np.nan),
                            "n_trades": mw["n_trades"] - mb["n_trades"]}]
            # rejim kırılımı (eşikler eğitimden)
            for mk in ("bist", "us"):
                f = feat[mk]
                medv = float(np.nanmedian(f.mvol[[world.start <= d < val_start for d in world.md[mk].dates]]))
                for nm, trd in (("BASE", tr_b), (winner, tr_w)):
                    t = trd[trd["market"] == mk]
                    if t.empty:
                        continue
                    ii = [world.md[mk].didx.get(d) for d in t["entry_date"]]
                    ii = [max(0, (x or 1) - 1) for x in ii]
                    hv = np.array([f.mvol[x] >= medv if np.isfinite(f.mvol[x]) else False for x in ii])
                    bl = np.array([bool(f.bull[x]) for x in ii])
                    for lab, msk in (("yüksek oynaklık", hv), ("düşük oynaklık", ~hv), ("boğa (200g üstü)", bl), ("ayı (200g altı)", ~bl)):
                        x = t["ret_pct"].values[msk]
                        regime_rows.append({"market": mk, "regime": lab, "policy": nm, "n": int(len(x)),
                                            "expectancy_pct": float(np.mean(x)) if len(x) else np.nan,
                                            "win_rate_pct": float(np.mean(x > 0) * 100) if len(x) else np.nan,
                                            "strong_claim_allowed": len(x) >= SUBGROUP_MIN})
            # PASS kapısı
            s15 = {r["policy"]: r for r in stress_rows if r["cost_multiplier"] == 1.5}
            g = {}
            g["1_ci_above_zero"] = bt_["diff"] > 0 and bt_["ci_lo"] > 0
            be = mb.get("expectancy_pct", np.nan)
            g["2_economic_10pct"] = (bt_["diff"] >= 0.10 * be) if be > 0 else bt_["diff"] > 0
            risk_adj = (mw.get("sharpe", -9) - mb.get("sharpe", 9) >= 0.10) and (mw.get("cagr_pct", -9) >= mb.get("cagr_pct", 9))
            g["3_winrate_2pp_or_riskadj"] = (mw.get("win_rate_pct", 0) - mb.get("win_rate_pct", 0) >= 2.0) or bool(risk_adj)
            g["4_mdd_cvar_pf"] = (mw.get("mdd_pct", -99) >= mb.get("mdd_pct", 0) * 1.05) and \
                                 (mw.get("cvar95_daily_pct", -99) >= mb.get("cvar95_daily_pct", 0) * 1.10) and \
                                 (mw.get("profit_factor", 0) >= 0.95 * mb.get("profit_factor", 0))
            g["5_cost_1.5x_direction"] = s15[winner].get("expectancy_pct", -9) - s15["BASE"].get("expectancy_pct", 9) > 0
            n_ok = mw["n_trades"] >= MIN_TEST_TRADES and all(mw.get(f"n_{mk}", 0) >= MIN_TEST_TRADES_MKT for mk in wpol.markets)
            g["6_sample_size"] = bool(n_ok)
            scope = "her iki piyasa" if wpol.markets == {"bist", "us"} else f"yalnızca {','.join(sorted(wpol.markets)).upper()}"
            if not n_ok:
                dec = "INCONCLUSIVE — kilitli dönem örneklemi yetersiz"
            elif all(g.values()):
                dec = f"PASS — {winner} ({scope}); üretime doğrudan değil, önce gölge çalışma"
            else:
                dec = f"FAIL — {winner} kilitli testte kapıları geçemedi: " + ", ".join(k for k, v in g.items() if not v)
            decision = {"decision": dec, "locked_window_opened": True, "gates": g, "scope": scope,
                        "locked_diff_ci95": [bt_["ci_lo"], bt_["ci_hi"]], "locked_diff": bt_["diff"],
                        "monthly_portfolio_diff_mean_pct": float((mrw - mrb).mean() * 100) if len(mrb) else np.nan}
        pd.DataFrame(locked_rows).to_csv(out / "locked_test_base_vs_winner.csv", index=False)
        pd.DataFrame(stress_rows).to_csv(out / "cost_stress.csv", index=False)
        pd.DataFrame(regime_rows).to_csv(out / "market_regime_breakdown.csv", index=False)

        # ---- rapor
        rep = []
        rep += ["## İlk sayfa: üç soru", ""]
        if decision.get("locked_window_opened"):
            lr = {r["policy"]: r for r in locked_rows}
            b, w_ = lr["BASE"], lr[winner]
            rep += [f"1. **BASE'e karşı net iyileşme var mı?** İşlem başı net fark {fmt(decision['locked_diff'], 3)} puan, "
                    f"%95 GA [{fmt(decision['locked_diff_ci95'][0], 3)}, {fmt(decision['locked_diff_ci95'][1], 3)}].",
                    f"2. **Win rate ve düşüş:** WR {fmt(b.get('win_rate_pct'), 1)}% → {fmt(w_.get('win_rate_pct'), 1)}%; "
                    f"en büyük düşüş {fmt(b.get('mdd_pct'), 1)}% → {fmt(w_.get('mdd_pct'), 1)}%.",
                    f"3. **Üretime aday mı?** {decision['decision']}", ""]
        else:
            rep += ["1. **BASE'e karşı net iyileşme var mı?** Doğrulamada istatistiksel ölçütleri geçen aday çıkmadı.",
                    "2. **Win rate ve düşüş:** Aday bazında eğitim/doğrulama değerleri candidate_comparison.csv'de.",
                    f"3. **Üretime aday mı?** Hayır. {decision['decision']}", ""]
        if q0 is not None:
            rep += ["## Soru 0: %100 ABD mi, karma (BIST+ABD) mi? — dolar bazında", "",
                    "| Dönem | Portföy | Yıllık getiri (USD) | Oynaklık | Sharpe | En büyük düşüş |", "|---|---|---|---|---|---|"]
            for r in q0["rows"]:
                rep.append(f"| {r['window']} | {r['policy']} | {fmt(r['cagr_usd_pct'], 1)}% | {fmt(r['vol_usd_pct'], 1)}% | "
                           f"{fmt(r['sharpe_usd'], 2)} | {fmt(r['mdd_usd_pct'], 1)}% |")
            rep += ["", f"Aylık getiri farkı (%100 ABD − karma): eğitim {fmt(q0.get('train_monthly_diff_mean_pct'), 2)} puan/ay · "
                        f"doğrulama {fmt(q0.get('validation_monthly_diff_mean_pct'), 2)} puan/ay"]
            lk = q0.get("locked", {})
            if lk.get("monthly_diff_ci95"):
                rep.append(f"Kilitli test aylık fark: {fmt(lk.get('monthly_diff_mean_pct'), 2)} puan/ay, %95 GA "
                           f"[{fmt(lk['monthly_diff_ci95'][0], 2)}, {fmt(lk['monthly_diff_ci95'][1], 2)}] · kapılar: "
                           + ", ".join(f"{k} {'✅' if v else '❌'}" for k, v in lk.get("gates", {}).items()))
            rep += [f"**Soru 0 sonucu: {lk.get('verdict')}**", ""]
        rep += ["## Doğrulama — aday tablosu (kilitli pencere hariç)", "",
                "| Aday | Piyasa | Eğitim fark | Doğrulama fark | t | Holm p | Plasebo sırası | Ölçülebilir en küçük etki (test) | Seçildi |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in mt_rows:
            rep.append(f"| {r['candidate']} | {r['markets']} | {fmt(r['train_diff_pct'], 3)} | {fmt(r['val_diff_pct'], 3)} | "
                       f"{fmt(r['val_t'], 2)} | {fmt(r['val_p_holm'], 3)} | {fmt(r['placebo_rank'], 2)} | "
                       f"{fmt(r['mde_test_abs_pct'], 3)} | {'✅' if r['selected_for_locked_test'] else ''} |")
        rep += ["", f"White Reality Check (en iyi adayın şans eseri olma olasılığı, aile bazında): p = {fmt(rc_p, 3)}", "",
                f"BASE eğitim dönemi işlem başı net: {fmt(base_tr.get('expectancy_pct'), 3)}% · doğrulama: {fmt(base_val.get('expectancy_pct'), 3)}%",
                "", "## Güç analizi (v1.1 madde 1)", "",
                "Kilitli pencerede güven aralığıyla ayırt edilebilecek en küçük işlem başı fark (tek yönlü %5, güç %80):", ""]
        rep.append("Asıl ölçü adayların gözlenen eşleşik oynaklığıdır (yukarıdaki tabloda 'Ölçülebilir en küçük etki' sütunu): "
                   "aday ile BASE çoğu hisseyi ortak tuttuğu için fark, iki bağımsız sistemden çok daha kesin ölçülür. "
                   "Bir adayın gerçek etkisi bu değerin altındaysa kilitli test onu güvenle ayırt edemez (sonuç FAIL/INCONCLUSIVE olur).")
        for r in power_rows[:3]:
            rep.append(f"- {r['assumption']}: ±{fmt(r['mde_abs_pct'], 3)} puan")
        rep += ["", "## Plasebo (v1.1 madde 3)", "",
                f"Her tip için {a.placebo_n} rastgele politika aynı motorla çalıştırıldı. Bir adayın seçilebilmesi için doğrulama farkı kendi tipindeki plasebo dağılımının %90'ını aşmalı.",
                ""] + [f"- {t}: plasebo doğrulama farkları {', '.join(fmt(x, 3) for x in sorted(v))}" for t, v in pl_dist.items()]
        rep += ["", "## H6 modeli", ""] + [f"- {json.dumps(d, ensure_ascii=False, default=float)}" for d in h6_diag]
        rep += ["", "## Veri ve yanlılık (v1.1 madde 4)", "",
                "Ayrıntı data_quality_report.csv. Evren yanlılığı aday ve BASE'i aynı biçimde etkiler; fark testi bu nedenle "
                "seviye sonuçlarından daha güvenilirdir, ama sıfır değildir."]
        write_report(out, manifest, decision, rep)
    except Exception as exc:
        traceback.print_exc()
        decision = {"decision": f"INCONCLUSIVE — çalışma hatası: {exc}", "locked_window_opened": False}
        if dq_rows:
            pd.DataFrame(dq_rows).to_csv(out / "data_quality_report.csv", index=False)
        write_report(out, manifest, decision, ["```", traceback.format_exc()[-3000:], "```"])
    finally:
        manifest["decision"] = decision
        manifest["runtime_sec"] = round(time.time() - t0, 1)
        finish(out, manifest)
    log(f"✅ Bitti: {decision.get('decision')}")


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def write_report(out, manifest, decision, body):
    title = "ABD Kolu (USD bazlı)" if MODE == "us" else "Karma Portföy"
    lines = [f"# GMQ Yeni Hipotezler — {title} — Nihai Karar Raporu (protokol v{PROTOCOL_VERSION})", "",
             f"**KARAR: {decision.get('decision')}**", "",
             f"Kilitli pencere açıldı mı: {'evet' if decision.get('locked_window_opened') else 'hayır'} · "
             f"bölme: {json.dumps(manifest.get('split', {}), ensure_ascii=False)}", ""]
    if decision.get("gates"):
        lines += ["### PASS kapıları", ""] + [f"- {k}: {'✅' if v else '❌'}" for k, v in decision["gates"].items()] + [""]
    lines += body
    (out / "final_decision_report.md").write_text("\n".join(lines), encoding="utf-8")
    notes = ["# Tekrarlanabilirlik notları", "",
             "- Komut: `python gmq_new_hypotheses_test.py --years 14 --placebo-n 6 --out-dir gmq_new_hypotheses_output`",
             "- Bağımlılıklar: requirements.txt (pandas 2.3.3, numpy 2.2.6, scikit-learn 1.7.2, yfinance, lxml, requests)",
             f"- Rastgele tohum: {SEED}; bootstrap: {BLOCK_MONTHS} aylık hareketli blok, ortak (eşleşik) örnekleme",
             "- Motor: üretim engine.py/portfolio.py değiştirilmeden, bellekte sarmalayarak (picks / exposure) çalıştırıldı",
             "- Risk paritesi gölge serileri tüm politikalar için ortak (BASE) tutuldu: fark yalnızca adayın kendi etkisini ölçer",
             "- Bilinen sınırlamalar: Yahoo verisi sonradan revize olabilir; BIST'te işlemden kalkmış hisseler yok; "
             "ABD'de endeksten çıkmış ve Yahoo'da verisi olmayan hisseler yok; üretim stratejisi 2014-26'nın tamamı "
             "görülerek tasarlandığı için BASE'in kendisi için kilitli pencere tamamen 'görülmemiş' değildir (adaylar için öyledir)."]
    (out / "reproducibility_notes.md").write_text("\n".join(notes), encoding="utf-8")


def finish(out, manifest):
    me = Path(__file__).resolve()
    files = {}
    for p in sorted(out.glob("*")):
        if p.name != "manifest.json" and p.is_file():
            files[p.name] = sha256(p)
    files["_script"] = sha256(me)
    proto = me.parent / "PROTOKOL_YENI_HIPOTEZLER.md"
    if proto.exists():
        files["_protocol"] = sha256(proto)
    manifest["sha256"] = files
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
