"""Varyant testi — mevcut sistemi BOZMADAN, aynı motorla karşılaştırmalı geçmiş test.

Bu dosyayı proje köküne (config.py'nin yanına) koy ve çalıştır:

    python varyant_test.py                 # 13 yıl, tüm varyantlar (ilk seferde veri iner, uzun sürer)
    python varyant_test.py --years 8       # daha kısa pencere
    python varyant_test.py --variants temel A_uzun_tutma
    python varyant_test.py --synthetic     # veri indirmeden uçtan uca sınama (yapay veri)

Nasıl çalışır:
  * Canlı sistemin kullandığı AYNI fonksiyonlarla gün gün ilerler:
    data.get_panel -> engine.MarketData -> portfolio.process_day.
  * Varyantlar yalnızca config değerlerini (bellekte) değiştirir; dosyalara dokunmaz,
    canlı durum dosyalarına (data/state.json vb.) yazmaz.
  * Her varyant temiz bir portföyle baştan koşulur; sonuçlar tek tabloda karşılaştırılır.

Bilinen fark (orijinal 13 yıllık araştırmaya göre): geçmiş S&P 500 üyelik listesi yok;
ABD kolu bugünkü üyelerle çalışır. Bu küçük bir iyimserlik ekler ama TÜM varyantlara
aynen uygulandığı için varyantlar arası karşılaştırmayı bozmaz.
"""
from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import config as C
import data as DA
import engine as E
import portfolio as P


# ------------------------------------------------------------------ varyant tanımları
def _bist_spec_amihud():
    return [["mom_12_1", 1], ["hi52", 1], ["amihud_illiq", 1]]

def _us_spec_lowivol():
    return [["resid_mom", 1], ["low_ivol", 1]]

VARIANTS = {
    "temel":           {"desc": "Mevcut sistem (v1.4, hiçbir değişiklik)", "cfg": {}, "spec": {}},
    "A_uzun_tutma":    {"desc": "Tutma 21->42 gün, 8 dilim (ciro ve maliyet düşer)",
                        "cfg": {"HOLD_DAYS": 42, "N_TRANCHES": 8}, "spec": {}},
    "B_bist_amihud":   {"desc": "BIST skoru: mom_12_1 + hi52 + amihud_illiq (illikidite primi)",
                        "cfg": {}, "spec": {"bist": _bist_spec_amihud()}},
    "D_us_lowivol":    {"desc": "ABD skoru: resid_mom + low_ivol (düşük içsel oynaklık)",
                        "cfg": {}, "spec": {"us": _us_spec_lowivol()}},
    "E_maliyet_dusuk": {"desc": "Maliyet indirimi: BIST %0.35->%0.20, ABD %0.10->%0.07 (komisyon pazarlığı)",
                        "cfg": {"COST": {"bist": 0.20, "us": 0.07}}, "spec": {}},
}


# ------------------------------------------------------------------ config geçici değişikliği
class Ctx:
    """config.py değerlerini bellekte geçici değiştirir; çıkışta geri alır. Dosyaya dokunmaz."""
    def __init__(self, variant):
        self.v = variant
        self.saved = {}

    def __enter__(self):
        for k, val in self.v["cfg"].items():
            if k == "COST":
                self.saved["COST"] = {m: C.MARKETS[m]["cost_rt_pct"] for m in val}
                for m, c in val.items():
                    C.MARKETS[m]["cost_rt_pct"] = c
            else:
                self.saved[k] = getattr(C, k)
                setattr(C, k, val)
        P.HIST_RATES["bist"] = P.HIST_RATES["us"] = None   # her varyant temiz başlar
        P.VOLSCALE["bist"] = P.VOLSCALE["us"] = None
        return self

    def __exit__(self, *a):
        for k, val in self.saved.items():
            if k == "COST":
                for m, c in val.items():
                    C.MARKETS[m]["cost_rt_pct"] = c
            else:
                setattr(C, k, val)


# ------------------------------------------------------------------ gölge seri için geçmişe bakan dilim
class SliceMD:
    """MarketData'nın yalnızca ilk k gününü gösteren görünüm (risk paritesi geleceğe bakmasın diye)."""
    def __init__(self, md, k):
        for a in ("o", "c", "c_ff", "o_adj", "c_adj", "score", "U", "event"):
            v = getattr(md, a)
            setattr(self, a, None if v is None else v[:k])
        self.dates = md.dates[:k]
        self.tickers, self.tidx, self.mk = md.tickers, md.tidx, md.mk
        self.fc, self.sectors, self.cfg, self.spec = md.fc, md.sectors, md.cfg, md.spec
        self.didx = md.didx

    def picks(self, i, n=None):
        n = n or C.N_PICKS
        row = self.score[i]
        m = self.U[i] & np.isfinite(row)
        if m.sum() < 30:
            return []
        idx = np.where(m)[0]
        order = idx[np.argsort(-row[idx], kind="stable")]
        return [self.tickers[j] for j in order[:n]]

    def universe(self, i):
        return [self.tickers[j] for j in np.where(self.U[i])[0]]

    def open_px(self, i, t):
        j = self.tidx.get(t)
        return float(self.o[i, j]) if j is not None else float("nan")

    def close_px(self, i, t):
        j = self.tidx.get(t)
        return float(self.c_ff[i, j]) if j is not None else float("nan")


# ------------------------------------------------------------------ tek varyant koşusu
def run_variant(name, variant, panels, fx, us_tbill, start=None, quiet=False):
    with Ctx(variant):
        P.set_hist_rates(us_tbill)
        mds, specs = {}, {}
        for mk in ("bist", "us"):
            spec = variant["spec"].get(mk) or C.MARKETS[mk]["spec"]
            mds[mk] = E.MarketData(mk, panels[mk], spec=spec)
        fx0 = P.fx_at(fx, mds["bist"].dates[0])
        state = P.new_state(C.CAPITAL_TL, fx0)

        # başlangıç günü: sinyallerin ısınması için her iki kolun da 260+ günü dolunca
        i0 = {}
        for mk in ("bist", "us"):
            dates = mds[mk].dates
            lo = 260
            if start:
                lo = max(lo, next((i for i, d in enumerate(dates) if d >= pd.Timestamp(start)), len(dates)))
            i0[mk] = lo if lo < len(dates) else len(dates) - 1

        shadow_cache = {}

        def shadow_fn(m, d):
            key = (m, f"{d.year}-{d.month:02d}")
            if key in shadow_cache:
                return shadow_cache[key]
            md = mds[m]
            k = sum(1 for x in md.dates if x < pd.Timestamp(d))
            out = None
            if k > 300:
                r = E.shadow_returns(SliceMD(md, k), last_n=260)
                r = P._to_tl_returns(r, m, fx)
                out = r[r.index < pd.Timestamp(d)]
            shadow_cache[key] = out
            return out

        # iki kolu tek takvimde gün gün ilerlet
        ptr = dict(i0)
        all_days = sorted(set(mds["bist"].dates[i0["bist"]:]) | set(mds["us"].dates[i0["us"]:]))
        for d in all_days:
            for mk in ("bist", "us"):
                dates = mds[mk].dates
                if ptr[mk] < len(dates) and dates[ptr[mk]] == d:
                    P.process_day(state, mk, mds[mk], ptr[mk], fx, shadow_fn)
                    ptr[mk] += 1
        if not quiet:
            print(f"  ✓ {name}: {len(all_days)} takvim günü işlendi")
        return state


# ------------------------------------------------------------------ istatistikler
def nav_series(state):
    rows = state["pf"]["nav_tl"]
    df = pd.DataFrame(rows, columns=["date", "market", "total_tl", "usdtry"])
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date")["total_tl"].last().sort_index()


def stats_of(state):
    nav = nav_series(state)
    if len(nav) < 30:
        return {}
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    m = nav.resample("ME").last().pct_change().dropna()
    y12 = nav / nav.shift(252) - 1
    out = {
        "Yıllık getiri %": round(ann * 100, 1),
        "En büyük düşüş %": round(dd * 100, 1),
        "Kârlı ay %": round((m > 0).mean() * 100, 1),
        "En kötü ay %": round(m.min() * 100, 1),
        "12 ay pozitif %": round((y12.dropna() > 0).mean() * 100, 1),
        "En kötü 12 ay %": round(y12.min() * 100, 1),
        "1 TL ne oldu": round(nav.iloc[-1] / nav.iloc[0], 2),
    }
    for mk in ("bist", "us"):
        tr = pd.DataFrame(state["markets"][mk]["trades"])
        if len(tr):
            wins = tr[tr["ret_pct"] > 0]["ret_pct"].sum()
            losses = -tr[tr["ret_pct"] <= 0]["ret_pct"].sum()
            out[f"{mk}: işlem"] = len(tr)
            out[f"{mk}: kârla kapanan %"] = round((tr["ret_pct"] > 0).mean() * 100, 1)
            out[f"{mk}: işlem başı net %"] = round(tr["ret_pct"].mean(), 2)
            out[f"{mk}: kâr faktörü"] = round(wins / losses, 2) if losses > 0 else np.nan
    return out


# ------------------------------------------------------------------ sentetik veri (hızlı sınama)
def synthetic_panel(mk, n_tickers=90, years=7, seed=42):
    rng = np.random.default_rng(seed + (0 if mk == "bist" else 1))
    days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(years * 252))
    drift = rng.normal(0.0004, 0.0003, n_tickers)          # kalıcı farklar -> momentumun tutacağı yapı
    vol = np.abs(rng.normal(0.02 if mk == "bist" else 0.013, 0.004, n_tickers))
    px = np.ones_like(drift) * (50 if mk == "bist" else 100)
    rows = []
    tickers = [f"T{j:03d}" for j in range(n_tickers)]
    for d in days:
        shock = rng.normal(0, 1, n_tickers)
        ret = drift + vol * shock
        px = px * (1 + ret)
        o = px / (1 + ret * rng.uniform(0.2, 0.5, n_tickers))
        h = np.maximum(o, px) * (1 + np.abs(rng.normal(0, 0.004, n_tickers)))
        l = np.minimum(o, px) * (1 - np.abs(rng.normal(0, 0.004, n_tickers)))
        v = rng.lognormal(15 if mk == "bist" else 16, 0.7, n_tickers)
        for j, t in enumerate(tickers):
            rows.append((d, t, o[j], h[j], l[j], px[j], v[j]))
    return pd.DataFrame(rows, columns=["tarih", "ticker", "open", "high", "low", "close", "volume"])


# ------------------------------------------------------------------ ana akış
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=13)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--start", default=None, help="ör. 2014-01-01 (varsayılan: sinyal ısınmasından hemen sonra)")
    ap.add_argument("--out", default="varyant_raporu.md")
    ap.add_argument("--synthetic", action="store_true", help="veri indirme; yapay veriyle uçtan uca sınama")
    args = ap.parse_args()

    if args.synthetic:
        print("Yapay veri üretiliyor (gerçek sonuç DEĞİL, sadece sınama)...")
        panels = {mk: synthetic_panel(mk) for mk in ("bist", "us")}
        days = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=7 * 252)
        fx = pd.Series(np.exp(np.cumsum(np.random.default_rng(7).normal(0.0006, 0.004, len(days)))), index=days)
        us_tbill = pd.Series(3.0, index=days)
    else:
        panels, fx, us_tbill = {}, None, None
        for mk in ("bist", "us"):
            panels[mk] = DA.get_panel(mk, years=args.years, force_full=True)
            if panels[mk].empty:
                sys.exit(f"{mk} verisi alınamadı.")
        fx = DA.get_fx(args.years)
        try:
            irx = DA._yf().download("^IRX", period=f"{args.years}y", interval="1d",
                                    auto_adjust=False, progress=False)["Close"]
            if isinstance(irx, pd.DataFrame):
                irx = irx.iloc[:, 0]
            irx.index = pd.to_datetime(irx.index).tz_localize(None).normalize()
            us_tbill = irx.dropna()
        except Exception as exc:
            print("^IRX alınamadı, nakit için %2 varsayılacak:", exc)

    panels = {mk: DA.to_wide(p) for mk, p in panels.items()}
    rows, descs = [], {}
    for name in args.variants:
        if name not in VARIANTS:
            print(f"bilinmeyen varyant: {name} (seçenekler: {list(VARIANTS)})")
            continue
        v = VARIANTS[name]
        print(f"[{datetime.now():%H:%M:%S}] koşuluyor: {name} — {v['desc']}")
        st = run_variant(name, v, panels, fx, us_tbill, start=args.start)
        s = stats_of(st)
        s["Varyant"] = name
        rows.append(s)
        descs[name] = v["desc"]

    tab = pd.DataFrame(rows).set_index("Varyant").T
    md = ["# Varyant Test Raporu", f"_Üretim: {datetime.now():%Y-%m-%d %H:%M} — "
          f"{'YAPAY VERİ (sınama)' if args.synthetic else f'{args.years} yıl'}_", "",
          " | ".join(["Metrik"] + list(tab.columns)), "|" + "---|" * (len(tab.columns) + 1)]
    for met, row in tab.iterrows():
        md.append(" | ".join([met] + [str(x) for x in row]))
    md += ["", "## Varyant açıklamaları", ""]
    md += [f"- **{n}**: {descs[n]}" for n in descs]
    md += ["", "Not: ABD kolu bugünkü S&P 500 üyeleriyle koşuldu (geçmiş üyelik yok) — tüm varyantlar için aynı."]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n" + tab.to_string())
    print(f"\nRapor yazıldı: {args.out}")


if __name__ == "__main__":
    main()
