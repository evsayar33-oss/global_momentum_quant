
#!/usr/bin/env python3
"""GMQ Adaptive Risk V2 research. Never modifies live state/config files."""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data as DA
import engine as E
import portfolio as P

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "gmq_adaptive_risk_v2_raporu.md"

YEARS = 14
COSTS = [0.35, 0.50, 0.75, 1.00, 1.25]
REF_COST = 0.35
BOOT = 5000
SEED = 20261005
WARMUP = 260

# Pre-registered V2 parameters; do not optimize after seeing results.
COV_WINDOW = 63
COV_BLEND = 0.50
DD_WINDOW = 63
DD_FLOOR = 0.60
DD_STRENGTH = 0.40
TREND_FAST = 21
TREND_SLOW = 63
TREND_STRENGTH = 0.20
MONTHLY_SHIFT = 0.10
HYSTERESIS = 0.025


@dataclass
class Scenario:
    name: str
    daily: pd.Series
    trades: pd.DataFrame


def fmt(x):
    try:
        x = float(x)
        return "—" if not np.isfinite(x) else f"{x:.2f}"
    except Exception:
        return "—"


def clone_cost(md, cost):
    x = copy.copy(md)
    x.fc = float(cost) / 200.0
    return x


def invvol_weight(b, u):
    df = pd.concat([b.rename("b"), u.rename("u")], axis=1).dropna().tail(C.RP_WINDOW)
    if len(df) < 40:
        return float(C.RP_DEFAULT["bist"])
    v = df.std()
    if (v <= 0).any() or not np.isfinite(v).all():
        return float(C.RP_DEFAULT["bist"])
    z = 1.0 / v
    return float(np.clip(z["b"] / z.sum(), *C.RP_BOUNDS))


def minvar_weight(b, u):
    df = pd.concat([b.rename("b"), u.rename("u")], axis=1).dropna().tail(COV_WINDOW)
    if len(df) < 40:
        return invvol_weight(b, u)
    cov = df.cov()
    vb, vu, c = float(cov.loc["b", "b"]), float(cov.loc["u", "u"]), float(cov.loc["b", "u"])
    den = vb + vu - 2 * c
    if den <= 0 or not np.isfinite(den):
        return invvol_weight(b, u)
    return float(np.clip((vu - c) / den, *C.RP_BOUNDS))


def dd_last(r):
    r = r.dropna().tail(DD_WINDOW)
    if len(r) < 20:
        return 0.0
    g = (1 + r).cumprod()
    return float((g / g.cummax() - 1).iloc[-1])


def trend_score(r):
    r = r.dropna()
    if len(r) < TREND_SLOW:
        return 0.0
    f = (1 + r.tail(TREND_FAST)).prod() - 1
    s = (1 + r.tail(TREND_SLOW)).prod() - 1
    return float(np.tanh((0.6 * f + 0.4 * s) / 0.10))


def make_risk_weight(model):
    state = {"prev": None, "month": None}

    def weight(b, u):
        if len(b) == 0 or len(u) == 0:
            return dict(C.RP_DEFAULT)

        month_dates = []
        if len(b):
            month_dates.append(pd.Timestamp(b.index[-1]))
        if len(u):
            month_dates.append(pd.Timestamp(u.index[-1]))
        month = max(month_dates).strftime("%Y-%m") if month_dates else None

        if state["month"] == month and state["prev"] is not None:
            wb = state["prev"]
            return {"bist": wb, "us": 1 - wb}

        iv = invvol_weight(b, u)
        if model == "BASE":
            wb = iv
        else:
            mv = minvar_weight(b, u)
            wb = (1 - COV_BLEND) * iv + COV_BLEND * mv

            if model in {"AR2_COV_DD", "AR2_FULL"}:
                db, du = dd_last(b), dd_last(u)
                mb = max(DD_FLOOR, 1 + DD_STRENGTH * db)
                mu = max(DD_FLOOR, 1 + DD_STRENGTH * du)
                wb = (wb * mb) / (wb * mb + (1 - wb) * mu)

            if model == "AR2_FULL":
                tb, tu = trend_score(b), trend_score(u)
                mb, mu = np.exp(TREND_STRENGTH * tb), np.exp(TREND_STRENGTH * tu)
                wb = (wb * mb) / (wb * mb + (1 - wb) * mu)

                if state["prev"] is not None:
                    d = wb - state["prev"]
                    wb = state["prev"] if abs(d) < HYSTERESIS else state["prev"] + np.clip(d, -MONTHLY_SHIFT, MONTHLY_SHIFT)

        wb = float(np.clip(wb, *C.RP_BOUNDS))
        state["prev"], state["month"] = wb, month
        return {"bist": wb, "us": 1 - wb}

    return weight


def simulate(name, mdb, mdu, fx, irx, start, model):
    old = E.rp_weights
    E.rp_weights = make_risk_weight(model)
    try:
        P.set_hist_rates(irx)
        rb, _ = P.self_financed_returns(mdb)
        ru_usd, _ = P.self_financed_returns(mdu)
        ru = P._to_tl_returns(ru_usd, "us", fx)

        def shadow(mk, d):
            s = rb if mk == "bist" else ru
            return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)

        days = sorted(d for d in set(mdb.dates + mdu.dates) if start <= pd.Timestamp(d))
        st = P.new_state(C.CAPITAL_TL, P.fx_at(fx, start))
        for d in days:
            if d in mdb.didx:
                P.process_day(st, "bist", mdb, mdb.didx[d], fx, shadow)
            if d in mdu.didx:
                P.process_day(st, "us", mdu, mdu.didx[d], fx, shadow)

        nav = pd.DataFrame(st["pf"]["nav_tl"], columns=["date", "mk", "nav", "fx"])
        nav["date"] = pd.to_datetime(nav["date"])
        daily = nav.groupby("date")["nav"].last().sort_index()

        trades = []
        for mk in ("bist", "us"):
            for tr in st["markets"][mk].get("trades", []):
                row = dict(tr)
                row["scenario"] = name
                trades.append(row)
        return Scenario(name, daily, pd.DataFrame(trades))
    finally:
        E.rp_weights = old


def metrics(s):
    nav = s.daily.dropna()
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    dd = (nav / nav.cummax() - 1).min()
    tr = s.trades
    if tr.empty:
        pf = win = avg = np.nan
    else:
        w = tr.loc[tr.ret_pct > 0, "ret_pct"].sum()
        l = -tr.loc[tr.ret_pct <= 0, "ret_pct"].sum()
        pf = w / l if l > 0 else np.inf
        win = (tr.ret_pct > 0).mean() * 100
        avg = tr.ret_pct.mean()
    m = nav.resample("ME").last().pct_change().dropna()
    worst12 = pos12 = np.nan
    if len(m) >= 12:
        r12 = (1 + m).rolling(12).apply(np.prod, raw=True) - 1
        worst12, pos12 = r12.min() * 100, (r12 > 0).mean() * 100
    return dict(CAGR=cagr * 100, TotalX=nav.iloc[-1] / nav.iloc[0], MaxDD=dd * 100,
                PF=pf, Win=win, AvgTrade=avg, Worst12M=worst12, Positive12M=pos12,
                Trades=len(tr))


def bootstrap(base, test):
    a = base.daily.resample("ME").last().pct_change()
    b = test.daily.resample("ME").last().pct_change()
    x = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (x.b - x.a).to_numpy()
    if len(d) < 12:
        return dict(mean=np.nan, median=np.nan, ci5=np.nan, ci95=np.nan, p0=np.nan, pos=np.nan, n=len(d))
    rng = np.random.default_rng(SEED)
    sims = rng.choice(d, (BOOT, len(d)), replace=True).mean(1)
    return dict(mean=d.mean() * 100, median=np.median(d) * 100, ci5=np.quantile(sims, .05) * 100,
                ci95=np.quantile(sims, .95) * 100, p0=(sims <= 0).mean() * 100,
                pos=(d > 0).mean() * 100, n=len(d))


def block(s, a, b):
    nav = s.daily[(s.daily.index >= a) & (s.daily.index <= b)].dropna()
    yrs = max((nav.index[-1] - nav.index[0]).days / 365.25, .01)
    cagr = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100
    dd = (nav / nav.cummax() - 1).min() * 100
    return cagr, dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=YEARS)
    ap.add_argument("--costs", default="0.35,0.50,0.75,1.00,1.25")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    costs = [float(x) for x in args.costs.split(",") if x.strip()]
    if args.years < 8:
        raise ValueError("En az 8 yıl gerekir.")

    bist_panel = DA.get_panel("bist", years=args.years, force_full=False)
    us_panel = DA.get_panel("us", years=args.years, force_full=False)
    fx = DA.get_fx(args.years)
    if bist_panel.empty or us_panel.empty or fx.empty:
        raise RuntimeError("BIST/ABD/USDTRY verisi alınamadı.")

    irx = pd.Series(dtype=float)
    try:
        yf = DA._yf()
        x = yf.download("^IRX", period=f"{args.years}y", interval="1d", auto_adjust=False, progress=False)["Close"]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
        irx = x.dropna()
    except Exception as e:
        print(f"IRX fallback: {e}")

    Wb, Wu = DA.to_wide(bist_panel), DA.to_wide(us_panel)
    bs = C.CANDIDATES["bist"][next(iter(C.CANDIDATES["bist"]))]
    uss = C.CANDIDATES["us"][next(iter(C.CANDIDATES["us"]))]
    md_b = E.MarketData("bist", Wb, spec=bs)
    md_u = E.MarketData("us", Wu, spec=uss)

    i0 = min(WARMUP, len(md_b.dates) - 1, len(md_u.dates) - 1)
    start = max(pd.Timestamp(md_b.dates[i0]), pd.Timestamp(md_u.dates[i0]))
    end = max(pd.Timestamp(md_b.dates[-1]), pd.Timestamp(md_u.dates[-1]))

    models = ["BASE", "AR2_COV", "AR2_COV_DD", "AR2_FULL"]
    all_results, all_metrics = {}, {}
    for cost in costs:
        key = f"{cost:.4f}"
        all_results[key], all_metrics[key] = {}, {}
        for model in models:
            s = simulate(model, clone_cost(md_b, cost), clone_cost(md_u, cost),
                          fx, irx, start, model)
            all_results[key][model] = s
            all_metrics[key][model] = metrics(s)

    ref = f"{REF_COST:.4f}" if f"{REF_COST:.4f}" in all_results else f"{costs[0]:.4f}"
    base = all_results[ref]["BASE"]
    bm = all_metrics[ref]["BASE"]

    blocks = [
        ("EARLY", pd.Timestamp(f"{start.year}-01-01"), pd.Timestamp(f"{min(start.year+4, end.year)}-12-31")),
        ("MID", pd.Timestamp(f"{min(start.year+5, end.year)}-01-01"), pd.Timestamp(f"{min(start.year+8, end.year)}-12-31")),
        ("HOLDOUT", pd.Timestamp(f"{min(start.year+9, end.year)}-01-01"), pd.Timestamp(f"{end.year}-12-31")),
    ]
    blocks = [x for x in blocks if x[1] <= x[2]]

    summaries, details = [], {}
    for model in models[1:]:
        tm = all_metrics[ref][model]
        boot = bootstrap(base, all_results[ref][model])
        bdata = []
        for label, a, b in blocks:
            bc, bd = block(base, a, b)
            tc, td = block(all_results[ref][model], a, b)
            bdata.append(dict(label=label, base_cagr=bc, test_cagr=tc, cagr_delta=tc-bc, base_dd=bd, test_dd=td))
        holdout = bdata[-1]["cagr_delta"] if bdata else np.nan

        stress = []
        for cost in costs:
            k = f"{cost:.4f}"
            mb, mt = all_metrics[k]["BASE"], all_metrics[k][model]
            stress.append(dict(cost=cost, cagr_delta=mt["CAGR"]-mb["CAGR"],
                               dd_delta=mt["MaxDD"]-mb["MaxDD"], pf_delta=mt["PF"]-mb["PF"]))

        good = sum(1 for x in stress if x["cagr_delta"] >= -.50 and x["dd_delta"] <= 1 and x["pf_delta"] >= -.05)
        cagr_delta = tm["CAGR"] - bm["CAGR"]
        dd_delta = tm["MaxDD"] - bm["MaxDD"]
        pf_delta = tm["PF"] - bm["PF"]

        if (cagr_delta >= .25 and dd_delta <= 1 and pf_delta >= -.05 and holdout >= -1.5
                and good >= 4 and np.isfinite(boot["ci5"]) and boot["ci5"] > 0):
            verdict = "DEVELOP FURTHER — STRONG"
        elif (dd_delta <= -2 and cagr_delta >= -.50 and pf_delta >= -.05
              and holdout >= -1.5 and good >= 4):
            verdict = "DEVELOP FURTHER — RISK"
        elif dd_delta <= -1 and cagr_delta >= -1 and good >= 3:
            verdict = "SECOND LOOK"
        else:
            verdict = "REJECT FOR NOW"

        score = (.35*np.clip(cagr_delta/2,-1,1) + .20*np.clip(-dd_delta/5,-1,1)
                 + .15*np.clip(pf_delta/.15,-1,1) + .15*(good/max(1,len(stress)))
                 + .10*np.clip(holdout/2,-1,1)
                 + .05*(1 if boot["ci5"] > 0 else .5 if boot["ci95"] > 0 else 0))

        row = dict(model=model, CAGR=tm["CAGR"], cagr_delta=cagr_delta,
                   MaxDD=tm["MaxDD"], dd_delta=dd_delta, PF=tm["PF"], pf_delta=pf_delta,
                   Win=tm["Win"], holdout=holdout, good=good, ci5=boot["ci5"],
                   decision=verdict, score=score)
        summaries.append(row)
        details[model] = dict(boot=boot, blocks=bdata, stress=stress)

    summaries.sort(key=lambda x: (x["score"], x["cagr_delta"]), reverse=True)

    lines = [
        "# Global Momentum Quant — Adaptive Risk V2 Research",
        f"_Üretim: {pd.Timestamp.now():%Y-%m-%d %H:%M} · veri: {args.years} yıl · ölçüm: {start.date()} → {end.date()}_",
        "",
        "## 1. Amaç",
        "BASE hisse seçimi değiştirilmeden BIST/ABD sermaye tahsis katmanının ikinci sürümü test edildi.",
        "Canlı config/engine/portfolio/signals ve state/order/NAV/trade dosyaları değiştirilmedi.",
        "",
        "## 2. Test edilen modeller",
        "- BASE: mevcut inverse-vol risk parity.",
        "- AR2_COV: inverse-vol + covariance-aware minimum-variance blend.",
        "- AR2_COV_DD: AR2_COV + son 63 günlük drawdown penalty.",
        "- AR2_FULL: AR2_COV_DD + trend persistence + aylık hysteresis/turnover cap.",
        "",
        "## 3. BASE referansı",
        f"- CAGR: **{fmt(bm['CAGR'])}%**",
        f"- Max DD: **{fmt(bm['MaxDD'])}%**",
        f"- PF: **{fmt(bm['PF'])}**",
        f"- Win: **{fmt(bm['Win'])}%**",
        f"- Worst 12M: **{fmt(bm['Worst12M'])}%**",
        f"- Trades: **{bm['Trades']}**",
        f"- Total: **{fmt(bm['TotalX'])}x**",
        "",
        "## 4. Ana sonuç",
        "| Aday | CAGR | ΔCAGR | MaxDD | ΔDD | PF | ΔPF | Win | Holdout ΔCAGR | Stress | CI5 | Karar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summaries:
        lines.append(f"| {r['model']} | {fmt(r['CAGR'])}% | {fmt(r['cagr_delta'])} pp | "
                     f"{fmt(r['MaxDD'])}% | {fmt(r['dd_delta'])} pp | {fmt(r['PF'])} | "
                     f"{fmt(r['pf_delta'])} | {fmt(r['Win'])}% | {fmt(r['holdout'])} pp | "
                     f"{r['good']}/{len(costs)} | {fmt(r['ci5'])} pp | {r['decision']} |")

    lines += ["", "## 5. Öncelik sıralaması", ""]
    for i, r in enumerate(summaries, 1):
        lines.append(f"{i}. **{r['model']}** — skor {fmt(r['score'])} — **{r['decision']}**")

    for r in summaries:
        d = details[r["model"]]
        lines += ["", f"## {r['model']} detay", "",
                  f"- Bootstrap ortalama aylık Δ: **{fmt(d['boot']['mean'])} pp**",
                  f"- Bootstrap medyan Δ: **{fmt(d['boot']['median'])} pp**",
                  f"- Bootstrap CI5–CI95: **[{fmt(d['boot']['ci5'])}, {fmt(d['boot']['ci95'])}] pp**",
                  f"- P(Δ≤0): **{fmt(d['boot']['p0'])}%**",
                  f"- Pozitif ay: **{fmt(d['boot']['pos'])}%**", "",
                  "| Dönem | BASE CAGR | Aday CAGR | ΔCAGR | BASE DD | Aday DD |",
                  "|---|---:|---:|---:|---:|---:|"]
        for x in d["blocks"]:
            lines.append(f"| {x['label']} | {fmt(x['base_cagr'])}% | {fmt(x['test_cagr'])}% | "
                         f"{fmt(x['cagr_delta'])} pp | {fmt(x['base_dd'])}% | {fmt(x['test_dd'])}% |")

    lines += ["", "## 9. Maliyet stresleri", "",
              "| RT cost | BASE CAGR | AR2_COV | AR2_COV_DD | AR2_FULL |",
              "|---:|---:|---:|---:|---:|"]
    for cost in costs:
        k = f"{cost:.4f}"
        mm = all_metrics[k]
        lines.append(f"| {cost:.2f}% | {fmt(mm['BASE']['CAGR'])}% | {fmt(mm['AR2_COV']['CAGR'])}% | "
                     f"{fmt(mm['AR2_COV_DD']['CAGR'])}% | {fmt(mm['AR2_FULL']['CAGR'])}% |")

    lines += ["", "## 10. V2 parametre kaydı",
              f"- COV_WINDOW={COV_WINDOW}, COV_BLEND={COV_BLEND}",
              f"- DD_WINDOW={DD_WINDOW}, DD_FLOOR={DD_FLOOR}, DD_STRENGTH={DD_STRENGTH}",
              f"- TREND_FAST={TREND_FAST}, TREND_SLOW={TREND_SLOW}, TREND_STRENGTH={TREND_STRENGTH}",
              f"- MONTHLY_SHIFT={MONTHLY_SHIFT}, HYSTERESIS={HYSTERESIS}",
              "",
              "## 11. Veri denetimi",
              f"- BIST evreni: **{bist_panel['ticker'].nunique()}**; satır: **{len(bist_panel)}**",
              f"- ABD evreni: **{us_panel['ticker'].nunique()}**; satır: **{len(us_panel)}**",
              f"- BIST: **{bist_panel['tarih'].min().date()} → {bist_panel['tarih'].max().date()}**",
              f"- ABD: **{us_panel['tarih'].min().date()} → {us_panel['tarih'].max().date()}**",
              f"- USD/TRY: **{fx.index.min().date()} → {fx.index.max().date()}**",
              "- Historical S&P membership kullanılmıyorsa survivorship bias sınırlaması sürer.",
              "- Bu çalışma gerçek engine.MarketData + portfolio.process_day mekaniklerini kullanır.",
              "",
              "## 12. Nihai karar",
              ""]
    for i, r in enumerate(summaries, 1):
        lines += [f"### {i}. {r['model']}", f"**{r['decision']}**",
                  "Bu sonuç üretime otomatik aktarılmamalıdır.", ""]
    lines += ["BASE üretim referansı olarak korunur; olumlu aday varsa sonraki adım ayrı final OOS doğrulamasıdır."]

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rapor: {args.out}")
    for r in summaries:
        print(f"{r['model']}: CAGRΔ={fmt(r['cagr_delta'])} pp | DDΔ={fmt(r['dd_delta'])} pp | "
              f"PFΔ={fmt(r['pf_delta'])} | HoldoutΔ={fmt(r['holdout'])} pp | Stress={r['good']}/{len(costs)} | {r['decision']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
