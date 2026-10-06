"""Global Momentum Quant — Tail-Risk Adaptive Position Sizing V1.

Research-only overlay. Production files are NOT modified.

Primary design:
  * OOS holdout predictions from the prior Full Trade Failure / Tail Loss test.
  * strict_signal only: no entry-open leakage.
  * Risk score = cross-sectional percentile blend of predicted P(tail<=-5),
    P(tail<=-10), P(tail<=-15), weights 0.25/0.35/0.40.
  * Only NEW pending BUY orders created by engine.step_market are scaled.
    Existing positions, exits, catastrophe stops, monthly risk-parity weights,
    insurance, and signal selection remain untouched.
  * Size is monotone non-increasing in risk and never above BASE size.
  * Four pre-registered policies: BASE, GENTLE, MODERATE, DEFENSIVE.
  * Primary evaluation is 2022-01-01 onward (OOS model holdout).
  * Cost stress uses the same convention as prior GMQ full-system research:
    both BIST and US RT costs are set to the scenario cost.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import data as DA
import engine as E
import portfolio as P


DEFAULT_COSTS = [0.35, 0.50, 0.75, 1.00, 1.25]
DEFAULT_START = "2022-01-01"
DEFAULT_YEARS = 14
PREDICTOR_FILE = "research_inputs/gmq_full_trade_failure_tail_holdout_predictions.csv"

POLICIES = {
    "BASE": lambda r: 1.0,
    "GENTLE": lambda r: max(0.70, 1.0 - 0.30 * r),
    "MODERATE": lambda r: max(0.50, 1.0 - 0.50 * r),
    "DEFENSIVE": lambda r: max(0.35, 1.0 - 0.65 * r),
}


def clean_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()


class CostContext(contextlib.AbstractContextManager):
    def __init__(self, cost: float):
        self.cost = float(cost)
        self.old = None

    def __enter__(self):
        self.old = {mk: C.MARKETS[mk]["cost_rt_pct"] for mk in ("bist", "us")}
        C.MARKETS["bist"]["cost_rt_pct"] = self.cost
        C.MARKETS["us"]["cost_rt_pct"] = self.cost
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.old is not None:
            C.MARKETS["bist"]["cost_rt_pct"] = self.old["bist"]
            C.MARKETS["us"]["cost_rt_pct"] = self.old["us"]
        return False


class TailRiskBook:
    """Risk lookup keyed by (market, signal_date, ticker)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.lookup: dict[tuple[str, pd.Timestamp, str], float] = {}
        self._build()

    def _build(self):
        df = self.df.copy()
        need = {"market", "ticker", "signal_date", "target", "pred_risk", "feature_set"}
        missing = sorted(need - set(df.columns))
        if missing:
            raise ValueError(f"Prediction dosyasında eksik sütunlar: {missing}")
        df = df[(df["feature_set"] == "strict_signal") & df["target"].isin(["tail_5", "tail_10", "tail_15"])].copy()
        df["signal_date"] = clean_dates(df["signal_date"])
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df["market"] = df["market"].astype(str).str.lower().str.strip()
        df["pred_risk"] = pd.to_numeric(df["pred_risk"], errors="coerce")
        df = df.dropna(subset=["signal_date", "pred_risk"])

        # Previous result file can contain repeated trade rows on the same
        # market/ticker/signal date. Aggregate first; this avoids duplicating a
        # single name's risk score inside same-day cross-sectional ranks.
        df = (df.groupby(["market", "ticker", "signal_date", "target"], as_index=False)["pred_risk"].mean())

        # Pivot the three OOS tail models and require all three.
        wide = df.pivot_table(index=["market", "ticker", "signal_date"],
                              columns="target", values="pred_risk", aggfunc="first").reset_index()
        for c in ("tail_5", "tail_10", "tail_15"):
            if c not in wide.columns:
                raise ValueError(f"OOS predictionlarda {c} bulunamadı")
        wide = wide.dropna(subset=["tail_5", "tail_10", "tail_15"]).copy()

        # Cross-sectional percentile risk is intentionally used instead of raw
        # probability calibration: the three target bases differ materially.
        def pct_rank(x: pd.Series) -> pd.Series:
            return x.rank(method="average", pct=True)

        for c in ("tail_5", "tail_10", "tail_15"):
            wide[f"r_{c}"] = wide.groupby(["market", "signal_date"])[c].transform(pct_rank)

        wide["risk_score"] = (
            0.25 * wide["r_tail_5"] +
            0.35 * wide["r_tail_10"] +
            0.40 * wide["r_tail_15"]
        ).clip(0.0, 1.0)

        for row in wide.itertuples(index=False):
            self.lookup[(row.market, pd.Timestamp(row.signal_date), row.ticker)] = float(row.risk_score)

        self.table = wide[["market", "ticker", "signal_date", "tail_5", "tail_10", "tail_15", "risk_score"]].copy()

    def risk(self, market: str, d, ticker: str) -> float | None:
        return self.lookup.get((market, pd.Timestamp(d).normalize(), str(ticker).upper().strip()))


@dataclass
class ScenarioResult:
    name: str
    cost: float
    daily: pd.Series
    state: dict
    scaling_log: pd.DataFrame



def patch_step_for_policy(risk_book: TailRiskBook, policy_name: str, scaling_rows: list[dict]):
    original = E.step_market
    size_fn = POLICIES[policy_name]

    def wrapped(md, i, st, ctx):
        ev = original(md, i, st, ctx)
        if policy_name == "BASE":
            return ev

        d = pd.Timestamp(md.dates[i]).normalize()
        for order in st.get("pending", []):
            if order.get("side") != "buy" or order.get("tail_sized"):
                continue
            ticker = str(order.get("t", "")).upper()
            r = risk_book.risk(md.mk, d, ticker)
            if r is None or not np.isfinite(r):
                order["tail_sized"] = True
                scaling_rows.append({
                    "date": str(d.date()), "market": md.mk, "ticker": ticker,
                    "tranche": order.get("tranche"), "risk_score": np.nan,
                    "size_factor": 1.0, "amount_before": float(order.get("amount", 0.0)),
                    "amount_after": float(order.get("amount", 0.0)), "missing_risk": True,
                })
                continue

            factor = float(np.clip(size_fn(float(r)), 0.0, 1.0))
            before = float(order.get("amount", 0.0))
            after = before * factor
            order["amount"] = after
            # process_day can later use full-vs-amount to refill a shortfall.
            # Scale `full` too, otherwise a transfer could undo the risk cap.
            if "full" in order:
                order["full"] = float(order["full"]) * factor
            order["tail_risk_score"] = float(r)
            order["tail_size_factor"] = factor
            order["tail_sized"] = True
            scaling_rows.append({
                "date": str(d.date()), "market": md.mk, "ticker": ticker,
                "tranche": order.get("tranche"), "risk_score": float(r),
                "size_factor": factor, "amount_before": before,
                "amount_after": after, "missing_risk": False,
            })
        return ev

    E.step_market = wrapped
    return original


def restore_step(original):
    E.step_market = original


@contextlib.contextmanager
def policy_patch(risk_book: TailRiskBook, policy_name: str, scaling_rows: list[dict]):
    original = patch_step_for_policy(risk_book, policy_name, scaling_rows)
    try:
        yield
    finally:
        restore_step(original)


def load_predictions(path: Path) -> TailRiskBook:
    df = pd.read_csv(path)
    book = TailRiskBook(df)
    if book.table.empty:
        raise RuntimeError("Tail-risk OOS lookup boş.")
    print(f"📌 OOS tail lookup: {len(book.table):,} unique market/ticker/date kayıt")
    return book


def prepare_data(years: int):
    panels = {}
    for mk in ("bist", "us"):
        panels[mk] = DA.get_panel(mk, years=years, force_full=True)
        if panels[mk].empty:
            raise RuntimeError(f"{mk} veri paneli boş.")
    fx = DA.get_fx(years)
    if fx is None or fx.empty:
        raise RuntimeError("USD/TRY verisi alınamadı.")
    W = {mk: DA.to_wide(panel) for mk, panel in panels.items()}
    return W, fx, panels


def load_irx(years: int):
    try:
        d = DA._yf().download("^IRX", period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
        if d is not None and len(d):
            s = d["Close"]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            return s.dropna()
    except Exception as exc:
        print(f"^IRX alınamadı; mevcut üretim varsayılan nakit oranları kullanılacak: {exc}")
    return None


def build_md(W, cost):
    with CostContext(cost):
        mdb = E.MarketData("bist", W["bist"], spec=C.MARKETS["bist"]["spec"])
        mdu = E.MarketData("us", W["us"], spec=C.MARKETS["us"]["spec"])
    return mdb, mdu


def shared_shadow(mdb, mdu, fx):
    # Exact production run_backtest shadow mechanism, with no tail overlay.
    original = E.step_market
    try:
        E.step_market = original
        rb, _ = P.self_financed_returns(mdb)
        ru_usd, _ = P.self_financed_returns(mdu)
    finally:
        E.step_market = original
    P.VOLSCALE["bist"] = P.make_vol_scale(rb)
    P.VOLSCALE["us"] = P.make_vol_scale(ru_usd)
    ru = P._to_tl_returns(ru_usd, "us", fx)
    return rb, ru


def run_main_sim(policy_name: str, risk_book: TailRiskBook, mdb, mdu, fx, rb, ru, start, end, cost, capital):
    scaling_rows: list[dict] = []

    def shadow(mk, d):
        s = rb if mk == "bist" else ru
        return s[s.index < pd.Timestamp(d)].tail(C.RP_WINDOW + 5)

    with policy_patch(risk_book, policy_name, scaling_rows):
        state = P.new_state(capital, P.fx_at(fx, start))
        days = sorted(set(d for d in mdb.dates + mdu.dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)))
        for d in days:
            if d in mdb.didx:
                P.process_day(state, "bist", mdb, mdb.didx[d], fx, shadow)
            if d in mdu.didx:
                P.process_day(state, "us", mdu, mdu.didx[d], fx, shadow)

    nav = pd.DataFrame(state["pf"]["nav_tl"], columns=["date", "mk", "total_tl", "fx"])
    if nav.empty:
        raise RuntimeError(f"{policy_name}/{cost:g}: NAV oluşmadı.")
    nav["date"] = pd.to_datetime(nav["date"])
    daily = nav.groupby("date")["total_tl"].last().sort_index()
    scaling = pd.DataFrame(scaling_rows)
    return ScenarioResult(policy_name, cost, daily, state, scaling)


def run_cost(cost: float, W, fx, risk_book, start, end, capital, irx):
    print(f"\n💰 Maliyet senaryosu: {cost:.2f}% RT")
    with CostContext(cost):
        if irx is not None:
            P.set_hist_rates(irx)
        mdb, mdu = build_md(W, cost)
        rb, ru = shared_shadow(mdb, mdu, fx)
        out = {}
        for policy in POLICIES:
            print(f"  ▶ {policy}")
            out[policy] = run_main_sim(policy, risk_book, mdb, mdu, fx, rb, ru, start, end, cost, capital)
        return out


def all_trades(state: dict) -> pd.DataFrame:
    rows = []
    for mk in ("bist", "us"):
        for tr in state["markets"][mk].get("trades", []):
            r = dict(tr)
            r["market"] = mk
            rows.append(r)
    return pd.DataFrame(rows)


def calc_metrics(sc: ScenarioResult, base_daily: pd.Series | None = None) -> dict:
    nav = sc.daily.dropna().sort_index()
    if len(nav) < 30:
        return {}
    years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.01)
    start_nav = float(nav.iloc[0])
    end_nav = float(nav.iloc[-1])
    cagr = (end_nav / start_nav) ** (1.0 / years) - 1.0 if start_nav > 0 else np.nan
    dd = nav / nav.cummax() - 1.0

    trades = all_trades(sc.state)
    wins = trades[trades["ret_pct"] > 0]["ret_pct"].sum() if len(trades) else 0.0
    losses = -trades[trades["ret_pct"] <= 0]["ret_pct"].sum() if len(trades) else 0.0
    pf = wins / losses if losses > 0 else np.nan
    win = float((trades["ret_pct"] > 0).mean() * 100.0) if len(trades) else np.nan

    monthly = nav.resample("ME").last().pct_change().dropna()
    worst12m = float(((1 + monthly).rolling(12).apply(np.prod, raw=True) - 1.0).min() * 100.0) if len(monthly) >= 12 else np.nan

    tail15 = trades[trades["ret_pct"] <= -15.0] if len(trades) else trades
    tail10 = trades[trades["ret_pct"] <= -10.0] if len(trades) else trades
    neg = trades[trades["ret_pct"] <= 0] if len(trades) else trades
    tail15_loss_share = float((-tail15["pnl"].sum()) / max(-neg["pnl"].sum(), 1e-12) * 100.0) if len(neg) else np.nan
    tail10_loss_share = float((-tail10["pnl"].sum()) / max(-neg["pnl"].sum(), 1e-12) * 100.0) if len(neg) else np.nan

    mean_factor = float(sc.scaling_log["size_factor"].mean()) if len(sc.scaling_log) else 1.0
    reduced_pct = float((sc.scaling_log["size_factor"] < 0.999999).mean() * 100.0) if len(sc.scaling_log) else 0.0
    missing_risk_pct = float(sc.scaling_log["missing_risk"].mean() * 100.0) if len(sc.scaling_log) else 0.0

    out = {
        "policy": sc.name,
        "cost_rt_pct": sc.cost,
        "CAGR_pct": cagr * 100.0,
        "MaxDD_pct": float(dd.min() * 100.0),
        "PF": float(pf),
        "Win_pct": win,
        "Worst12M_pct": worst12m,
        "Trades": int(len(trades)),
        "Final_x": end_nav / start_nav,
        "Tail10_loss_share_pct": tail10_loss_share,
        "Tail15_loss_share_pct": tail15_loss_share,
        "Mean_size_factor": mean_factor,
        "Reduced_buy_pct": reduced_pct,
        "Missing_risk_pct": missing_risk_pct,
    }
    if base_daily is not None and sc.name != "BASE":
        common = pd.concat([base_daily.rename("base"), nav.rename("cand")], axis=1).dropna()
        if len(common):
            bm = common["base"].resample("ME").last().pct_change()
            cm = common["cand"].resample("ME").last().pct_change()
            d = (cm - bm).dropna()
            out["Mean_monthly_delta_pp"] = float(d.mean() * 100.0)
            out["Median_monthly_delta_pp"] = float(d.median() * 100.0)
            out["Positive_months_pct"] = float((d > 0).mean() * 100.0)
            out["Bootstrap_CI5_pp"], out["Bootstrap_CI95_pp"], out["Bootstrap_P_delta_le_0"] = bootstrap_monthly(d)
            out["Holdout_delta_CAGR_pp"] = float(out["CAGR_pct"] - ((base_daily.iloc[-1] / base_daily.iloc[0]) ** (365.25 / max((base_daily.index[-1]-base_daily.index[0]).days,1)) - 1.0) * 100.0)
    return out


def bootstrap_monthly(delta: pd.Series, n=3000, seed=1701):
    x = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(float)
    if len(x) < 8:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n, float)
    for i in range(n):
        means[i] = rng.choice(x, size=len(x), replace=True).mean() * 100.0
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95)), float((means <= 0).mean())


def write_outputs(results: dict, out_dir: Path, risk_book: TailRiskBook):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    stress_rows = []
    base_by_cost = {}

    for cost, bundle in results.items():
        base = bundle["BASE"]
        base_by_cost[cost] = base.daily
        bm = calc_metrics(base)
        rows.append(bm)
        for policy in ("GENTLE", "MODERATE", "DEFENSIVE"):
            rows.append(calc_metrics(bundle[policy], base.daily))

        for policy, sc in bundle.items():
            m = calc_metrics(sc, base.daily if policy != "BASE" else None)
            stress_rows.append({
                "cost_rt_pct": cost,
                "policy": policy,
                "CAGR_pct": m.get("CAGR_pct"),
                "MaxDD_pct": m.get("MaxDD_pct"),
                "PF": m.get("PF"),
                "Win_pct": m.get("Win_pct"),
                "Worst12M_pct": m.get("Worst12M_pct"),
                "Tail10_loss_share_pct": m.get("Tail10_loss_share_pct"),
                "Tail15_loss_share_pct": m.get("Tail15_loss_share_pct"),
                "Mean_size_factor": m.get("Mean_size_factor"),
                "Reduced_buy_pct": m.get("Reduced_buy_pct"),
                "Mean_monthly_delta_pp": m.get("Mean_monthly_delta_pp"),
                "Bootstrap_CI5_pp": m.get("Bootstrap_CI5_pp"),
                "Bootstrap_CI95_pp": m.get("Bootstrap_CI95_pp"),
                "Bootstrap_P_delta_le_0": m.get("Bootstrap_P_delta_le_0"),
            })

    metrics_df = pd.DataFrame(rows)
    stress_df = pd.DataFrame(stress_rows)
    metrics_df.to_csv(out_dir / "gmq_tail_risk_adaptive_sizing_metrics.csv", index=False)
    stress_df.to_csv(out_dir / "gmq_tail_risk_adaptive_sizing_stress.csv", index=False)

    all_scaling = []
    all_trades = []
    for cost, bundle in results.items():
        for policy, sc in bundle.items():
            if len(sc.scaling_log):
                x = sc.scaling_log.copy()
                x.insert(0, "policy", policy)
                x.insert(0, "cost_rt_pct", cost)
                all_scaling.append(x)
            tr = all_trades_state(sc.state)
            if len(tr):
                tr.insert(0, "policy", policy)
                tr.insert(0, "cost_rt_pct", cost)
                all_trades.append(tr)
    pd.concat(all_scaling, ignore_index=True).to_csv(out_dir / "gmq_tail_risk_adaptive_sizing_orders.csv", index=False) if all_scaling else pd.DataFrame().to_csv(out_dir / "gmq_tail_risk_adaptive_sizing_orders.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_csv(out_dir / "gmq_tail_risk_adaptive_sizing_trades.csv", index=False) if all_trades else pd.DataFrame().to_csv(out_dir / "gmq_tail_risk_adaptive_sizing_trades.csv", index=False)

    risk_book.table.to_csv(out_dir / "gmq_tail_risk_oos_scores.csv", index=False)

    # Pairwise 0.35% holdout comparison, plus a simple non-binding research flag.
    primary = metrics_df[metrics_df["cost_rt_pct"] == 0.35].copy()
    base = primary[primary["policy"] == "BASE"].iloc[0]
    gate_rows = []
    for _, r in primary[primary["policy"] != "BASE"].iterrows():
        gate_rows.append({
            "policy": r["policy"],
            "CAGR_delta_pp": r["CAGR_pct"] - base["CAGR_pct"],
            "MaxDD_delta_pp": r["MaxDD_pct"] - base["MaxDD_pct"],
            "PF_delta": r["PF"] - base["PF"],
            "Tail15_loss_share_delta_pp": r["Tail15_loss_share_pct"] - base["Tail15_loss_share_pct"],
            "bootstrap_CI5_pp": r.get("Bootstrap_CI5_pp", np.nan),
            "research_flag": "REVIEW" if (
                (r["Tail15_loss_share_pct"] < base["Tail15_loss_share_pct"] and r["PF"] >= base["PF"])
                or (r["MaxDD_pct"] >= base["MaxDD_pct"] + 2.0 and r["CAGR_pct"] >= base["CAGR_pct"] - 0.5)
            ) else "NO_CLEAR_ADVANTAGE",
        })
    pd.DataFrame(gate_rows).to_csv(out_dir / "gmq_tail_risk_adaptive_sizing_primary_review.csv", index=False)

    report = build_report(metrics_df, stress_df, gate_rows)
    (out_dir / "gmq_tail_risk_adaptive_sizing_raporu.md").write_text(report, encoding="utf-8")


def all_trades_state(state: dict) -> pd.DataFrame:
    return all_trades(state)


def build_report(metrics_df: pd.DataFrame, stress_df: pd.DataFrame, gate_rows: list[dict]) -> str:
    now = pd.Timestamp.utcnow().tz_localize(None)
    lines = [
        "# Global Momentum Quant — Tail-Risk Adaptive Position Sizing V1",
        f"_Üretim: {now.strftime('%Y-%m-%d %H:%M')} UTC_",
        "",
        "## 1. Araştırma amacı",
        "Önceki OOS tail-loss predictor'ın ürettiği strict-signal tail_5/tail_10/tail_15 risk sıralamasını, işlem seçimi yerine yalnızca yeni alım notionalini monotonic olarak küçülten bir sizing overlay olarak test etmek.",
        "Canlı `config.py`, `engine.py`, `portfolio.py`, `signals.py` değiştirilmez.",
        "Primary dönem: 2022-01-01 sonrası OOS holdout.",
        "",
        "## 2. Risk skoru",
        "- strict_signal OOS tahminleri kullanıldı; entry_open kullanılmadı.",
        "- Aynı market + signal_date içindeki percentile rank'lar kullanıldı.",
        "- Blend: tail_5=0.25, tail_10=0.35, tail_15=0.40.",
        "- Aynı market/ticker/signal_date tekrarları hedef bazında ortalandı.",
        "",
        "## 3. Pre-registered sizing politikaları",
        "| Policy | Formula | Floor |",
        "|---|---|---:|",
        "| BASE | 1.00 | 1.00 |",
        "| GENTLE | max(0.70, 1 - 0.30*risk) | 0.70 |",
        "| MODERATE | max(0.50, 1 - 0.50*risk) | 0.50 |",
        "| DEFENSIVE | max(0.35, 1 - 0.65*risk) | 0.35 |",
        "",
        "Sadece `engine.step_market()` tarafından o anda oluşturulan yeni BUY pending order'ları ölçeklenir. Mevcut pozisyonlar, satışlar, stoplar, seçim, risk-parity ağırlıkları ve sigorta mekanizması aynı kalır.",
        "",
        "## 4. 0.35% RT primary sonuçları",
    ]
    prim = metrics_df[metrics_df["cost_rt_pct"] == 0.35].copy()
    if not prim.empty:
        cols = ["policy", "CAGR_pct", "MaxDD_pct", "PF", "Win_pct", "Worst12M_pct", "Tail10_loss_share_pct", "Tail15_loss_share_pct", "Mean_size_factor", "Reduced_buy_pct", "Mean_monthly_delta_pp", "Bootstrap_CI5_pp", "Bootstrap_CI95_pp", "Bootstrap_P_delta_le_0"]
        lines.append(prim[cols].to_markdown(index=False, floatfmt=".3f"))
    lines += ["", "## 5. Maliyet stresi", ""]
    if not stress_df.empty:
        pivot = stress_df.pivot(index="cost_rt_pct", columns="policy", values="CAGR_pct").reset_index()
        lines.append(pivot.to_markdown(index=False, floatfmt=".3f"))
    lines += ["", "## 6. Primary review", ""]
    if gate_rows:
        lines.append(pd.DataFrame(gate_rows).to_markdown(index=False, floatfmt=".3f"))
    lines += [
        "",
        "## 7. Karar disiplini",
        "Bu çalışma otomatik canlı entegrasyon kararı vermez.",
        "Bir adayın sonraki aşamaya taşınması için yalnızca trade-level getiri değil; MaxDD, PF, tail-loss contribution, bootstrap aylık farkı ve tüm maliyet streslerindeki tutarlılık birlikte incelenmelidir.",
        "Özellikle sizing overlay yalnızca aşağı yönlü boyutlandırdığı için, kârlı fakat tail-risk skoru yüksek işlemlerden ne kadar alpha kaybedildiği ayrıca kontrol edilmelidir.",
        "",
        "## 8. Veri/mimari sınırlar",
        "- OOS model tahminleri önceki Full Trade Failure / Tail Loss çalışmasından alınmıştır.",
        "- Risk lookup yalnızca holdout dönemi için kullanılır; primary backtest 2022+ olarak tutulur.",
        "- Test araştırma amaçlıdır; üretim state/order/NAV dosyaları yazılmaz.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=float(C.CAPITAL_TL))
    ap.add_argument("--costs", default=",".join(str(x) for x in DEFAULT_COSTS))
    ap.add_argument("--predictions", default=PREDICTOR_FILE)
    ap.add_argument("--out-dir", default="local_output/gmq_tail_risk_adaptive_sizing")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return

    root = Path(__file__).resolve().parent
    pred_path = Path(args.predictions)
    if not pred_path.is_absolute():
        pred_path = root / pred_path
    if not pred_path.exists():
        raise FileNotFoundError(f"OOS prediction dosyası bulunamadı: {pred_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    risk_book = load_predictions(pred_path)
    W, fx, panels = prepare_data(args.years)
    irx = load_irx(args.years)
    start = pd.Timestamp(args.start).normalize()
    natural_end = max(pd.Timestamp(max(W["bist"]["c"].index)), pd.Timestamp(max(W["us"]["c"].index)))
    end = pd.Timestamp(args.end).normalize() if args.end else natural_end
    if start >= end:
        raise ValueError("start >= end")

    costs = [float(x.strip()) for x in args.costs.split(",") if x.strip()]
    if not costs:
        raise ValueError("En az bir cost gerekli")

    results = {}
    for cost in costs:
        results[cost] = run_cost(cost, W, fx, risk_book, start, end, args.capital, irx)
    write_outputs(results, out_dir, risk_book)

    print(f"\n✅ Araştırma tamamlandı: {out_dir}")
    print(f"📄 Rapor: {out_dir / 'gmq_tail_risk_adaptive_sizing_raporu.md'}")


def smoke():
    """No network; verifies risk aggregation + monotonic sizing only."""
    rows = []
    d = pd.Timestamp("2024-01-02")
    for market in ("bist", "us"):
        for i in range(10):
            t = f"T{i:02d}"
            for target, base in (("tail_5", 0.10), ("tail_10", 0.05), ("tail_15", 0.02)):
                rows.append({"market": market, "ticker": t, "signal_date": d,
                             "target": target, "feature_set": "strict_signal",
                             "pred_risk": base + i * 0.01})
    book = TailRiskBook(pd.DataFrame(rows))
    for p, fn in POLICIES.items():
        vals = [fn(x) for x in np.linspace(0, 1, 101)]
        assert all(0 < x <= 1 for x in vals)
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    assert len(book.table) == 20
    print("SMOKE OK: risk lookup + monotonic policies")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise
