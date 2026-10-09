#!/usr/bin/env python3
"""GMQ V4.2.1 — diagnostic BASE replay and order/fill/cycle reconciliation.

Research-only. Runs one unmodified BASE simulation in memory and compares:
(1) legacy V2/V3 feature construction and (2) V4.1 execution-verified cycles,
using the SAME panels, warm-up start, BASE state and evaluation start.
No candidate sizing policy is applied. No production files are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from types import SimpleNamespace
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "V4.2.1 BASE Replay Diagnostics"
DEFAULT_YEARS = 14
DEFAULT_START = "2018-01-01"
MIN_CLASS_N = 30


def norm_date(value: Any) -> pd.Timestamp | pd.NaT:
    x = pd.to_datetime(value, errors="coerce")
    if pd.isna(x):
        return pd.NaT
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def key_market_ticker_tranche_date(row: dict[str, Any], date_field: str) -> tuple[str, str, int, str] | None:
    market = str(row.get("market", "")).strip().lower()
    ticker = str(row.get("ticker", row.get("t", ""))).strip()
    try:
        tranche = int(row.get("tranche", -1))
    except (TypeError, ValueError):
        tranche = -1
    dt = norm_date(row.get(date_field))
    if not market or not ticker or pd.isna(dt):
        return None
    return market, ticker, tranche, dt.strftime("%Y-%m-%d")


def composite_identity_audit(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    if df.empty:
        return {"rows": 0, "unique_identity_rows": 0, "duplicate_identity_rows": 0, "missing_identity_rows": 0}
    present = [c for c in columns if c in df.columns]
    if len(present) != len(columns):
        return {"rows": int(len(df)), "unique_identity_rows": 0, "duplicate_identity_rows": 0,
                "missing_identity_rows": int(len(df)), "missing_required_columns": len(columns) - len(present)}
    tmp = df[columns].copy()
    for c in ("signal_date", "entry_date", "exit_date"):
        if c in tmp:
            tmp[c] = pd.to_datetime(tmp[c], errors="coerce").dt.normalize()
    missing = tmp.isna().any(axis=1)
    duplicate = tmp.duplicated(columns, keep="first") & ~missing
    unique = tmp.loc[~missing, columns].drop_duplicates().shape[0]
    return {"rows": int(len(df)), "unique_identity_rows": int(unique),
            "duplicate_identity_rows": int(duplicate.sum()), "missing_identity_rows": int(missing.sum())}


def extract_raw_cycles(state: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cycle_idx = 0
    for market in ("bist", "us"):
        trade_rows = state.get("markets", {}).get(market, {}).get("trades", []) or []
        for trade in trade_rows:
            row = dict(trade)
            row["market"] = market
            row["raw_cycle_id"] = f"cycle-{cycle_idx:07d}"
            row["reason_text"] = str(row.get("reason", ""))
            low_reason = row["reason_text"].strip().lower()
            row["continuation_cycle"] = ("döngü yenilendi" in low_reason or "devam" in low_reason or "continue" in low_reason)
            rows.append(row)
            cycle_idx += 1
    return pd.DataFrame(rows)


def capture_base_with_ledger(md_b, md_u, fx, start: pd.Timestamp, end: pd.Timestamp | None,
                              cost_rt: float, capital: float) -> dict[str, Any]:
    """Replay BASE once and retain an explicit in-memory BUY signal/fill ledger.

    engine.py fill events do not carry a persistent order id. If more than one
    pending BUY shares market/ticker/tranche on the same day, the mapping is
    therefore marked FIFO-inferred/ambiguous, not falsely claimed as exact.
    """
    import engine as E
    import portfolio as P
    import gmq_momentum_entry_timing_chase_avoidance_v4_1 as V41

    old_fc_b, old_fc_u, rb, ru = V41._prepare_shadow(md_b, md_u, fx, cost_rt)
    old_step = E.step_market
    state = P.new_state(capital, P.fx_at(fx, start))
    orders: list[dict[str, Any]] = []
    order_by_id: dict[str, dict[str, Any]] = {}
    fills: list[dict[str, Any]] = []
    orphan_fills: list[dict[str, Any]] = []
    counter = 0

    def order_key(order: dict[str, Any], market: str) -> tuple[str, str, int]:
        try:
            tr = int(order.get("tranche", -1))
        except (TypeError, ValueError):
            tr = -1
        return market.lower(), str(order.get("t", order.get("ticker", ""))).strip(), tr

    def instrumented_step(md, i, st, ctx):
        nonlocal counter
        day = pd.Timestamp(md.dates[i]).normalize()
        before = [o for o in st.get("pending", []) if o.get("side") == "buy"]
        before_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        for order in before:
            before_by_key[order_key(order, md.mk)].append(order)

        ev = old_step(md, i, st, ctx)

        # Register new pending BUY orders after the original strategy step.
        for order in st.get("pending", []) or []:
            if order.get("side") != "buy":
                continue
            if order.get("_gmq_v42_replay_order_id"):
                continue
            order_id = f"buy-{counter:08d}"
            counter += 1
            order["_gmq_v42_replay_order_id"] = order_id  # in-memory research metadata only
            row = {
                "order_id": order_id,
                "market": str(md.mk).lower(),
                "ticker": str(order.get("t", "")).strip(),
                "tranche": int(order.get("tranche", -1)),
                "signal_date": day.strftime("%Y-%m-%d"),
                "signal_amount": float(order.get("amount", 0.0) or 0.0),
                "cycle_end": order.get("cycle_end"),
                "status": "pending",
                "fill_date": "",
                "fill_px": np.nan,
                "fill_qty": np.nan,
                "fill_match_ambiguous": False,
                "same_key_pending_count_at_fill": 0,
                "status_date": "",
                "status_note": "",
            }
            orders.append(row)
            order_by_id[order_id] = row

        # A fill event has no source order id. Pair only to a BUY that was
        # pending BEFORE this step; a newly created order cannot fill today.
        used_by_key: dict[tuple[str, str, int], int] = defaultdict(int)
        for fill_number, fill in enumerate(ev.get("fills", []) or []):
            if fill.get("side") != "AL":
                continue
            key = order_key(fill, md.mk)
            candidates = before_by_key.get(key, [])
            pos = used_by_key[key]
            used_by_key[key] += 1
            if pos >= len(candidates):
                orphan_fills.append({
                    "market": str(md.mk).lower(), "ticker": str(fill.get("ticker", "")).strip(),
                    "tranche": int(fill.get("tranche", -1)), "fill_date": day.strftime("%Y-%m-%d"),
                    "fill_px": fill.get("px"), "fill_qty": fill.get("qty"),
                    "reason": "BUY fill event has no known pending BUY order in this replay step",
                })
                continue
            order = candidates[pos]
            order_id = str(order.get("_gmq_v42_replay_order_id", ""))
            signal = order_by_id.get(order_id)
            if signal is None:
                orphan_fills.append({
                    "market": str(md.mk).lower(), "ticker": str(fill.get("ticker", "")).strip(),
                    "tranche": int(fill.get("tranche", -1)), "fill_date": day.strftime("%Y-%m-%d"),
                    "fill_px": fill.get("px"), "fill_qty": fill.get("qty"),
                    "reason": "pending BUY object has no diagnostic order record",
                })
                continue
            ambiguous = len(candidates) > 1
            signal.update({
                "status": "filled",
                "fill_date": day.strftime("%Y-%m-%d"),
                "fill_px": float(fill.get("px", np.nan)),
                "fill_qty": float(fill.get("qty", np.nan)),
                "fill_match_ambiguous": bool(ambiguous),
                "same_key_pending_count_at_fill": int(len(candidates)),
                "status_date": day.strftime("%Y-%m-%d"),
                "status_note": "FIFO-inferred; engine fill event has no persistent order id" if ambiguous else "One matching pending BUY for market/ticker/tranche in this step",
            })
            fills.append({
                "execution_id": order_id,
                "market": str(md.mk).lower(),
                "ticker": str(fill.get("ticker", order.get("t", ""))).strip(),
                "tranche": int(fill.get("tranche", order.get("tranche", -1))),
                "signal_date": signal["signal_date"],
                "entry_date": day.strftime("%Y-%m-%d"),
                "side": "AL",
                "px": float(fill.get("px", np.nan)),
                "qty": float(fill.get("qty", np.nan)),
                "signal_amount": signal["signal_amount"],
                "order_match_ambiguous": bool(ambiguous),
                "same_key_pending_count_at_fill": int(len(candidates)),
            })

        current_pending_ids = {
            str(o.get("_gmq_v42_replay_order_id", ""))
            for o in (st.get("pending", []) or [])
            if o.get("side") == "buy" and o.get("_gmq_v42_replay_order_id")
        }
        filled_ids = {str(r["execution_id"]) for r in fills if r["entry_date"] == day.strftime("%Y-%m-%d") and r["market"] == str(md.mk).lower()}
        for order in before:
            order_id = str(order.get("_gmq_v42_replay_order_id", ""))
            signal = order_by_id.get(order_id)
            if signal is None or signal["status"] == "filled":
                continue
            if order_id not in current_pending_ids and order_id not in filled_ids:
                signal["status"] = "cancelled_no_fill"
                signal["status_date"] = day.strftime("%Y-%m-%d")
                signal["status_note"] = "Previous pending BUY disappeared during engine step without a matching BUY fill; inspect engine event notes/shortfall"
        return ev

    E.step_market = instrumented_step
    try:
        def shadow(market, day):
            series = rb if market == "bist" else ru
            return series[series.index < pd.Timestamp(day)].tail(P.C.RP_WINDOW + 5)

        days = V41._days_for_run(md_b, md_u, start, end)
        for day in days:
            if day in md_b.didx:
                P.process_day(state, "bist", md_b, md_b.didx[day], fx, shadow)
            if day in md_u.didx:
                P.process_day(state, "us", md_u, md_u.didx[day], fx, shadow)
    finally:
        E.step_market = old_step
        md_b.fc, md_u.fc = old_fc_b, old_fc_u

    pending_ids = {
        str(o.get("_gmq_v42_replay_order_id", ""))
        for market_state in (state.get("markets", {}) or {}).values()
        for o in (market_state.get("pending", []) or [])
        if o.get("side") == "buy" and o.get("_gmq_v42_replay_order_id")
    }
    for row in orders:
        if row["status"] == "pending":
            if row["order_id"] in pending_ids:
                row["status"] = "pending_at_end"
                row["status_note"] = "No future simulation bar remained to determine whether the pending BUY would fill"
            else:
                row["status"] = "unresolved"
                row["status_note"] = "Order left active pending set without a recorded fill/cancellation reason"

    nav = pd.DataFrame(state["pf"]["nav_tl"], columns=["date", "mk", "total_tl", "fx"])
    if not nav.empty:
        nav["date"] = pd.to_datetime(nav["date"], errors="coerce")
        daily = nav.groupby("date")["total_tl"].last().sort_index()
    else:
        daily = pd.Series(dtype=float)
    return {"state": state, "daily": daily, "orders": pd.DataFrame(orders),
            "fills": pd.DataFrame(fills), "orphan_fills": pd.DataFrame(orphan_fills),
            "start_full": pd.Timestamp(start).normalize(), "days_simulated": len(days)}


def reconcile_cycles(raw_cycles: pd.DataFrame, fills: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mirror V4.1's market/ticker/tranche/entry-date queue match and expose each row."""
    fill_map: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    if not fills.empty:
        for row in fills.to_dict("records"):
            key = key_market_ticker_tranche_date(row, "entry_date")
            if key is not None:
                fill_map[key].append(row)
    used: dict[tuple[str, str, int, str], int] = defaultdict(int)
    out: list[dict[str, Any]] = []
    for cycle in raw_cycles.to_dict("records"):
        row = dict(cycle)
        if "continuation_cycle" not in row:
            reason = str(row.get("reason", row.get("reason_text", ""))).strip().lower()
            row["continuation_cycle"] = ("döngü yenilendi" in reason or "devam" in reason or "continue" in reason)
        key = key_market_ticker_tranche_date(row, "entry_date")
        row["cycle_match_status"] = "identity_missing" if key is None else "unmatched_no_buy_fill"
        row["matched_execution_id"] = ""
        row["matched_signal_date"] = ""
        row["same_fill_key_count"] = 0
        if key is not None:
            q = fill_map.get(key, [])
            used_pos = used[key]
            if used_pos < len(q):
                match = q[used_pos]
                used[key] += 1
                row["cycle_match_status"] = "matched_fifo_inferred" if len(q) > 1 else "matched_unique_key"
                row["matched_execution_id"] = str(match.get("execution_id", ""))
                row["matched_signal_date"] = str(match.get("signal_date", ""))
                row["same_fill_key_count"] = int(len(q))
        out.append(row)
    ledger = pd.DataFrame(out)
    matched_keys = {r["matched_execution_id"] for r in out if r.get("matched_execution_id")}
    fill_ids = set(fills["execution_id"].astype(str)) if not fills.empty and "execution_id" in fills else set()
    counts = {
        "raw_closed_cycle_rows": int(len(raw_cycles)),
        "matched_cycle_rows": int(ledger["cycle_match_status"].str.startswith("matched").sum()) if not ledger.empty else 0,
        "unmatched_cycle_rows": int(ledger["cycle_match_status"].eq("unmatched_no_buy_fill").sum()) if not ledger.empty else 0,
        "cycle_rows_missing_identity": int(ledger["cycle_match_status"].eq("identity_missing").sum()) if not ledger.empty else 0,
        "continuation_cycle_rows": int(raw_cycles.get("continuation_cycle", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not raw_cycles.empty else 0,
        "fill_rows": int(len(fills)),
        "fill_rows_without_closed_cycle": int(len(fill_ids - matched_keys)),
        "cycle_match_ambiguous_fifo_rows": int(ledger["cycle_match_status"].eq("matched_fifo_inferred").sum()) if not ledger.empty else 0,
    }
    return ledger, counts


def normalize_and_filter(ds: pd.DataFrame, start: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, int]]:
    x = ds.copy()
    before = int(len(x))
    for col in ("signal_date", "entry_date", "exit_date"):
        if col in x.columns:
            x[col] = pd.to_datetime(x[col], errors="coerce")
            try:
                x[col] = x[col].dt.tz_localize(None)
            except (TypeError, AttributeError):
                pass
            x[col] = x[col].dt.normalize()
    if "ret_pct" in x.columns:
        x["ret_pct"] = pd.to_numeric(x["ret_pct"], errors="coerce")
    valid = x.dropna(subset=[c for c in ("signal_date", "ret_pct") if c in x.columns]).copy()
    invalid = before - int(len(valid))
    evaluation = valid.loc[valid["signal_date"] >= start].copy()
    pre_start = int(len(valid) - len(evaluation))
    return evaluation.sort_values([c for c in ("signal_date", "market", "ticker", "tranche") if c in evaluation.columns]).reset_index(drop=True), {
        "feature_rows_before_date_cleanup": before,
        "feature_rows_dropped_invalid_signal_or_return": invalid,
        "feature_rows_before_start_filter_valid": int(len(valid)),
        "feature_rows_removed_before_start": pre_start,
        "feature_rows_in_evaluation_window": int(len(evaluation)),
    }


def count_classes(pred: pd.DataFrame, version: str) -> dict[str, Any]:
    def sum_bool(col: str) -> int | None:
        if col not in pred.columns:
            return None
        return int(pd.Series(pred[col]).fillna(False).astype(bool).sum())
    candidate = sum_bool("chase_candidate")
    dangerous = sum_bool("dangerous_chase")
    healthy = sum_bool("healthy_chase")
    if dangerous is None or healthy is None:
        gate = "NOT_ASSESSABLE_MISSING_CLASS_COLUMNS"
    elif min(dangerous, healthy) < MIN_CLASS_N:
        gate = "INCONCLUSIVE_LOW_N"
    else:
        gate = "SAMPLE_FLOOR_MET_NOT_PREDICTIVE_VALIDATION"
    return {"dataset": version, "n": int(len(pred)), "chase_candidates": candidate,
            "dangerous_chase": dangerous, "healthy_chase": healthy,
            "minimum_class_n_diagnostic_floor": MIN_CLASS_N, "classification_evidence_status": gate,
            "applied_sizing_events": None, "sizing_status": "NOT_RUN_DIAGNOSTIC_ONLY"}


def policy_trigger_rows(pred: pd.DataFrame, version: str, module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policies = getattr(module, "POLICIES", {})
    for name, spec in policies.items():
        mode = spec.get("mode", "none") if isinstance(spec, dict) else "none"
        try:
            cond = module.policy_condition(pred, mode)
            count = int(pd.Series(cond, index=pred.index).fillna(False).astype(bool).sum())
            status = "COUNTED_TRIGGER_CONDITION_ONLY"
        except Exception as exc:
            count = None
            status = f"NOT_ASSESSABLE:{type(exc).__name__}"
        rows.append({"dataset": version, "policy": name, "trigger_count": count,
                     "applied_sizing_count": None, "applied_sizing_status": "NOT_RUN_DIAGNOSTIC_ONLY",
                     "evidence_status": status})
    return rows


def historical_artifact_summary(prior_dir: Path, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read prior V4.2 audit's downloaded source prediction CSVs, if supplied."""
    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    expected = {
        "V2": "gmq_momentum_entry_timing_chase_avoidance_v2_predictions.csv",
        "V3": "gmq_momentum_entry_timing_chase_avoidance_v3_predictions.csv",
        "V4.1": "gmq_momentum_entry_timing_chase_avoidance_v4_1_predictions.csv",
    }
    for version, filename in expected.items():
        candidates = list(prior_dir.rglob(filename)) if prior_dir.exists() else []
        candidates = [p for p in candidates if "smoke_" not in str(p)]
        df = pd.DataFrame()
        source = "not_found"
        if candidates:
            # Prefer the extracted source artifact path, not a loose smoke/report file.
            candidates.sort(key=lambda p: ("_diagnostic_inputs" not in str(p), len(str(p))))
            source = str(candidates[0])
            try:
                df = pd.read_csv(candidates[0], low_memory=False)
            except Exception as exc:
                source = f"read_error:{type(exc).__name__}:{candidates[0]}"
        if not df.empty and "signal_date" in df.columns:
            df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce").dt.normalize()
            frames[version] = df
            eval_df = df.loc[df["signal_date"] >= start]
            identity_cols = [c for c in ("market", "ticker", "signal_date", "entry_date", "tranche") if c in df.columns]
            counts = composite_identity_audit(eval_df, identity_cols) if identity_cols else {"rows": len(eval_df), "unique_identity_rows": 0, "duplicate_identity_rows": 0, "missing_identity_rows": len(eval_df)}
            rows.append({"version": version, "artifact_found": True, "source": source,
                         "prediction_rows_all": int(len(df)), "signal_date_min": str(df["signal_date"].min().date()) if df["signal_date"].notna().any() else "",
                         "signal_date_max": str(df["signal_date"].max().date()) if df["signal_date"].notna().any() else "",
                         "prediction_rows_from_start": int(len(eval_df)), **{f"start_{k}": v for k, v in counts.items()}})
        else:
            rows.append({"version": version, "artifact_found": bool(candidates), "source": source,
                         "prediction_rows_all": None, "signal_date_min": "", "signal_date_max": "",
                         "prediction_rows_from_start": None, "start_rows": None,
                         "note": "Historical prediction artifact unavailable; not treated as zero."})
    history = pd.DataFrame(rows)
    source_meta_path = prior_dir / "gmq_momentum_entry_timing_chase_avoidance_v4_2_population_summary.csv"
    if not history.empty and source_meta_path.exists():
        try:
            source_meta = pd.read_csv(source_meta_path, low_memory=False)
            meta_cols = [c for c in ("version", "source_run_id", "source_run_number", "source_created_at", "source_head_sha", "source_note") if c in source_meta.columns]
            if "version" in meta_cols:
                source_meta = source_meta[meta_cols].drop_duplicates("version")
                history = history.merge(source_meta, on="version", how="left", suffixes=("", "_source"))
        except Exception:
            # Source commit metadata is useful, but a malformed summary must not erase the row-count audit.
            pass

    # Compare historical artifact row counts within their common signal-date range.
    comp_rows: list[dict[str, Any]] = []
    if len(frames) == 3:
        common_start = max(df["signal_date"].min() for df in frames.values())
        common_end = min(df["signal_date"].max() for df in frames.values())
        normalized: dict[str, pd.DataFrame] = {}
        for ver, df in frames.items():
            normalized[ver] = df.loc[df["signal_date"].between(common_start, common_end)].copy()
            comp_rows.append({"comparison": "HISTORICAL_COMMON_DATE_WINDOW", "version": ver,
                              "window_start": str(common_start.date()), "window_end": str(common_end.date()),
                              "rows": int(len(normalized[ver])),
                              "interpretation": "Date-aligned archive rows; does not resolve differing trade-cycle definitions or source data snapshots."})
        # Four-field historical key sensitivity. V4.1 drops tranche only for this overlap diagnostic.
        id_cols = ["market", "ticker", "signal_date", "entry_date"]
        sets: dict[str, set[tuple[Any, ...]]] = {}
        for ver, df in normalized.items():
            if all(c in df.columns for c in id_cols):
                t = df[id_cols].copy()
                t["signal_date"] = pd.to_datetime(t["signal_date"], errors="coerce").dt.normalize()
                t["entry_date"] = pd.to_datetime(t["entry_date"], errors="coerce").dt.normalize()
                sets[ver] = set(map(tuple, t.dropna().itertuples(index=False, name=None)))
        if len(sets) == 3:
            for ver in ("V2", "V3", "V4.1"):
                comp_rows.append({"comparison": "HISTORICAL_4KEY_SENSITIVITY", "version": ver,
                                  "window_start": str(common_start.date()), "window_end": str(common_end.date()),
                                  "rows": int(len(normalized[ver])), "unique_composite_identity_rows": int(len(sets[ver])),
                                  "overlap_with_v2": int(len(sets[ver] & sets["V2"])),
                                  "overlap_with_v3": int(len(sets[ver] & sets["V3"])),
                                  "overlap_with_v4_1": int(len(sets[ver] & sets["V4.1"])),
                                  "interpretation": "Four-key identity ignores V4.1 tranche and is sensitivity-only, not exact identity proof."})
    return history, pd.DataFrame(comp_rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_report(out: Path, meta: dict[str, Any], stages: pd.DataFrame,
                 populations: pd.DataFrame, cycles: pd.DataFrame,
                 orders: pd.DataFrame, fills: pd.DataFrame,
                 orphan_fills: pd.DataFrame, classes: pd.DataFrame,
                 history: pd.DataFrame, history_comp: pd.DataFrame) -> None:
    stage_map = {str(r.stage): r.n for r in stages.itertuples(index=False)}
    pop_lines = ["| Dataset | Feature rows pre-start | Evaluation rows | Unique identities | Duplicates | Missing keys | Chase | Dangerous | Healthy | Evidence |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in populations.to_dict("records"):
        def fmt(v): return "not recorded" if pd.isna(v) else str(int(v)) if isinstance(v, (int, np.integer, float, np.floating)) and float(v).is_integer() else str(v)
        pop_lines.append("| {dataset} | {pre} | {n} | {unique} | {dup} | {miss} | {chase} | {danger} | {healthy} | {gate} |".format(
            dataset=r.get("dataset", ""), pre=fmt(r.get("feature_rows_before_start_filter")), n=fmt(r.get("n")),
            unique=fmt(r.get("unique_identity_rows")), dup=fmt(r.get("duplicate_identity_rows")),
            miss=fmt(r.get("missing_identity_rows")), chase=fmt(r.get("chase_candidates")),
            danger=fmt(r.get("dangerous_chase")), healthy=fmt(r.get("healthy_chase")), gate=r.get("classification_evidence_status", "")))
    historical_text = "Historical V2/V3/V4.1 artifacts were not supplied to this replay step; see historical_artifact_summary.csv." if history.empty else history[[c for c in ("version", "artifact_found", "prediction_rows_all", "prediction_rows_from_start", "signal_date_min", "signal_date_max") if c in history.columns]].to_markdown(index=False)
    common_text = "Historical common-window counts unavailable." if history_comp.empty else history_comp.to_markdown(index=False)
    report = f"""# GMQ V4.2.1 — BASE replay and sample-population reconciliation

**Diagnostic only.** One BASE run was replayed. No candidate sizing policies were applied, no thresholds were optimized, and no production/live files were modified.

## Run metadata

- Repository commit: `{meta.get('git_sha', 'unknown')}`
- Requested years: `{meta.get('years_requested')}`
- Evaluation start: `{meta.get('evaluation_start')}`
- Shared BASE warm-up start: `{meta.get('start_full')}`
- Primary transaction cost (`PRIMARY_COST_RT`): `{meta.get('primary_cost_rt')}`
- Simulated days: `{meta.get('days_simulated')}`
- BIST panel: `{meta.get('bist_panel_min')}` to `{meta.get('bist_panel_max')}`
- US panel: `{meta.get('us_panel_min')}` to `{meta.get('us_panel_max')}`
- FX panel: `{meta.get('fx_panel_min')}` to `{meta.get('fx_panel_max')}`

## Same-run population comparison

{chr(10).join(pop_lines)}

Legacy V2/V3 feature construction and V4.1 execution-verified construction were built from the **same BASE state and panels**. This is the cleanest comparison in this report. It does not make the historical V2/V3 artifact itself identical to the current replay; those artifacts came from earlier run commits/data snapshots.

## Execution ledger

- BUY order signals captured: **{len(orders)}**
- BUY fill events linked to a pending order: **{len(fills)}**
- Signals canceled/disappeared without a fill: **{int(orders['status'].eq('cancelled_no_fill').sum()) if not orders.empty else 0}**
- Pending BUY orders at simulation end: **{int(orders['status'].eq('pending_at_end').sum()) if not orders.empty else 0}**
- Unresolved order states: **{int(orders['status'].eq('unresolved').sum()) if not orders.empty else 0}**
- BUY fill events without a known pending order: **{len(orphan_fills)}**
- Fill rows mapped by FIFO when multiple pending orders shared the key: **{int(fills['order_match_ambiguous'].fillna(False).sum()) if not fills.empty else 0}**
- Raw closed trade rows: **{int(stage_map.get('raw_closed_trade_rows', 0))}**
- Closed cycle rows matched to BUY fills by market/ticker/tranche/entry date: **{int(stage_map.get('matched_closed_cycles', 0))}**
- Closed cycle rows with no corresponding BUY fill: **{int(stage_map.get('unmatched_closed_cycles', 0))}**
- Continuation rows labelled `döngü yenilendi (devam)`: **{int(stage_map.get('continuation_cycle_rows', 0))}**

Engine-generated fill events do not carry a durable order ID. If multiple pending orders share market/ticker/tranche in one step, the report flags FIFO inference as ambiguous. This is not represented as an exact identifier match.

## Historical V2/V3/V4.1 artifacts

{historical_text}

{common_text}

Historical archive counts are kept separate from this replay. Differences can also reflect the code commit and data snapshot used by each historical run.

## Classification evidence gate

Every class with fewer than {MIN_CLASS_N} observations is reported as `INCONCLUSIVE_LOW_N`. Meeting this count is not proof of predictive performance. Applied sizing counts are intentionally **not run** in this diagnostic.

## Files

- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_stage_ledger.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_population_comparison.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_order_fill_ledger.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_captured_buy_fills.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_orphan_fills.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_cycle_match_ledger.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_identity_comparison.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_classification_counts.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_policy_triggers.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_artifact_summary.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_common_window.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_vs_current.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_manifest.json`

## Acceptance state

This replay provides a same-run bridge between the legacy and execution-verified definitions. Do not promote any policy. Review unmatched/ambiguous events and the remaining historical artifact differences before making any strategy or production changes.
"""
    (out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_report.md").write_text(report, encoding="utf-8")


def self_test_capture_ledger() -> None:
    """Run the actual in-memory instrumentation against deterministic fake engine modules."""
    import types
    import sys as _sys

    module_names = ("engine", "portfolio", "gmq_momentum_entry_timing_chase_avoidance_v4_1")
    old_modules = {name: _sys.modules.get(name) for name in module_names}

    class FakeMarket:
        def __init__(self, mk: str, dates: pd.DatetimeIndex):
            self.mk = mk
            self.dates = dates
            self.didx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
            self.fc = 0.001

    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    md_b = FakeMarket("bist", dates)
    md_u = FakeMarket("us", dates)
    fx = pd.Series([30.0, 30.1, 30.2], index=dates)

    fake_engine = types.ModuleType("engine")

    def original_step(md, i, st, ctx):
        ev = {"date": str(pd.Timestamp(md.dates[i]).date()), "fills": [], "orders": [], "stops": [], "notes": []}
        prior = list(st.get("pending", []))
        st["pending"] = []
        for order in prior:
            if order.get("side") == "buy" and order.get("t") != "BAD":
                ev["fills"].append({"market": md.mk, "ticker": order["t"], "side": "AL",
                                    "px": 10.0, "qty": float(order.get("amount", 1.0)),
                                    "tranche": order.get("tranche", 0)})
            # BAD orders intentionally disappear without a fill.
        if md.mk == "bist" and i == 0:
            st["pending"].extend([
                {"side": "buy", "t": "AAA", "amount": 100.0, "tranche": 0, "cycle_end": 10},
                {"side": "buy", "t": "BAD", "amount": 50.0, "tranche": 1, "cycle_end": 10},
            ])
        if md.mk == "bist" and i == 1:
            st["pending"].append({"side": "buy", "t": "BBB", "amount": 80.0, "tranche": 2, "cycle_end": 11})
        return ev

    fake_engine.step_market = original_step
    fake_portfolio = types.ModuleType("portfolio")
    fake_portfolio.C = SimpleNamespace(RP_WINDOW=1)
    fake_portfolio.VOLSCALE = {"bist": None, "us": None}
    fake_portfolio.set_hist_rates = lambda _: None
    fake_portfolio.new_state = lambda capital, fx_now: {
        "markets": {"bist": {"pending": []}, "us": {"pending": []}},
        "pf": {"nav_tl": []},
    }
    fake_portfolio.fx_at = lambda series, day: float(series.iloc[0])
    def fake_process_day(state, market, md, index, fx_series, shadow):
        fake_engine.step_market(md, index, state["markets"][market], {})
        state["pf"]["nav_tl"].append([str(pd.Timestamp(md.dates[index]).date()), market, 100000.0, 30.0])
    fake_portfolio.process_day = fake_process_day
    fake_v41 = types.ModuleType("gmq_momentum_entry_timing_chase_avoidance_v4_1")
    def fake_prepare(mdb, mdu, fx_series, cost):
        return mdb.fc, mdu.fc, pd.Series(0.0, index=mdb.dates), pd.Series(0.0, index=mdu.dates)
    fake_v41._prepare_shadow = fake_prepare
    fake_v41._days_for_run = lambda mdb, mdu, start, end: list(dates)
    _sys.modules["engine"] = fake_engine
    _sys.modules["portfolio"] = fake_portfolio
    _sys.modules["gmq_momentum_entry_timing_chase_avoidance_v4_1"] = fake_v41
    try:
        test_run = capture_base_with_ledger(md_b, md_u, fx, dates[0], None, 0.35, 100000.0)
        assert len(test_run["orders"]) == 3, test_run["orders"].to_dict("records")
        assert len(test_run["fills"]) == 2, test_run["fills"].to_dict("records")
        assert test_run["orders"]["status"].eq("cancelled_no_fill").sum() == 1, test_run["orders"].to_dict("records")
        assert test_run["orders"]["status"].eq("filled").sum() == 2, test_run["orders"].to_dict("records")
        assert len(test_run["orphan_fills"]) == 0, test_run["orphan_fills"].to_dict("records")
    finally:
        for name, old in old_modules.items():
            if old is None:
                _sys.modules.pop(name, None)
            else:
                _sys.modules[name] = old


def self_test() -> None:
    self_test_capture_ledger()
    sample = pd.DataFrame([
        {"market": "bist", "ticker": "AAA", "signal_date": "2024-01-02", "entry_date": "2024-01-03", "exit_date": "2024-02-01", "tranche": 0},
        {"market": "bist", "ticker": "AAA", "signal_date": "2024-01-02", "entry_date": "2024-01-03", "exit_date": "2024-02-02", "tranche": 1},
        {"market": "us", "ticker": "BBB", "signal_date": "2024-02-01", "entry_date": "2024-02-02", "exit_date": "2024-02-20", "tranche": 0},
    ])
    four = composite_identity_audit(sample.drop(columns=["tranche", "exit_date"]), ["market", "ticker", "signal_date", "entry_date"])
    five = composite_identity_audit(sample, ["market", "ticker", "signal_date", "entry_date", "tranche"])
    assert four["rows"] == 3 and four["duplicate_identity_rows"] == 1, four
    assert five["unique_identity_rows"] == 3 and five["duplicate_identity_rows"] == 0, five

    cyc = pd.DataFrame([
        {"market": "bist", "ticker": "AAA", "tranche": 0, "entry_date": "2024-01-03", "exit_date": "2024-02-01", "reason": "satış"},
        {"market": "bist", "ticker": "AAA", "tranche": 0, "entry_date": "2024-02-01", "exit_date": "2024-03-01", "reason": "döngü yenilendi (devam)"},
    ])
    filldf = pd.DataFrame([{"execution_id": "buy-1", "market": "bist", "ticker": "AAA", "tranche": 0, "signal_date": "2024-01-02", "entry_date": "2024-01-03", "side": "AL", "px": 10.0, "qty": 100.0}])
    ledger, audit = reconcile_cycles(cyc, filldf)
    assert audit["raw_closed_cycle_rows"] == 2, audit
    assert audit["matched_cycle_rows"] == 1, audit
    assert audit["unmatched_cycle_rows"] == 1, audit
    assert bool(ledger.iloc[1]["continuation_cycle"]), ledger.iloc[1].to_dict()
    sample_classes = count_classes(pd.DataFrame({"chase_candidate": [1] * 16, "dangerous_chase": [1] * 13 + [0] * 3, "healthy_chase": [0] * 13 + [1] * 3}), "SYNTHETIC")
    assert sample_classes["classification_evidence_status"] == "INCONCLUSIVE_LOW_N", sample_classes
    print("SELF-TEST PASS: in-memory order→fill ledger, cancellations, tranche identity, cycle join and low-N gate")


def main() -> None:
    ap = argparse.ArgumentParser(description=VERSION)
    ap.add_argument("--years", type=int, default=DEFAULT_YEARS)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--out", default="research/results_momentum_entry_timing_chase_avoidance_v4_2_replay")
    ap.add_argument("--prior-audit-dir", default="research/results_momentum_entry_timing_chase_avoidance_v4_2_diagnostic")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = norm_date(args.start)
    if pd.isna(start):
        raise ValueError(f"Invalid --start date: {args.start}")

    import config as C
    import data as DA
    import engine as E
    import gmq_early_adverse_predictor as EA
    import gmq_momentum_entry_timing_chase_avoidance_v3 as V3
    import gmq_momentum_entry_timing_chase_avoidance_v4_1 as V41
    import gmq_tail_risk_alpha_conditional_sizing as LEGACY
    import portfolio as P

    years = max(int(args.years), 5)
    panels = {
        "bist": DA.get_panel("bist", years=years, force_full=True),
        "us": DA.get_panel("us", years=years, force_full=True),
    }
    if panels["bist"].empty or panels["us"].empty:
        raise RuntimeError("BIST or US market panel is empty; replay stopped.")
    fx = DA.get_fx(years)
    if fx.empty:
        raise RuntimeError("USD/TRY FX panel is empty; replay stopped.")
    W = {mk: DA.to_wide(panel) for mk, panel in panels.items()}
    md_b = E.MarketData("bist", W["bist"], spec=C.MARKETS["bist"]["spec"])
    md_u = E.MarketData("us", W["us"], spec=C.MARKETS["us"]["spec"])
    warm = max(260, P.C.RP_WINDOW + 5)
    if len(md_b.dates) <= warm or len(md_u.dates) <= warm:
        raise RuntimeError(f"Insufficient panel bars for warm-up index {warm}.")
    start_full = max(pd.Timestamp(md_b.dates[warm]), pd.Timestamp(md_u.dates[warm]))

    cost_rt = float(V41.PRIMARY_COST_RT)
    replay = capture_base_with_ledger(md_b, md_u, fx, start_full, None, cost_rt, C.CAPITAL_TL)
    state = replay["state"]
    raw_cycles = extract_raw_cycles(state)
    cycle_ledger, cycle_counts = reconcile_cycles(raw_cycles, replay["fills"])

    # Re-run the canonical V4.1 join with the same replay evidence. Fail fast if
    # our per-row ledger disagrees with the existing join count.
    verified_cycles = V41.build_verified_execution_dataset(state, replay["fills"].to_dict("records"))
    our_matched = int(cycle_ledger["cycle_match_status"].str.startswith("matched").sum()) if not cycle_ledger.empty else 0
    if our_matched != len(verified_cycles):
        raise RuntimeError(f"Cycle ledger parity check failed: ledger={our_matched}, V4.1 join={len(verified_cycles)}")

    md_by_mk = {"bist": md_b, "us": md_u}
    legacy_raw_features = LEGACY.build_feature_dataset(state, md_by_mk, W)
    legacy_eval, legacy_filter_stats = normalize_and_filter(legacy_raw_features, start)

    verified_feature_rows = EA.build_trade_dataset(
        verified_cycles.drop(columns=["execution_verified", "execution_id"], errors="ignore"), md_by_mk, W
    )
    verified_eval, verified_filter_stats = normalize_and_filter(verified_feature_rows, start)

    # Compute fixed V3 labels on the legacy feature construction and V4.1 labels
    # on the execution-verified construction. This is descriptive, not a policy run.
    legacy_pred = V3.add_adaptive_features(legacy_eval.copy()) if not legacy_eval.empty else legacy_eval.copy()
    verified_pred = V41.add_adaptive_features(verified_eval.copy()) if not verified_eval.empty else verified_eval.copy()
    class_rows = [count_classes(legacy_pred, "LEGACY_V2_V3_FEATURES_USING_V3_CLASSIFIER"),
                  count_classes(verified_pred, "V4_1_EXECUTION_VERIFIED_FEATURES")]
    policy_rows = policy_trigger_rows(legacy_pred, "LEGACY_V2_V3_FEATURES", V3)
    policy_rows += policy_trigger_rows(verified_pred, "V4_1_EXECUTION_VERIFIED_FEATURES", V41)

    legacy_ident = composite_identity_audit(legacy_eval, ["market", "ticker", "signal_date", "entry_date"])
    verified_ident = composite_identity_audit(verified_eval, ["market", "ticker", "signal_date", "entry_date", "tranche"])
    lset_cols = ["market", "ticker", "signal_date", "entry_date"]
    def keyset(df: pd.DataFrame, cols: list[str]) -> set[tuple[Any, ...]]:
        if df.empty or any(c not in df.columns for c in cols): return set()
        z = df[cols].copy()
        for dcol in ("signal_date", "entry_date"):
            z[dcol] = pd.to_datetime(z[dcol], errors="coerce").dt.normalize()
        z = z.dropna()
        return set(map(tuple, z.itertuples(index=False, name=None)))
    legacy_keys = keyset(legacy_eval, lset_cols)
    verified_keys_4 = keyset(verified_eval, lset_cols)
    verified_keys_5 = keyset(verified_eval, lset_cols + ["tranche"])

    population_rows = [
        {"dataset": "LEGACY_V2_V3_FEATURES", "feature_rows_before_start_filter": len(legacy_raw_features),
         "n": len(legacy_eval), **legacy_ident, **{k: v for k, v in count_classes(legacy_pred, "legacy").items() if k in ("chase_candidates", "dangerous_chase", "healthy_chase", "classification_evidence_status")}},
        {"dataset": "V4_1_EXECUTION_VERIFIED_FEATURES", "feature_rows_before_start_filter": len(verified_feature_rows),
         "n": len(verified_eval), **verified_ident, **{k: v for k, v in count_classes(verified_pred, "verified").items() if k in ("chase_candidates", "dangerous_chase", "healthy_chase", "classification_evidence_status")}},
    ]
    populations = pd.DataFrame(population_rows)

    stage_rows = [
        {"stage": "raw_closed_trade_rows", "n": int(len(raw_cycles)), "definition": "state markets/*/trades rows before feature construction"},
        {"stage": "raw_continuation_cycle_rows", "n": int(cycle_counts["continuation_cycle_rows"]), "definition": "raw trade reason contains döngü yenilendi (devam)/continue"},
        {"stage": "unmatched_continuation_cycle_rows", "n": int((cycle_ledger.get("continuation_cycle", pd.Series(dtype=bool)).fillna(False).astype(bool) & cycle_ledger.get("cycle_match_status", pd.Series(dtype=str)).eq("unmatched_no_buy_fill")).sum()) if not cycle_ledger.empty else 0, "definition": "continuation cycle rows that have no matching BUY fill"},
        {"stage": "unmatched_non_continuation_cycle_rows", "n": int((~cycle_ledger.get("continuation_cycle", pd.Series(dtype=bool)).fillna(False).astype(bool) & cycle_ledger.get("cycle_match_status", pd.Series(dtype=str)).eq("unmatched_no_buy_fill")).sum()) if not cycle_ledger.empty else 0, "definition": "unmatched cycle rows not labelled as continuation; review cycle ledger"},
        {"stage": "matched_continuation_cycle_rows", "n": int((cycle_ledger.get("continuation_cycle", pd.Series(dtype=bool)).fillna(False).astype(bool) & cycle_ledger.get("cycle_match_status", pd.Series(dtype=str).str.startswith("matched"))).sum()) if not cycle_ledger.empty else 0, "definition": "continuation cycle rows that nevertheless matched a BUY fill identity"},
        {"stage": "unique_raw_cycle_identities", "n": int(raw_cycles.drop_duplicates([c for c in ["market", "ticker", "tranche", "entry_date", "exit_date"] if c in raw_cycles]).shape[0]) if not raw_cycles.empty else 0, "definition": "market,ticker,tranche,entry_date,exit_date"},
        {"stage": "buy_order_signals_captured", "n": int(len(replay["orders"])), "definition": "new pending BUY orders observed at research step boundary"},
        {"stage": "buy_fill_events_matched_to_pending_order", "n": int(len(replay["fills"])), "definition": "engine BUY fill event paired to a pending order; ambiguous same-key mappings flagged"},
        {"stage": "buy_order_cancelled_without_fill", "n": int(replay["orders"]["status"].eq("cancelled_no_fill").sum()) if not replay["orders"].empty else 0, "definition": "pending BUY disappeared without linked fill"},
        {"stage": "buy_order_pending_at_end", "n": int(replay["orders"]["status"].eq("pending_at_end").sum()) if not replay["orders"].empty else 0, "definition": "still pending at final simulation bar; not counted as zero fill"},
        {"stage": "orphan_buy_fill_events", "n": int(len(replay["orphan_fills"])), "definition": "BUY fill event without known pending BUY order"},
        {"stage": "cycle_rows_matched_to_buy_fill", "n": int(cycle_counts["matched_cycle_rows"]), "definition": "market,ticker,tranche,entry_date queue join; same as V4.1 join count checked"},
        {"stage": "cycle_rows_unmatched_to_buy_fill", "n": int(cycle_counts["unmatched_cycle_rows"]), "definition": "closed trade/cycle row without a matching recorded BUY fill"},
        {"stage": "cycle_rows_missing_identity", "n": int(cycle_counts["cycle_rows_missing_identity"]), "definition": "closed trade/cycle row with missing ticker/tranche/entry date"},
        {"stage": "raw_cycles_not_returned_by_legacy_feature_builder", "n": int(max(0, len(raw_cycles) - len(legacy_raw_features))), "definition": "difference between raw closed trade rows and legacy feature rows; indicates trade normalization/panel/feature exclusions, not necessarily date filtering"},
        {"stage": "matched_cycles_not_returned_by_verified_feature_builder", "n": int(max(0, len(verified_cycles) - len(verified_feature_rows))), "definition": "difference between BUY-fill matched cycles and verified feature rows"},
        {"stage": "legacy_feature_rows_before_date_cleanup", "n": int(legacy_filter_stats["feature_rows_before_date_cleanup"]), "definition": "legacy V2/V3 feature output before invalid date/return cleanup"},
        {"stage": "legacy_feature_rows_dropped_invalid", "n": int(legacy_filter_stats["feature_rows_dropped_invalid_signal_or_return"]), "definition": "legacy feature rows missing signal date or return"},
        {"stage": "legacy_feature_rows_valid_before_start_filter", "n": int(legacy_filter_stats["feature_rows_before_start_filter_valid"]), "definition": "legacy feature rows remaining after date/return cleanup"},
        {"stage": "legacy_feature_rows_removed_before_start", "n": int(legacy_filter_stats["feature_rows_removed_before_start"]), "definition": "valid legacy rows with signal_date earlier than evaluation start"},
        {"stage": "legacy_feature_rows_after_start_filter", "n": int(len(legacy_eval)), "definition": "legacy features after cleanup and signal_date >= evaluation start"},
        {"stage": "verified_feature_rows_before_date_cleanup", "n": int(verified_filter_stats["feature_rows_before_date_cleanup"]), "definition": "V4.1 feature output before invalid date/return cleanup"},
        {"stage": "verified_feature_rows_dropped_invalid", "n": int(verified_filter_stats["feature_rows_dropped_invalid_signal_or_return"]), "definition": "verified feature rows missing signal date or return"},
        {"stage": "verified_feature_rows_valid_before_start_filter", "n": int(verified_filter_stats["feature_rows_before_start_filter_valid"]), "definition": "verified feature rows remaining after date/return cleanup"},
        {"stage": "verified_feature_rows_removed_before_start", "n": int(verified_filter_stats["feature_rows_removed_before_start"]), "definition": "valid verified rows with signal_date earlier than evaluation start"},
        {"stage": "verified_feature_rows_after_start_filter", "n": int(len(verified_eval)), "definition": "execution-verified features after cleanup and signal_date >= evaluation start"},
        {"stage": "legacy_duplicate_four_key_rows", "n": int(legacy_ident["duplicate_identity_rows"]), "definition": "duplicated market,ticker,signal_date,entry_date composite keys; tranche unavailable in legacy features"},
        {"stage": "verified_duplicate_five_key_rows", "n": int(verified_ident["duplicate_identity_rows"]), "definition": "duplicated market,ticker,signal_date,entry_date,tranche composite keys"},
        {"stage": "legacy_verified_four_key_overlap", "n": int(len(legacy_keys & verified_keys_4)), "definition": "sensitivity only; four-key join ignores V4.1 tranche"},
        {"stage": "verified_fills_with_ambiguous_same_key_pending", "n": int(replay["fills"]["order_match_ambiguous"].fillna(False).sum()) if not replay["fills"].empty else 0, "definition": "multiple pending BUY orders shared market,ticker,tranche at fill step"},
        {"stage": "applied_sizing_events", "n": None, "definition": "NOT RUN: diagnostic uses BASE only and applies no candidate policy"},
    ]
    stages = pd.DataFrame(stage_rows)

    identity_rows = [
        {"comparison": "CURRENT_REPLAY_LEGACY_VS_VERIFIED", "legacy_rows": len(legacy_eval), "verified_rows": len(verified_eval),
         "legacy_unique_four_key": len(legacy_keys), "verified_unique_four_key_ignoring_tranche": len(verified_keys_4),
         "verified_unique_five_key": len(verified_keys_5), "four_key_overlap": len(legacy_keys & verified_keys_4),
         "legacy_only_four_key": len(legacy_keys - verified_keys_4), "verified_only_four_key": len(verified_keys_4 - legacy_keys),
         "interpretation": "Same BASE state/panels/warm-up; four-key overlap ignores tranche and is not exact economic identity proof."},
    ]
    identity = pd.DataFrame(identity_rows)

    hist_dir = Path(args.prior_audit_dir)
    historical, historical_common = historical_artifact_summary(hist_dir, start) if hist_dir.exists() else (pd.DataFrame(), pd.DataFrame())
    if not historical.empty:
        historical.to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_artifact_summary.csv", index=False)
        historical_vs_current = historical.copy()
        historical_vs_current["current_same_run_legacy_rows"] = int(len(legacy_eval))
        historical_vs_current["current_same_run_verified_rows"] = int(len(verified_eval))
        if "prediction_rows_from_start" in historical_vs_current.columns:
            historical_vs_current["historical_minus_current_legacy_rows"] = pd.to_numeric(historical_vs_current["prediction_rows_from_start"], errors="coerce") - int(len(legacy_eval))
            historical_vs_current["historical_minus_current_verified_rows"] = pd.to_numeric(historical_vs_current["prediction_rows_from_start"], errors="coerce") - int(len(verified_eval))
        historical_vs_current["interpretation"] = "Counts are comparable in concept, but historic artifacts and this replay may use different commits or data snapshots."
        historical_vs_current.to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_vs_current.csv", index=False)
    else:
        pd.DataFrame([{"note": "Prior V4.2 artifact audit directory not available; run V4.2 archive audit step to populate historical comparison."}]).to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_artifact_summary.csv", index=False)
        pd.DataFrame([{"note": "Historical-vs-current row comparison unavailable."}]).to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_vs_current.csv", index=False)
    if not historical_common.empty:
        historical_common.to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_common_window.csv", index=False)
    else:
        pd.DataFrame([{"note": "Historical common-window comparison unavailable."}]).to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_historical_common_window.csv", index=False)

    replay["orders"].to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_order_fill_ledger.csv", index=False)
    replay["fills"].to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_captured_buy_fills.csv", index=False)
    replay["orphan_fills"].to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_orphan_fills.csv", index=False)
    cycle_ledger.to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_cycle_match_ledger.csv", index=False)
    stages.to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_stage_ledger.csv", index=False)
    populations.to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_population_comparison.csv", index=False)
    identity.to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_identity_comparison.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_classification_counts.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_policy_triggers.csv", index=False)

    source_hashes = {}
    for name in ("config.py", "data.py", "engine.py", "portfolio.py", "gmq_early_adverse_predictor.py",
                 "gmq_tail_risk_alpha_conditional_sizing.py", "gmq_momentum_entry_timing_chase_avoidance_v3.py",
                 "gmq_momentum_entry_timing_chase_avoidance_v4_1.py"):
        p = Path(name)
        if p.exists(): source_hashes[name] = sha256_file(p)
    meta = {
        "name": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local-unknown"),
        "years_requested": years,
        "evaluation_start": start.strftime("%Y-%m-%d"),
        "start_full": pd.Timestamp(start_full).strftime("%Y-%m-%d"),
        "primary_cost_rt": cost_rt,
        "days_simulated": int(replay["days_simulated"]),
        "bist_panel_min": str(pd.Timestamp(md_b.dates[0]).date()),
        "bist_panel_max": str(pd.Timestamp(md_b.dates[-1]).date()),
        "us_panel_min": str(pd.Timestamp(md_u.dates[0]).date()),
        "us_panel_max": str(pd.Timestamp(md_u.dates[-1]).date()),
        "fx_panel_min": str(pd.Timestamp(fx.index.min()).date()),
        "fx_panel_max": str(pd.Timestamp(fx.index.max()).date()),
        "raw_closed_trade_rows": int(len(raw_cycles)),
        "legacy_feature_rows_all_dates": int(len(legacy_raw_features)),
        "legacy_evaluation_rows": int(len(legacy_eval)),
        "verified_feature_rows_all_dates": int(len(verified_feature_rows)),
        "verified_evaluation_rows": int(len(verified_eval)),
        "production_files_modified": False,
        "candidate_sizing_run": False,
        "thresholds_optimized": False,
        "historical_artifacts_replayed": False,
        "current_base_replay_performed": True,
        "module_sha256": source_hashes,
        "limitations": [
            "Historical V2/V3/V4.1 artifacts are from previous commits/data snapshots and are reported separately.",
            "Engine fill events do not carry broker-style durable order ids; same-key collisions are marked ambiguous.",
            "No sizing policy was applied; applied_sizing_events is intentionally not recorded as zero.",
        ],
    }
    (out / "gmq_momentum_entry_timing_chase_avoidance_v4_2_replay_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(out, meta, stages, populations, cycle_ledger, replay["orders"], replay["fills"], replay["orphan_fills"], pd.DataFrame(class_rows), historical, historical_common)

    print(f"=== {VERSION} ===")
    print(populations[["dataset", "feature_rows_before_start_filter", "n", "unique_identity_rows", "duplicate_identity_rows", "chase_candidates", "dangerous_chase", "healthy_chase", "classification_evidence_status"]].to_string(index=False))
    print(stages[["stage", "n"]].to_string(index=False))
    print(f"Outputs: {out.resolve()}")


if __name__ == "__main__":
    main()
