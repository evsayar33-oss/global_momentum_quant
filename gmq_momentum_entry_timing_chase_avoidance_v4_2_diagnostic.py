#!/usr/bin/env python3
"""GMQ Momentum Entry Timing / Chase Avoidance V4.2 — diagnostic-only.

This module only reads prior research artifacts and writes diagnostic reports.
It does not import or modify engine.py, portfolio.py, config.py, signals.py,
or live state/order/NAV files. It does not run sizing policies or optimize
thresholds. Missing evidence is reported as unavailable, never as zero.

Artifact inputs are expected to be ZIPs created by prior V2, V3 and V4.1
GitHub Actions workflows, or extracted result directories.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import pandas as pd

VERSION_SPECS = {
    "V2": {
        "workflow": "gmq_momentum_entry_timing_chase_avoidance_v2.yml",
        "artifact_hint": "gmq-momentum-entry-timing-chase-avoidance-v2-results",
        "predictions_suffix": "_v2_predictions.csv",
        "population_suffix": "_v2_population_audit.csv",
        "execution_suffix": "_v2_execution_audit.csv",
        "classification_suffix": "_v2_classification.csv",
        "policy_suffix": "_v2_policy_usage.csv",
        "trades_suffix": "_v2_trades.csv",
        "stress_suffix": "_v2_stress.csv",
    },
    "V3": {
        "workflow": "gmq_momentum_entry_timing_chase_avoidance_v3.yml",
        "artifact_hint": "gmq-momentum-entry-timing-chase-avoidance-v3-results",
        "predictions_suffix": "_v3_predictions.csv",
        "population_suffix": "_v3_population_audit.csv",
        "execution_suffix": "_v3_execution_audit.csv",
        "classification_suffix": "_v3_classification.csv",
        "policy_suffix": "_v3_policy_usage.csv",
        "trades_suffix": "_v3_trades.csv",
        "stress_suffix": "_v3_stress.csv",
    },
    "V4.1": {
        "workflow": "gmq_momentum_entry_timing_chase_avoidance_v4_1.yml",
        "artifact_hint": "gmq-momentum-entry-timing-chase-avoidance-v4-1-results",
        "predictions_suffix": "_v4_1_predictions.csv",
        "population_suffix": "_v4_1_population_audit.csv",
        "execution_suffix": "_v4_1_execution_audit.csv",
        "classification_suffix": "_v4_1_classification.csv",
        "policy_suffix": "_v4_1_policy_usage.csv",
        "trades_suffix": "_v4_1_trades.csv",
        "stress_suffix": "_v4_1_stress.csv",
    },
}
API_ROOT = "https://api.github.com"
IDENTITY_PREFERENCE = ["market", "ticker", "signal_date", "entry_date", "tranche"]
MIN_CLASS_N = 30  # diagnostic evidence floor; it is not an optimization threshold.


@dataclass
class ArtifactSource:
    version: str
    found: bool
    source: str = ""
    run_id: str = ""
    run_number: str = ""
    created_at: str = ""
    head_sha: str = ""
    artifact_name: str = ""
    note: str = ""


def truthy_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    normalized = s.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def numeric_count(df: pd.DataFrame, col: str) -> int | None:
    if col not in df.columns:
        return None
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def safe_extract(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP without allowing path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            raw = info.filename.replace("\\", "/")
            pure = PurePosixPath(raw)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"Unsafe ZIP member path rejected: {info.filename!r}")
            if not raw or info.is_dir():
                continue
            target = destination.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def api_get_json(url: str, token: str) -> dict[str, Any]:
    import requests
    response = requests.get(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "GMQ-V4.2-Diagnostic-Only",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_latest_successful_artifact(repo: str, token: str, version: str,
                                    download_root: Path) -> ArtifactSource:
    """Download an artifact from the newest successful run of a V2/V3/V4.1 workflow."""
    import requests
    spec = VERSION_SPECS[version]
    url = f"{API_ROOT}/repos/{repo}/actions/workflows/{spec['workflow']}/runs?status=success&per_page=20"
    try:
        listing = api_get_json(url, token)
    except Exception as exc:
        return ArtifactSource(version, False, note=f"Workflow run list could not be read: {exc}")

    runs = listing.get("workflow_runs", [])
    runs = [r for r in runs if r.get("status") == "completed" and r.get("conclusion") == "success"]
    runs.sort(key=lambda r: (r.get("updated_at") or "", int(r.get("run_number") or 0)), reverse=True)
    if not runs:
        return ArtifactSource(version, False, note="No successful completed workflow run was found.")

    last_error = ""
    for run in runs:
        rid = run.get("id")
        try:
            artifact_json = api_get_json(f"{API_ROOT}/repos/{repo}/actions/runs/{rid}/artifacts?per_page=100", token)
            artifacts = [a for a in artifact_json.get("artifacts", [])
                         if not a.get("expired") and int(a.get("size_in_bytes") or 0) > 0]
            if not artifacts:
                continue
            hint = spec["artifact_hint"].lower()
            hinted = [a for a in artifacts if hint in str(a.get("name", "")).lower()]
            selected = hinted or artifacts
            selected.sort(key=lambda a: (a.get("created_at") or "", int(a.get("id") or 0)), reverse=True)
            artifact = selected[0]
            download_url = f"{API_ROOT}/repos/{repo}/actions/artifacts/{artifact['id']}/zip"
            response = requests.get(
                download_url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                    "X-GitHub-Api-Version": "2022-11-28",
                    "User-Agent": "GMQ-V4.2-Diagnostic-Only",
                },
                timeout=180,
                allow_redirects=True,
            )
            response.raise_for_status()
            if not response.content.startswith(b"PK\x03\x04"):
                raise ValueError("Artifact download response is not a ZIP file.")
            download_root.mkdir(parents=True, exist_ok=True)
            out_zip = download_root / f"{version.lower().replace('.', '_')}_latest_artifact.zip"
            out_zip.write_bytes(response.content)
            extract_dir = download_root / f"{version.lower().replace('.', '_')}_extracted"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            safe_extract(out_zip, extract_dir)
            return ArtifactSource(
                version=version, found=True, source=str(out_zip),
                run_id=str(rid), run_number=str(run.get("run_number", "")),
                created_at=str(run.get("created_at", "")),
                head_sha=str(run.get("head_sha", "")),
                artifact_name=str(artifact.get("name", "")),
                note="Downloaded from newest successful completed workflow run.",
            )
        except Exception as exc:
            last_error = f"Run {rid}: {exc}"
    return ArtifactSource(version, False, note=(last_error or "No unexpired artifact was found on the recent successful runs."))


def select_file(root: Path, suffix: str) -> Path | None:
    matches = [p for p in root.rglob("*.csv") if p.name.lower().endswith(suffix.lower())]
    if not matches:
        # Accept the uploaded V4.1 output ZIP and small renamed test fixtures.
        generic_map = {
            "_v2_predictions.csv": ["predictions.csv", "v2_predictions.csv"],
            "_v3_predictions.csv": ["predictions.csv", "v3_predictions.csv"],
            "_v4_1_predictions.csv": ["predictions.csv", "v4_1_predictions.csv", "v4.1_predictions.csv"],
        }
        for base in generic_map.get(suffix, []):
            matches = [p for p in root.rglob("*.csv") if p.name.lower() == base]
            if matches:
                break
    if not matches:
        return None
    def score(p: Path) -> tuple[int, int, str]:
        parts = [x.lower() for x in p.parts]
        result_score = 3 if any("results_" in x or x.startswith("results") for x in parts) else 0
        smoke_score = -5 if any("smoke" in x for x in parts) else 0
        name_score = 1 if "predictions" in p.name.lower() else 0
        return result_score + smoke_score + name_score, p.stat().st_size, str(p)
    return sorted(matches, key=score, reverse=True)[0]


def read_csv_if_found(root: Path, suffix: str) -> tuple[pd.DataFrame | None, Path | None]:
    path = select_file(root, suffix)
    if path is None:
        return None, None
    try:
        return pd.read_csv(path, low_memory=False), path
    except Exception as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc


def normalize_id_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = [c for c in IDENTITY_PREFERENCE if c in df.columns]
    out = df.copy()
    for c in ("signal_date", "entry_date"):
        if c in out.columns:
            parsed = pd.to_datetime(out[c], errors="coerce", utc=False)
            out[c] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), "")
    for c in ("market", "ticker"):
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str).str.strip().str.lower()
    if "tranche" in out.columns:
        out["tranche"] = out["tranche"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    return out, cols


def identity_audit(df: pd.DataFrame) -> dict[str, Any]:
    normalized, key_cols = normalize_id_frame(df)
    row_n = int(len(df))
    if len(key_cols) < 3:
        return {
            "row_count": row_n, "identity_key": ",".join(key_cols),
            "unique_identity_count": None, "duplicate_identity_rows": None,
            "missing_identity_rows": None,
            "note": "Not enough identity columns to establish a robust composite identity.",
        }
    incomplete = pd.Series(False, index=normalized.index)
    for col in key_cols:
        incomplete |= normalized[col].isna() | normalized[col].astype(str).str.strip().eq("")
    complete = normalized.loc[~incomplete, key_cols]
    unique_n = int(complete.drop_duplicates().shape[0])
    dup_rows = int(complete.duplicated(key_cols, keep="first").sum())
    return {
        "row_count": row_n,
        "identity_key": ",".join(key_cols),
        "unique_identity_count": unique_n,
        "duplicate_identity_rows": dup_rows,
        "missing_identity_rows": int(incomplete.sum()),
        "note": "" if len(key_cols) == len(IDENTITY_PREFERENCE) else "Identity is partial because one or more preferred key columns are absent.",
    }


def unique_identity_set(df: pd.DataFrame) -> tuple[set[tuple[str, ...]], list[str]]:
    normalized, key_cols = normalize_id_frame(df)
    if len(key_cols) < 3:
        return set(), key_cols
    complete = normalized[key_cols].copy()
    for col in key_cols:
        complete = complete[complete[col].notna() & complete[col].astype(str).str.strip().ne("")]
    return {tuple(map(str, row)) for row in complete.drop_duplicates().itertuples(index=False, name=None)}, key_cols


def population_map(population: pd.DataFrame | None) -> dict[str, int]:
    if population is None or population.empty or not {"stage", "n"}.issubset(population.columns):
        return {}
    out: dict[str, int] = {}
    for _, row in population.iterrows():
        try:
            out[str(row["stage"])] = int(float(row["n"]))
        except (TypeError, ValueError):
            continue
    return out


def candidate_counts(pred: pd.DataFrame | None, classification: pd.DataFrame | None,
                     population: dict[str, int]) -> dict[str, int | None]:
    output: dict[str, int | None] = {"chase_candidates": None, "dangerous_chase": None, "healthy_chase": None}
    aliases = {
        "chase_candidates": ("chase_candidate", "flag_chase_risk_020"),
        "dangerous_chase": ("dangerous_chase",),
        "healthy_chase": ("healthy_chase",),
    }
    if pred is not None:
        for target, col_names in aliases.items():
            col = next((c for c in col_names if c in pred.columns), None)
            if col is not None:
                output[target] = int(truthy_series(pred[col]).sum())
        if output["chase_candidates"] is None and "chase_risk" in pred.columns:
            output["chase_candidates"] = int((pd.to_numeric(pred["chase_risk"], errors="coerce") >= 0.20).sum())
    if classification is not None and {"class", "n"}.issubset(classification.columns):
        class_map = {str(r["class"]).strip().upper(): int(r["n"]) for _, r in classification.iterrows()
                     if pd.notna(r.get("n")) and str(r.get("n")).replace(".", "", 1).isdigit()}
        if output["chase_candidates"] is None:
            output["chase_candidates"] = class_map.get("ALL_CHASE")
        if output["dangerous_chase"] is None:
            output["dangerous_chase"] = class_map.get("DANGEROUS_CHASE")
        if output["healthy_chase"] is None:
            output["healthy_chase"] = class_map.get("HEALTHY_CHASE")
    stage_aliases = {
        "chase_candidates": "chase_candidates",
        "dangerous_chase": "dangerous_chase",
        "healthy_chase": "healthy_chase",
    }
    for target, stage in stage_aliases.items():
        if output[target] is None and stage in population:
            output[target] = population[stage]
    return output


def sample_status(dangerous_n: int | None, healthy_n: int | None) -> str:
    if dangerous_n is None or healthy_n is None:
        return "NOT_ASSESSABLE_MISSING_COUNTS"
    if dangerous_n < MIN_CLASS_N or healthy_n < MIN_CLASS_N:
        return f"INCONCLUSIVE_LOW_N (need >= {MIN_CLASS_N} in each class)"
    return "SAMPLE_FLOOR_MET; predictive validity still not established"


def audit_version(version: str, root: Path, source: ArtifactSource | None = None) -> dict[str, Any]:
    spec = VERSION_SPECS[version]
    suffix_map = {
        "predictions": spec["predictions_suffix"],
        "population": spec["population_suffix"],
        "execution": spec["execution_suffix"],
        "classification": spec["classification_suffix"],
        "policy": spec["policy_suffix"],
        "trades": spec["trades_suffix"],
        "stress": spec["stress_suffix"],
    }
    loaded: dict[str, pd.DataFrame | None] = {}
    paths: dict[str, str] = {}
    for kind, suffix in suffix_map.items():
        df, path = read_csv_if_found(root, suffix)
        loaded[kind] = df
        if path:
            paths[kind] = str(path)
    pred = loaded["predictions"]
    population = loaded["population"]
    execution = loaded["execution"]
    classification = loaded["classification"]
    policy = loaded["policy"]
    trades = loaded["trades"]
    stress = loaded["stress"]
    pmap = population_map(population)
    id_audit = identity_audit(pred) if pred is not None else {
        "row_count": None, "identity_key": "", "unique_identity_count": None,
        "duplicate_identity_rows": None, "missing_identity_rows": None,
        "note": "Predictions file is missing; version population cannot be counted.",
    }
    counts = candidate_counts(pred, classification, pmap)
    if pred is not None and "execution_verified" in pred.columns:
        verified_rows: int | None = int(truthy_series(pred["execution_verified"]).sum())
    else:
        verified_rows = None
    if pred is not None:
        signal_min = str(pd.to_datetime(pred["signal_date"], errors="coerce").min().date()) if "signal_date" in pred.columns and pd.to_datetime(pred["signal_date"], errors="coerce").notna().any() else ""
        signal_max = str(pd.to_datetime(pred["signal_date"], errors="coerce").max().date()) if "signal_date" in pred.columns and pd.to_datetime(pred["signal_date"], errors="coerce").notna().any() else ""
    else:
        signal_min = signal_max = ""
    # Counts of execution-matching errors cannot be inferred from policy-sizing
    # audit rows. V4.1's delivered archive contains no order/fill-level ledger.
    raw_cycles = pmap.get("actual_verified_closed_cycles")
    if raw_cycles is None:
        raw_cycles = pmap.get("raw_closed_trades")
    feature_before_start = pmap.get("helper_feature_dataset_before_start_filter")
    feature_eval = pmap.get("feature_rows_in_evaluation_window")
    if feature_eval is None and pred is not None:
        feature_eval = len(pred)
    if feature_before_start is None and pred is not None and pmap.get("feature_rows_in_evaluation_window") is None:
        feature_before_start = None  # must not pretend the already-filtered file is pre-filter data.
    if feature_eval is None:
        feature_eval = int(id_audit["row_count"]) if id_audit["row_count"] is not None else None

    event_records: list[dict[str, Any]] = []
    if execution is not None and not execution.empty:
        for _, row in execution.iterrows():
            rec = {"version": version, "policy": str(row.get("policy", "")),
                   "expected_classified_count": _int_or_none(row.get("expected_classified_count")),
                   "applied_sizing_count": _int_or_none(row.get("applied_sizing_count")),
                   "apply_match_rate_pct": _float_or_none(row.get("apply_match_rate_pct")),
                   "candidate_universe_n": _int_or_none(row.get("candidate_universe_n")),
                   "dangerous_n": _int_or_none(row.get("dangerous_n")),
                   "healthy_n": _int_or_none(row.get("healthy_n")),
                   "event_evidence": "execution_audit.csv; sizing-application audit, not raw order-to-fill reconciliation"}
            event_records.append(rec)
    elif policy is not None and not policy.empty:
        for _, row in policy.iterrows():
            event_records.append({
                "version": version, "policy": str(row.get("policy", "")),
                "expected_classified_count": _int_or_none(row.get("condition_n")),
                "applied_sizing_count": None,
                "apply_match_rate_pct": None,
                "candidate_universe_n": _int_or_none(row.get("n")),
                "dangerous_n": None, "healthy_n": None,
                "event_evidence": "policy_usage.csv; condition triggers only; applied events are not recorded here",
            })

    stress_audit = "not recorded"
    stress_levels = None
    if stress is not None and not stress.empty:
        has_pair = {"cost_rt_pct", "base_cagr_pct", "cand_cagr_pct"}.issubset(stress.columns)
        if has_pair:
            stress_levels = int(stress["cost_rt_pct"].nunique(dropna=True))
            costs = pd.to_numeric(stress["cost_rt_pct"], errors="coerce")
            cagr_delta = pd.to_numeric(stress.get("cagr_delta_vs_base_pp"), errors="coerce") if "cagr_delta_vs_base_pp" in stress.columns else pd.Series(dtype=float)
            stress_audit = (
                f"summary has BASE/candidate CAGR on {stress_levels} cost level(s); "
                f"same evaluation window/warm-up cannot be confirmed from this CSV alone"
            )
            if len(cagr_delta.dropna()) == len(stress) and (costs.notna().all()):
                stress_audit += f"; candidate CAGR delta min={cagr_delta.min():.6f} pp, max={cagr_delta.max():.6f} pp"

    row = {
        "version": version,
        "artifact_found": bool(pred is not None),
        "source_kind": "GitHub Actions artifact" if source and source.found else ("local input" if pred is not None else "unavailable"),
        "source_run_id": source.run_id if source else "",
        "source_run_number": source.run_number if source else "",
        "source_created_at": source.created_at if source else "",
        "source_head_sha": source.head_sha if source else "",
        "predictions_rows_evaluation_window": _int_or_none(id_audit.get("row_count")),
        "raw_closed_cycles_reported": raw_cycles,
        "feature_rows_before_start_filter_reported": feature_before_start,
        "feature_rows_in_evaluation_window": feature_eval,
        "unique_identity_rows": id_audit.get("unique_identity_count"),
        "duplicate_identity_rows": id_audit.get("duplicate_identity_rows"),
        "missing_identity_rows": id_audit.get("missing_identity_rows"),
        "identity_key": id_audit.get("identity_key", ""),
        "signal_date_min": signal_min,
        "signal_date_max": signal_max,
        "execution_verified_prediction_rows": verified_rows,
        "order_fill_cycle_unmatched_count": None,
        "order_fill_cycle_duplicate_match_count": None,
        "order_fill_cycle_audit_note": "Not present in this artifact. A sizing execution_audit is not equivalent to a raw order→fill→closed-cycle ledger.",
        "chase_candidates": counts["chase_candidates"],
        "dangerous_chase": counts["dangerous_chase"],
        "healthy_chase": counts["healthy_chase"],
        "classification_evidence_status": sample_status(counts["dangerous_chase"], counts["healthy_chase"]),
        "policy_usage_rows": int(len(policy)) if policy is not None else None,
        "trades_file_rows_including_policy_repeats": int(len(trades)) if trades is not None else None,
        "execution_audit_rows": int(len(execution)) if execution is not None else None,
        "stress_cost_levels": stress_levels,
        "stress_pairing_audit": stress_audit,
        "file_paths": json.dumps(paths, ensure_ascii=False),
        "source_note": source.note if source else "",
    }
    return {"summary": row, "predictions": pred, "population": population, "execution": execution,
            "classification": classification, "policy": policy, "trades": trades, "stress": stress,
            "events": event_records, "identity_audit": id_audit, "population_map": pmap,
            "prediction_path": paths.get("predictions", "")}


def _int_or_none(value: Any) -> int | None:
    try:
        if pd.isna(value) or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if pd.isna(value) or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_identities(audit_a: dict[str, Any], audit_b: dict[str, Any], version_a: str, version_b: str) -> dict[str, Any]:
    df_a = audit_a.get("predictions")
    df_b = audit_b.get("predictions")
    result: dict[str, Any] = {
        "version_a": version_a, "version_b": version_b,
        "identity_overlap_n": None, "a_only_n": None, "b_only_n": None,
        "a_unique_identity_n": None, "b_unique_identity_n": None,
        "a_identity_key": "", "b_identity_key": "", "comparability": "NOT_COMPARABLE",
        "note": "Both versions need prediction-level composite identities.",
    }
    if df_a is None or df_b is None:
        return result
    set_a, cols_a = unique_identity_set(df_a)
    set_b, cols_b = unique_identity_set(df_b)
    result["a_identity_key"] = ",".join(cols_a)
    result["b_identity_key"] = ",".join(cols_b)
    result["a_unique_identity_n"] = len(set_a)
    result["b_unique_identity_n"] = len(set_b)
    if not set_a or not set_b:
        result["note"] = "Unable to form composite identity sets from one or both versions."
        return result
    if cols_a != cols_b:
        result["comparability"] = "PARTIAL_KEY_ONLY"
        result["note"] = "Versions expose different identity columns; exact identity comparison is not valid."
        return result
    overlap = set_a & set_b
    result.update({
        "identity_overlap_n": len(overlap), "a_only_n": len(set_a - set_b), "b_only_n": len(set_b - set_a),
        "comparability": "EXACT_COMPOSITE_KEY_SET",
        "note": "Exact normalized composite identity sets compared; differing trade filters may still make the samples intentionally different.",
    })
    return result


def write_report(out_dir: Path, audits: dict[str, dict[str, Any]], sources: dict[str, ArtifactSource]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [audits[v]["summary"] for v in VERSION_SPECS if v in audits]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v4_2_population_summary.csv", index=False)

    comparisons = []
    for a, b in (("V2", "V3"), ("V2", "V4.1"), ("V3", "V4.1")):
        if a in audits and b in audits:
            comparisons.append(compare_identities(audits[a], audits[b], a, b))
    pd.DataFrame(comparisons).to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v4_2_identity_comparison.csv", index=False)

    events = [r for v in audits for r in audits[v]["events"]]
    pd.DataFrame(events, columns=[
        "version", "policy", "expected_classified_count", "applied_sizing_count", "apply_match_rate_pct",
        "candidate_universe_n", "dangerous_n", "healthy_n", "event_evidence"
    ]).to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v4_2_policy_events.csv", index=False)

    source_rows = [asdict(sources[v]) for v in VERSION_SPECS if v in sources]
    pd.DataFrame(source_rows).to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v4_2_artifact_sources.csv", index=False)

    # A compact stage ledger separates known counts from evidence that is absent.
    stage_rows: list[dict[str, Any]] = []
    for v, audit in audits.items():
        sm = audit["summary"]
        pmap = audit["population_map"]
        for stage, count, evidence in [
            ("raw_closed_cycles_or_trades", sm.get("raw_closed_cycles_reported"), "population_audit.csv if present; otherwise not exported"),
            ("feature_rows_before_start_filter", sm.get("feature_rows_before_start_filter_reported"), "population_audit.csv if present; otherwise not exported"),
            ("feature_rows_in_evaluation_window", sm.get("feature_rows_in_evaluation_window"), "prediction row count if available"),
            ("unique_composite_identities", sm.get("unique_identity_rows"), "prediction identity audit"),
            ("duplicate_composite_identity_rows", sm.get("duplicate_identity_rows"), "prediction identity audit"),
            ("missing_identity_rows", sm.get("missing_identity_rows"), "prediction identity audit"),
            ("execution_verified_prediction_rows", sm.get("execution_verified_prediction_rows"), "execution_verified column if present"),
            ("unmatched_order_fill_cycle_matches", None, "not present in packaged V4.1 archive; never imputed as zero"),
            ("duplicate_order_fill_cycle_matches", None, "not present in packaged V4.1 archive; never imputed as zero"),
            ("chase_candidates", sm.get("chase_candidates"), "prediction-level classifications"),
            ("dangerous_chase", sm.get("dangerous_chase"), "prediction-level classifications"),
            ("healthy_chase", sm.get("healthy_chase"), "prediction-level classifications"),
            ("applied_sizing_events", None, "see policy_events.csv; event counts are policy-specific, not additive across policies"),
        ]:
            stage_rows.append({"version": v, "stage": stage, "n": count, "evidence": evidence})
    pd.DataFrame(stage_rows).to_csv(out_dir / "gmq_momentum_entry_timing_chase_avoidance_v4_2_stage_ledger.csv", index=False)

    lines = [
        "# GMQ Momentum Entry Timing / Chase Avoidance V4.2 — Diagnostic Report",
        "",
        "**DIAGNOSTIC-ONLY. No production source files changed. No thresholds were optimized.**",
        "",
        "## Executive status",
        "",
    ]
    available = [v for v in VERSION_SPECS if v in audits and audits[v]["summary"]["artifact_found"]]
    missing = [v for v in VERSION_SPECS if v not in audits or not audits[v]["summary"]["artifact_found"]]
    lines.append(f"Available prediction artifacts: {', '.join(available) if available else 'none'}.")
    if missing:
        lines.append(f"Missing prediction artifacts: {', '.join(missing)}. Exact V2–V4.1 sample reconciliation is incomplete until these historical artifacts are available.")
    lines.extend([
        "",
        "A sizing application match rate is not the same as a complete raw BUY order → fill → closed-cycle match audit. The reports preserve unknown unmatched/duplicate order-fill-cycle counts as blank, because the packaged V4.1 output does not contain a record-level execution ledger.",
        "",
        "## Population counts",
        "",
        "| Version | Raw closed cycles reported | Feature rows before start filter | Evaluation prediction rows | Unique identities | Duplicate identity rows | Missing identity rows | Chase | Dangerous | Healthy | Classification evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for v in VERSION_SPECS:
        if v not in audits:
            continue
        s = audits[v]["summary"]
        def fmt(x: Any) -> str:
            return "not recorded" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
        lines.append(
            f"| {v} | {fmt(s.get('raw_closed_cycles_reported'))} | {fmt(s.get('feature_rows_before_start_filter_reported'))} | "
            f"{fmt(s.get('feature_rows_in_evaluation_window'))} | {fmt(s.get('unique_identity_rows'))} | "
            f"{fmt(s.get('duplicate_identity_rows'))} | {fmt(s.get('missing_identity_rows'))} | {fmt(s.get('chase_candidates'))} | "
            f"{fmt(s.get('dangerous_chase'))} | {fmt(s.get('healthy_chase'))} | {fmt(s.get('classification_evidence_status'))} |"
        )
    lines.extend([
        "",
        "## Version identity reconciliation",
        "",
        "See `gmq_momentum_entry_timing_chase_avoidance_v4_2_identity_comparison.csv`. Exact composite keys are compared only when both versions expose the same identity columns. Row-count deltas alone are not treated as proof of why the sample changed.",
        "",
        "## Execution matching evidence",
        "",
        "- `execution_audit.csv` rows describe expected versus applied sizing events by policy when that file exists.",
        "- Those counts do not provide a count of missing fills, duplicate fills, ambiguous FIFO matches, or extra fills left unmatched.",
        "- The V4.1 ZIP currently analyzed contains 4,994 reported execution-verified closed cycles, 4,994 feature rows before the start-date filter, and 3,432 prediction rows after the filter. The ZIP does not contain raw order/fill records to independently verify every join.",
        "- Exact composite identities in the V4.1 prediction CSV are audited using market, ticker, signal_date, entry_date and tranche.",
        "",
        "## Classification evidence gate",
        "",
        f"A class sample floor of {MIN_CLASS_N} observations per class is used only as a diagnostic sufficiency warning. Meeting this floor would not by itself prove predictive validity. A class below the floor is marked INCONCLUSIVE, never PASS.",
        "",
        "## Cost-stress audit",
        "",
        "See `population_summary.csv` for per-version stress evidence. A summarized stress CSV can confirm the reported cost levels and paired BASE/candidate metrics. It cannot, on its own, prove identical hidden warm-up state and evaluation dates unless those fields are explicitly recorded.",
        "",
        "## Scope and limitations",
        "",
        "1. This tool reads prior result artifacts; it does not re-run strategy variants or modify their code.",
        "2. If V2/V3 artifact files are unavailable or generated from different commits/parameters, the report will state that exact comparison is incomplete or caution that the samples may not be comparable.",
        "3. Raw closed-cycle count, pre-start row count, order/fill match failures and applied events are never inferred from unrelated files.",
        "4. No profitability or production-promotion claim is made.",
        "",
        "## Outputs",
        "",
        "- `gmq_momentum_entry_timing_chase_avoidance_v4_2_population_summary.csv`",
        "- `gmq_momentum_entry_timing_chase_avoidance_v4_2_stage_ledger.csv`",
        "- `gmq_momentum_entry_timing_chase_avoidance_v4_2_identity_comparison.csv`",
        "- `gmq_momentum_entry_timing_chase_avoidance_v4_2_policy_events.csv`",
        "- `gmq_momentum_entry_timing_chase_avoidance_v4_2_artifact_sources.csv`",
        "",
    ])
    (out_dir / "gmq_momentum_entry_timing_chase_avoidance_v4_2_report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "name": "GMQ V4.2 diagnostic-only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "available_versions": available,
        "missing_versions": missing,
        "minimum_class_n_diagnostic_floor": MIN_CLASS_N,
        "production_files_modified": False,
        "thresholds_optimized": False,
        "historical_backtest_run": False,
        "warnings": [
            "Missing order/fill/cycle counts are not inferred as zero.",
            "V2/V3 sample comparison remains incomplete when their result artifacts are absent.",
            "Archive summary alone cannot prove hidden warm-up/evaluation-window parity.",
        ],
    }
    (out_dir / "gmq_momentum_entry_timing_chase_avoidance_v4_2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def source_root_for(version: str, args: argparse.Namespace, work_root: Path,
                    sources: dict[str, ArtifactSource]) -> Path | None:
    explicit = getattr(args, {"V2": "v2_zip", "V3": "v3_zip", "V4.1": "v4_1_zip"}[version], None)
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Input ZIP/path does not exist: {candidate}")
        if candidate.is_file() and zipfile.is_zipfile(candidate):
            out = work_root / f"local_{version.lower().replace('.', '_')}"
            if out.exists():
                shutil.rmtree(out)
            safe_extract(candidate, out)
            return out
        if candidate.is_dir():
            return candidate
        raise ValueError(f"Input is neither a ZIP nor a directory: {candidate}")
    if sources.get(version) and sources[version].found:
        extracted = work_root / f"{version.lower().replace('.', '_')}_extracted"
        return extracted if extracted.exists() else None
    if args.input_root:
        candidate = Path(args.input_root).expanduser().resolve()
        if candidate.exists():
            return candidate
    return None


def run_self_test() -> None:
    base = pd.DataFrame([
        {"market": "bist", "ticker": "AAA", "signal_date": "2024-01-01", "entry_date": "2024-01-02", "tranche": 1},
        {"market": "bist", "ticker": "AAA", "signal_date": "2024-01-01", "entry_date": "2024-01-02", "tranche": 1},
        {"market": "us", "ticker": "BBB", "signal_date": "2024-01-03", "entry_date": "2024-01-04", "tranche": 2},
        {"market": "us", "ticker": "", "signal_date": "2024-01-05", "entry_date": "2024-01-06", "tranche": 3},
    ])
    audit = identity_audit(base)
    assert audit["row_count"] == 4, audit
    assert audit["unique_identity_count"] == 2, audit
    assert audit["duplicate_identity_rows"] == 1, audit
    assert audit["missing_identity_rows"] == 1, audit
    assert sample_status(13, 3).startswith("INCONCLUSIVE_LOW_N"), sample_status(13, 3)
    assert sample_status(40, 40).startswith("SAMPLE_FLOOR_MET"), sample_status(40, 40)
    with tempfile.TemporaryDirectory(prefix="gmq_v42_test_") as td:
        root = Path(td)
        bad = root / "bad.zip"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("../escape.txt", "bad")
        try:
            safe_extract(bad, root / "extract")
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe ZIP path was not rejected")
    print("SELF-TEST PASS: identity duplicates/missing keys, low-n classification gate, and ZIP path traversal rejection.")


def main() -> int:
    parser = argparse.ArgumentParser(description="GMQ V4.2 diagnostic-only artifact auditor")
    parser.add_argument("--v2-zip", help="Optional V2 result ZIP or extracted directory")
    parser.add_argument("--v3-zip", help="Optional V3 result ZIP or extracted directory")
    parser.add_argument("--v4-1-zip", dest="v4_1_zip", help="Optional V4.1 result ZIP or extracted directory")
    parser.add_argument("--input-root", help="Directory containing result artifact folders")
    parser.add_argument("--out", default="research/results_momentum_entry_timing_chase_avoidance_v4_2_diagnostic")
    parser.add_argument("--fetch-latest-artifacts", action="store_true", help="Read latest successful V2/V3/V4.1 Actions artifacts using GH_TOKEN")
    parser.add_argument("--self-test", action="store_true", help="Run deterministic synthetic unit smoke tests")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return 0

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = out_dir / "_diagnostic_inputs"
    work_root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, ArtifactSource] = {}

    if args.fetch_latest_artifacts:
        repo = os.getenv("GITHUB_REPOSITORY", "").strip()
        token = os.getenv("GH_TOKEN", "").strip()
        if not repo or not token:
            print("WARNING: --fetch-latest-artifacts set but GITHUB_REPOSITORY/GH_TOKEN is missing; local inputs only will be used.", file=sys.stderr)
        else:
            for version in VERSION_SPECS:
                explicit = getattr(args, {"V2": "v2_zip", "V3": "v3_zip", "V4.1": "v4_1_zip"}[version], None)
                if explicit:
                    sources[version] = ArtifactSource(version, True, source=str(explicit), note="Explicit local input takes precedence over remote artifacts.")
                    continue
                sources[version] = fetch_latest_successful_artifact(repo, token, version, work_root)
                if not sources[version].found:
                    print(f"WARNING: {version} artifact unavailable: {sources[version].note}", file=sys.stderr)

    audits: dict[str, dict[str, Any]] = {}
    for version in VERSION_SPECS:
        root = source_root_for(version, args, work_root, sources)
        if root is None:
            src = sources.get(version) or ArtifactSource(version, False, note="No local result archive/directory supplied.")
            audits[version] = audit_version(version, work_root / f"missing_{version.lower().replace('.', '_')}", src)
            audits[version]["summary"]["artifact_found"] = False
            continue
        audits[version] = audit_version(version, root, sources.get(version))
        if not audits[version]["summary"]["artifact_found"]:
            audits[version]["summary"]["source_note"] = "Input located, but prediction CSV was not found in that ZIP/directory."
            print(f"WARNING: prediction CSV for {version} was not found under {root}", file=sys.stderr)

    write_report(out_dir, audits, sources)
    print(f"V4.2 diagnostic report written: {out_dir}")
    for version in VERSION_SPECS:
        if version in audits:
            s = audits[version]["summary"]
            print(f"{version}: rows={s.get('predictions_rows_evaluation_window')}; unique={s.get('unique_identity_rows')}; duplicates={s.get('duplicate_identity_rows')}; missing_identity={s.get('missing_identity_rows')}; chase={s.get('chase_candidates')}; dangerous={s.get('dangerous_chase')}; healthy={s.get('healthy_chase')}")
    print("Diagnostic completion does not imply a strategy PASS, valid prediction edge, or production approval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
