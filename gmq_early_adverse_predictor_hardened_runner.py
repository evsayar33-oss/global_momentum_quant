#!/usr/bin/env python3
"""
GMQ Early-Adverse Predictor — Hardened runner

Bu wrapper araştırma kodunun matematiğini değiştirmez.
Amaç:
1) bağımlılık sürümlerini sabitlemek,
2) Yahoo/yfinance istek yükünü küçültmek,
3) gerçek exception tipini + tam traceback'i GitHub loguna yazmak,
4) mevcut predictor.run_project() fonksiyonunu doğrudan çağırmak.

Canlı config/state/order/NAV/trades değiştirilmez.
"""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    years = int(os.environ.get("YEARS", "14"))
    cost = float(os.environ.get("COST", "0.35"))
    out_dir = Path(os.environ.get("OUT_DIR", "gmq_early_adverse_predictor_output"))

    print("=== GMQ EARLY-ADVERSE HARDENED RUNNER ===", flush=True)
    print(f"Python: {sys.version}", flush=True)
    print(f"pandas: {pd.__version__}", flush=True)
    print(f"numpy: {np.__version__}", flush=True)

    try:
        import yfinance as yf
        print(f"yfinance: {yf.__version__}", flush=True)
    except Exception as exc:
        print(f"yfinance version okunamadı: {type(exc).__name__}: {exc!r}", flush=True)

    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}", flush=True)
    except Exception as exc:
        print(f"scikit-learn version okunamadı: {type(exc).__name__}: {exc!r}", flush=True)

    # Research-only network hardening. Production data.py/config.py are untouched.
    try:
        import config as C
        old_chunk = getattr(C, "DOWNLOAD_CHUNK", None)
        C.DOWNLOAD_CHUNK = 30
        print(f"DOWNLOAD_CHUNK: {old_chunk} -> {C.DOWNLOAD_CHUNK} (research-only)", flush=True)
    except Exception as exc:
        print(f"DOWNLOAD_CHUNK override uygulanamadı: {type(exc).__name__}: {exc!r}", flush=True)

    # Force a clean import of the predictor module from repo root.
    predictor = importlib.import_module("gmq_early_adverse_predictor")

    print("Stage: run_project() starting", flush=True)
    try:
        predictor.run_project(years, cost, out_dir)
    except BaseException as exc:
        print("", flush=True)
        print("========== FATAL RESEARCH EXCEPTION ==========", flush=True)
        print(f"TYPE : {type(exc).__name__}", flush=True)
        print(f"STR  : {str(exc)!r}", flush=True)
        print(f"REPR : {exc!r}", flush=True)
        print("TRACEBACK:", flush=True)
        traceback.print_exc()
        print("===============================================", flush=True)
        raise
    print("Stage: run_project() completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
