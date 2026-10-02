"""Sinyaller — araştırmadaki tanımlarla birebir aynı (13 yıllık testte kullanılan formüller).

Girdi: geniş tablolar (satır=tarih, sütun=hisse) o,h,l,c,v.  Çıktı: hisse başına skorlar.
Hiçbir sinyal geleceğe bakmaz: t günündeki değer yalnızca t ve öncesindeki verilerle hesaplanır.
"""
import numpy as np
import pandas as pd


def basics(W, max_move):
    o, h, l, c, v = W["o"], W["h"], W["l"], W["c"], W["v"].fillna(0)
    r = c.pct_change(fill_method=None)
    valid = c.notna() & (v > 0) & (h >= l) & (r.abs() <= max_move)
    vt = (c * v).where(v > 0)
    liq = vt.rolling(20, min_periods=10).median()
    return r, valid, liq


def universe_mask(valid, liq, liq_min_pct):
    return valid & (liq.rank(axis=1, pct=True) >= liq_min_pct) & liq.notna()


def compute(W, names, max_move):
    """İstenen sinyalleri hesaplar. names: kümelik sinyal adları."""
    o, h, l, c, v = W["o"], W["h"], W["l"], W["c"], W["v"].fillna(0)
    r, valid, liq = basics(W, max_move)
    S = {}
    need = set(names)
    if "mom_12_1" in need:
        S["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    if "hi52" in need:
        S["hi52"] = c / c.rolling(252, min_periods=200).max()
    if "low_max" in need:
        S["low_max"] = -r.rolling(21, min_periods=15).max()
    if "clv" in need:
        S["clv"] = (((c - l) - (h - c)) / (h - l).replace(0, np.nan)).rolling(5).mean()
    if need & {"resid_mom", "upvol_ratio", "pos_days", "overnight_mom"}:
        rv = r.where(valid)
        if "resid_mom" in need:
            m = rv.mean(axis=1)
            W_ = 252
            Erm = rv.mul(m, axis=0).rolling(W_, min_periods=150).mean()
            Er = rv.rolling(W_, min_periods=150).mean()
            Em = m.rolling(W_, min_periods=150).mean()
            var = m.rolling(W_, min_periods=150).var()
            beta = (Erm.sub(Er.mul(Em, axis=0))).div(var, axis=0).clip(-1, 4)
            res = rv.sub(beta.shift(1).mul(m, axis=0))
            rs = res.shift(21).rolling(231, min_periods=150)
            S["resid_mom"] = rs.sum() / rs.std()
        if "upvol_ratio" in need:
            S["upvol_ratio"] = (v * (rv > 0)).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
        if "pos_days" in need:
            pos = (rv > 0).astype(float).where(rv.notna())
            S["pos_days"] = pos.rolling(252, min_periods=150).mean()
        if "overnight_mom" in need:
            gap = o / c.shift(1) - 1
            S["overnight_mom"] = gap.rolling(252, min_periods=150).sum()
    return S, valid, liq


def score(W, spec, max_move, liq_min_pct):
    """spec: [[sinyal, yön], ...] -> evren içi yüzdelik sıra ortalaması (yüksek = iyi), evren maskesi."""
    S, valid, liq = compute(W, [k for k, _ in spec], max_move)
    U = universe_mask(valid, liq, liq_min_pct)
    out = None
    for k, s in spec:
        x = S[k].where(U).rank(axis=1, pct=True)
        x = x if s > 0 else 1 - x
        out = x if out is None else out + x
    return out / len(spec), U


def top_picks(score_row, mask_row, n):
    s = score_row.where(mask_row).dropna()
    return list(s.sort_values(ascending=False).index[:n])
