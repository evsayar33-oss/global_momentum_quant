# GMQ Early-Adverse Predictor — Fixed v5

v4 çalışmasındaki kesin syntax hatasını düzeltir:

`SyntaxError: f-string: unmatched '['`

Hatalı yapı:

```python
print(f"Trades={len(trades)} | EarlyAdverse={trades["early_adverse"].mean()*100:.2f}%")
```

Güvenli yapı:

```python
early_adverse_pct = float(trades["early_adverse"].mean() * 100.0)
print(f"Trades={len(trades)} | EarlyAdverse={early_adverse_pct:.2f}%")
```

Ayrıca workflow gerçek araştırmadan önce syntax/f-string regression kontrolü yapar.

## Kurulum

Repo'da:

1. `gmq_early_adverse_predictor.py` → bu v5 dosyasıyla değiştir.
2. `gmq_early_adverse_predictor_hardened_runner.py` → paketteki sürümü kullan.
3. `.github/workflows/gmq_early_adverse_predictor.yml` → paketteki sürümle değiştir.

Sonra:

Actions → Global Momentum Quant - Early-Adverse Predictor (Fixed v5) → Run workflow

`years=14`, `cost=0.35`.

Bu paket canlı config/state/order/NAV/trades dosyalarını değiştirmez.
