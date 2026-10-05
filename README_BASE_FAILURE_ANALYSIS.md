# Global Momentum Quant — BASE Failure Analysis

Bu paket canlı stratejiyi değiştirmez. Amaç, mevcut BASE işlemlerinin neden kaybettiğini ölçmek ve sonraki araştırma eksenini daraltmaktır.

## Ne ölçüyor?

Bu araştırma, 2026-10-05 tarihli Adaptive Risk V2 sonucundaki BASE referansını daha ileri teşhis etmek için hazırlanmıştır; o referans BASE için CAGR %54.15, Max DD -%24.15 ve işlem win %55.34 raporlamıştır. Bu script aynı üretim mekaniklerini yeniden koşturur; güncel çalıştırmanın sonuçları raporda esas alınmalıdır.

Pozisyon bazında:

- sinyal ve giriş skor percentile değişimi
- 5/21/63 günlük momentum
- 20 günlük volatilite
- ATR(14)
- giriş gap / ATR
- piyasa 5/21 günlük breadth
- piyasa medyan 1/21/63 günlük getiri
- cross-sectional dispersion
- MFE / MAE
- MFE/MAE'nin ATR karşılığı
- sinyal→giriş score decay
- önceden sabit failure etiketleri

Failure etiketleri optimizer değildir. Önceden tanımlıdır:

- `late_extension`
- `gap_risk`
- `rank_decay`
- `market_stress`
- `trend_fade`
- `early_adverse`

## Çıktılar

- `gmq_base_failure_analysis_raporu.md`
- `gmq_base_failure_analysis_trades.csv`
- `gmq_base_failure_analysis_baskets.csv`
- `gmq_base_failure_analysis_patterns.csv`
- `gmq_base_failure_analysis_buckets.csv`

## Manuel çalıştırma

```bash
python gmq_base_failure_analysis.py --years 14 --cost 0.35 --out-dir failure_output
```

Smoke test:

```bash
python gmq_base_failure_analysis.py --synthetic --out-dir smoke_test_output
```

## GitHub Actions

`.github/workflows/gmq_base_failure_analysis.yml` manuel tetiklenebilir. Ayrıca haftalık Cuma 15:15 UTC ile planlanmıştır.

Bu workflow yalnızca artifact üretir; `config.py`, `engine.py`, `portfolio.py`, `signals.py`, state/order/NAV/trade dosyalarını commit etmez.

## Araştırma disiplini

Failure Analysis sonucunda doğrudan filtre eklenmemelidir. Önce tek bir failure mekanizması seçilip bağımsız A/B testi, cost stress ve final holdout ile doğrulanmalıdır.
