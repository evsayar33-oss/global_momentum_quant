# GMQ — Early-Adverse Predictor / Entry Risk Quality

## Kurulum

Bu paket araştırma içindir; canlı strateji dosyalarını değiştirmez.

Repo köküne şu iki dosya/yolu kopyala:

- `gmq_early_adverse_predictor.py`
- `.github/workflows/gmq_early_adverse_predictor.yml`

Sonra GitHub → **Actions** → **Global Momentum Quant - Early-Adverse Predictor** → **Run workflow**.

Varsayılan ayarlar:

- Veri: `14` yıl
- Round-trip maliyet: `0.35%`
- Holdout başlangıcı: `2022-01-01`

0.35% RT benchmark maliyeti değiştirilmemelidir.

## Testin ne yaptığı

BASE üretim motorunu aynen çalıştırır:

`engine.MarketData` → `portfolio.process_day`

Ardından her işlem için yalnızca signal-date bilgilerinden ex-ante feature seti çıkarır ve ertesi gün girişten sonraki ilk 4 gözlemde:

`early_mae_atr <= -1.0`

ise `early_adverse = True` olarak etiketler.

Model fit yalnızca `2013-...` → `2021-12-31` TRAIN bölümündedir. HOLDOUT 2022-01-01 sonrası tamamen dışarıda tutulur.

Model:

- standardize edilmiş sabit feature set
- Logistic Regression (`C=0.25`, `class_weight=balanced`)
- median imputation

Bu parametreler test sırasında optimize edilmez.

## Çıktılar

`gmq_early_adverse_predictor_raporu.md`

Ana karar raporu.

`gmq_early_adverse_predictor_model.csv`

Train + holdout tahminleri.

`gmq_early_adverse_predictor_holdout.csv`

Yalnızca holdout.

`gmq_early_adverse_predictor_buckets.csv`

Tek-değişkenli erken-tarama sonuçları.

`gmq_early_adverse_predictor_filter_scenarios.csv`

Train'de öğrenilen Q75/Q90 risk eşiklerinin holdout ve maliyet stresindeki teşhis etkisi.

`gmq_early_adverse_predictor_trades.csv`

İşlem bazlı feature + outcome dataset.

## Önemli karar kuralı

Bu testin pozitif çıkması bile canlı filtre eklemek için tek başına yeterli değildir.

Gerekli sonraki aşama:

1. Holdout predictor doğruluğu korunmalı.
2. Holdout'ta Q75/Q90 filtreleri early-adverse oranını düşürmeli.
3. Ortalama işlem getirisi / maliyet sonrası getiri korunmalı.
4. 0.50 / 0.75 / 1.00 / 1.25% maliyet stresinde yön bozulmamalı.
5. BIST ve US ayrı ayrı aynı mekanizmayı göstermeli.

Bu şartlar karşılanırsa bir sonraki bağımsız çalışma:
**Entry Risk Quality A/B validation**.

Canlı config/state/order/NAV/trades değişmez.
