# GMQ Early-Adverse Predictor — Fixed v4

Bu paket, GitHub Actions run #5'te tespit edilen gerçek kod hatasını düzeltir.

## Kesin hata

`gmq_early_adverse_predictor.py` satır 611-612'de:

`preds.sample=="train"`
`preds.sample=="holdout"`

kullanılmıştı.

Pandas'ta `DataFrame.sample` gerçek sütun değil, DataFrame'in `sample()` metodudur. Bu nedenle ifade `False` üretiyor ve `preds[False]` çağrısı `KeyError: False` oluşturuyordu.

Düzeltme:

`preds["sample"] == "train"`
`preds["sample"] == "holdout"`

Aynı bölümde yakın DataFrame sütun erişimleri de bracket notation'a çekildi.

## Kurulum

1. `gmq_early_adverse_predictor.py` dosyasını bu paketle değiştir.
2. `gmq_early_adverse_predictor_hardened_runner.py` dosyasını repo köküne ekle/değiştir.
3. `.github/workflows/gmq_early_adverse_predictor.yml` dosyasını paket içindekiyle değiştir.

Sonra:

Actions → Global Momentum Quant - Early-Adverse Predictor (Fixed v4) → Run workflow

Parametreler:
- years = 14
- cost = 0.35

## Kontroller

Workflow gerçek araştırmadan önce:
- syntax
- dependency versions
- DataFrame column regression
- Yahoo preflight
- synthetic smoke

kontrollerini yapar.

Canlı config/state/order/NAV/trades değiştirilmez.
