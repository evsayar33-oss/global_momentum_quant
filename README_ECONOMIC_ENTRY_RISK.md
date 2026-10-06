# GMQ — Economic Entry Risk / Adverse Severity Test

## Neden bu test?

Önceki Early-Adverse Predictor, giriş sonrası erken adverse hareketi holdout'ta bir miktar öngörebildi. Ancak Q75/Q90 filtrelerinin dışladığı işlemler hâlâ ortalama olarak pozitifti.

Bu test bu problemi ikiye ayırır:

- `temporary_adverse`: erken adverse oldu, ancak işlem sonunda pozitif kapandı.
- `harmful_adverse`: erken adverse oldu ve işlem sonunda sıfır veya negatif kapandı.

Ana hedef artık doğrudan `harmful_adverse`'dir.

## Veri

Bu paket, önceki çalışmada üretilen gerçek:

`gmq_early_adverse_predictor_trades.csv`

dosyasının aynı kopyasını `research_inputs/` altında taşır.

Bu nedenle yeni Yahoo indirmesi yoktur. Önceki veri seti üzerinde kontrollü, hızlı ve tekrarlanabilir bir teşhis çalışmasıdır.

## Özellikler

Model sadece signal_date'te bilinen feature'ları kullanır.

Şunlar KULLANILMAZ:

- final `ret_pct`
- `early_adverse`
- `early_mae_pct`
- `early_mfe_pct`
- `path_class`
- `harmful_adverse`
- `temporary_adverse`
- geleceğe ait herhangi bir bilgi

Model:

- multinomial Logistic Regression
- fixed `C=0.25`
- balanced class weights
- standard scaler + median imputation

Model sadece TRAIN'de fit edilir.

Holdout:

`2022-01-01` ve sonrası.

Q75/Q90 risk eşikleri yalnızca TRAIN predicted-risk dağılımından alınır.

## Kurulum

Repo kökünde:

1. `gmq_economic_entry_risk_test.py` ekle.
2. `research_inputs/gmq_early_adverse_predictor_trades.csv` ekle.
3. `.github/workflows/gmq_economic_entry_risk_test.yml` ekle.

Canlı `engine.py`, `portfolio.py`, `config.py`, state/order/NAV/trades dosyaları değiştirilmez.

## Çalıştırma

GitHub:

Actions → Global Momentum Quant - Economic Entry Risk Test → Run workflow

Bu workflow yalnızca manuel çalışır.

## Çıktılar

- `gmq_economic_entry_risk_raporu.md`
- `gmq_economic_entry_risk_trades.csv`
- `gmq_economic_entry_risk_holdout.csv`
- `gmq_economic_entry_risk_deciles.csv`
- `gmq_economic_entry_risk_filter_scenarios.csv`
- `gmq_economic_entry_risk_confusion_matrix.csv`
- `gmq_economic_entry_risk_market_breakdown.csv`

## Karar

Pozitif sonuç şu an için canlı entegrasyon anlamına gelmez.

Pozitif bir ekonomik predictor bulunursa sonraki test:

**Entry Risk Quality full-system A/B validation**

olmalıdır.

Bu aşamada amaç sadece `temporary` adverse ile `harmful` adverse'i ayırabildiğimizi kanıtlamaktır.
