# GMQ — Full Trade Failure / Tail Loss Predictor

Bu paket **araştırma katmanıdır**. Canlı Global Momentum Quant sistemine hiçbir kod değişikliği yapmaz.

## Ne araştırıyor?

Önceki Early-Adverse çalışması, giriş sonrası ters hareketi öngörme sinyali olduğunu gösterdi ancak ekonomik hard-filter doğrulanmadı. Bu paket doğrudan nihai işlem sonucuna bakar:

- `bad_trade`: `ret_pct <= 0%`
- `tail_5`: `ret_pct <= -5%`
- `tail_10`: `ret_pct <= -10%`
- `tail_15`: `ret_pct <= -15%`
- `blind_spot`: negatif işlem + `early_adverse=False`

Amaç, “işlem neden sonunda kaybediyor?” sorusunu daha doğrudan test etmektir.

## İki feature set

### `strict_signal`
Sinyal kapanışında bilinebildiği varsayılan değişkenleri kullanır. `gap_pct` ve `gap_atr` bilinçli olarak dışarıdadır.

### `entry_open`
`gap_pct` ve `gap_atr` eklenir. Bunlar ertesi açılışta bilindiği için ancak entry anında karar veren model açısından meşrudur; sinyal kapanışında bilinemez.

Ticker ve market kimliği modele verilmez; bunlar yalnızca BIST/ABD kırılımı için kullanılır.

## Model

Tek, sabit ve yorumlanabilir model kullanılır:

`median imputation → StandardScaler → LogisticRegression(C=0.25, class_weight="balanced")`

Hiperparametre optimizasyonu yapılmaz.

## Zaman bölmesi

- TRAIN: `< 2022-01-01`
- HOLDOUT: `>= 2022-01-01`

Holdout tahminleri ve hard-filter eşikleri TRAIN'den bağımsız şekilde tekrar optimize edilmez. Filter eşikleri, yalnızca TRAIN risk tahminlerinin `%75 / %80 / %90 / %95` üst quantile değerleridir.

## Gerekli dosya

Şu CSV pakette hazır gelir:

`research/input/gmq_early_adverse_predictor_trades.csv`

Dosya, önceki Early-Adverse araştırmasında kullanılan 12,944 işlemlik veri setinin aynısıdır.

## Kurulum

1. Paketin içeriğini repo köküne kopyalayın.
2. Aşağıdaki dosya repo kökünde kalmalıdır:

`gmq_full_trade_failure_tail_predictor.py`

3. GitHub Actions için workflow dosyasını şuraya kopyalayın:

`.github/workflows/gmq_full_trade_failure_tail_predictor.yml`

4. CSV şu konumda olmalıdır:

`research/input/gmq_early_adverse_predictor_trades.csv`

Bu araştırma için `config.py`, `engine.py`, `portfolio.py`, `signals.py` veya state dosyalarını değiştirmeyin.

## Manuel çalıştırma

```bash
python gmq_full_trade_failure_tail_predictor.py \
  --input research/input/gmq_early_adverse_predictor_trades.csv \
  --out research/results_full_trade_failure_tail
```

CI/smoke:

```bash
python gmq_full_trade_failure_tail_predictor.py --synthetic
```

## GitHub Actions

Workflow sadece `workflow_dispatch` ile çalışır; otomatik schedule eklenmemiştir. Böylece araştırma koşusunun gecikmesi üretim sistemine bağlı değildir.

Çalıştırma yolu:

`GitHub → Actions → GMQ Full Trade Failure Tail Predictor → Run workflow`

Sonuçlar Actions artifact olarak yüklenir.

## Üretilen ana çıktılar

`gmq_full_trade_failure_tail_metrics.csv`

AUC, PR-AUC, Brier ve hedef oranlarını TRAIN/HOLDOUT bazında verir.

`gmq_full_trade_failure_tail_deciles.csv`

Holdout risk decile'ları ile gerçekleşen failure/tail oranlarını karşılaştırır.

`gmq_full_trade_failure_tail_filter_scenarios.csv`

TRAIN'den türetilmiş risk quantile eşikleri ile Q75/Q80/Q90/Q95 hard-filter diagnostikleri.

`gmq_full_trade_failure_tail_market_breakdown.csv`

BIST ve ABD kırılımı.

`gmq_full_trade_failure_tail_period_breakdown.csv`

Dönem bazında drift kontrolü.

`gmq_full_trade_failure_tail_coefficients.csv`

Standartlaştırılmış logistic katsayıları.

`gmq_full_trade_failure_tail_holdout_predictions.csv`

Her hedef ve feature set için holdout risk tahminleri.

`gmq_full_trade_failure_tail_blind_spot.csv`

Negatif işlemlerin Early-Adverse tarafından kaçırılan kısmının kayıp katkısını gösterir.

`gmq_full_trade_failure_tail_raporu.md`

Tek özet rapor.

## Canlıya alma kuralı

Bu paket kendi başına canlı filtre üretmez. Bir aday ancak şu aşamaları geçerse üretim adayı olur:

`predictive signal → holdout stability → BIST/US consistency → cost stress → full production portfolio A/B → bootstrap/period robustness → no material drawdown regression`

Sonuç ne olursa olsun mevcut BASE sistemi değiştirilmez.


## Dependency fix
The research writes Markdown tables via `pandas.DataFrame.to_markdown()`, so `tabulate==0.9.0` is explicitly installed.
