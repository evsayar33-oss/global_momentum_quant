# GMQ — Alpha-Quality Conditional Allocation V1 — Walk-Forward Validation

## Amaç

Önceki **Alpha-Quality Conditional Allocation V1** testi, `COND_GENTLE / BALANCED / STRONG` ailelerinin tek bir 2022+ holdout üzerinde BASE ile olumlu yönde ayrıştığını gösterdi. Bu paket o hipotezi **yeniden optimize etmez**; aynı önceden tanımlı politika ailesini expanding-window walk-forward ile zaman içinde tekrar sınar.

Bu nedenle testin temel sorusu:

> `LOW_ALPHA + LOW_RISK` koşuluna dayalı küçük bir BUY allocation azaltımı, farklı piyasa dönemlerinde yeni OOS verisi görülmeden önce öğrenilmiş model/CDF ile aynı ekonomik avantajı tekrar üretebiliyor mu?

## Primary candidate

Walk-forward başlamadan önce dondurulmuş primary policy:

`COND_STRONG = alpha percentile <= 25% AND tail-risk percentile <= 50% -> BUY multiplier 0.50`

Bu seçim walk-forward sonuçlarından yapılmaz. Önceki V1 testinden gelen politika ailesi sabitlenmiştir.

## Walk-forward protokolü

- İlk OOS: `2018-01-01`
- Refit: her `6` takvim ayında bir
- Expanding training window: geçmişin tamamı
- Her fold için fit satırları yalnızca:
  - `signal_date < fold_start`
  - `exit_date < fold_start`
- Fold modeli ve training-only empirical CDF'ler ilgili OOS blok boyunca kilitlidir.
- OOS trade sonuçları o fold'un model fit'inde kullanılmaz.
- Yeni threshold/multiplier/grid search yapılmaz.
- Production engine/portfolio/config/signals/state/order/NAV dosyaları değiştirilmez.

## Test edilen politikalar

| Policy | Koşul | BUY multiplier |
|---|---|---:|
| BASE | yok | 1.00 |
| ALPHA_GENTLE | alpha <= Q25 | 0.85 |
| ALPHA_BALANCED | alpha <= Q25 | 0.70 |
| ALPHA_STRONG | alpha <= Q25 | 0.50 |
| COND_GENTLE | alpha <= Q25 & risk <= Q50 | 0.85 |
| COND_BALANCED | alpha <= Q25 & risk <= Q50 | 0.70 |
| **COND_STRONG** | **alpha <= Q25 & risk <= Q50** | **0.50** |

Tüm politikalar walk-forward döneminde karşılaştırma amacıyla çalıştırılır. Cost stress yalnızca önceden belirlenmiş primary `COND_STRONG` için uygulanır; böylece hesap maliyeti düşük tutulurken ana adayın robustness testi korunur.

## Model

Önceki V1 ile aynı sabit model ailesi kullanılır:

- Alpha: `Ridge(alpha=10)`; `ret_pct` hedefi `[-25%, +25%]` ile clip edilir.
- Tail risk:
  - `P(ret <= -5%)`
  - `P(ret <= -10%)`
  - `P(ret <= -15%)`
- Tail-risk birleşimi:
  - `0.25 * tail_5 + 0.35 * tail_10 + 0.40 * tail_15`
- Classifier: `LogisticRegression(C=0.25, class_weight='balanced')`
- Feature seti: strict signal-date seti.
- Percentile CDF'ler market bazında ve sadece fold training tahminlerinden hesaplanır.

## Production-path kuralları

Araştırma önce gerçek repository `engine.MarketData` + `portfolio.process_day` yoluyla BASE trade history üretir. Daha sonra walk-forward modelinin ürettiği frozen BUY multiplier yalnızca araştırma runtime wrapper'ında uygulanır.

Değiştirilmez:

- hisse seçimi
- ranking / score
- tranche timing
- hold / refresh
- sell path
- stop / catastrophe stop
- monthly risk parity
- cash / FX mechanics
- production state/order/NAV dosyaları
- `engine.py`, `portfolio.py`, `config.py`, `signals.py`

## Çıktılar

`research/results_alpha_quality_walk_forward_validation_v1/` altında:

- `gmq_alpha_quality_walk_forward_validation_v1_raporu.md`
- `gmq_alpha_quality_walk_forward_validation_v1_metrics.csv`
- `gmq_alpha_quality_walk_forward_validation_v1_folds.csv`
- `gmq_alpha_quality_walk_forward_validation_v1_monthly.csv`
- `gmq_alpha_quality_walk_forward_validation_v1_stress.csv`
- `gmq_alpha_quality_walk_forward_validation_v1_model_metrics.csv`
- `gmq_alpha_quality_walk_forward_validation_v1_threshold_drift.csv`
- `gmq_alpha_quality_walk_forward_validation_v1_policy_usage.csv`
- `gmq_alpha_quality_walk_forward_validation_v1_predictions.csv`
- `gmq_alpha_quality_walk_forward_validation_v1_trades.csv`

## Önemli rapor kontrolleri

Raporda özellikle şu dört kontrol primary `COND_STRONG` için kararın merkezindedir:

1. OOS toplam CAGR farkı BASE'e göre negatif mi/pozitif mi?
2. Fold'ların en az `%60`'ında dönem getirisi BASE'in üzerinde mi?
3. Eşleştirilmiş aylık bootstrap `%5` alt sınırı sıfırın üzerinde mi?
4. `%0.35 / 0.50 / 0.75 / 1.00 / 1.25` RT maliyet stresinin en az `4/5` senaryosunda CAGR farkı pozitif mi?

Ek olarak:

- MaxDD / PF
- model OOS AUC / PR-AUC
- threshold drift
- policy usage
- BIST / US kullanım dağılımı

raporlanır.

## Karar standardı

`COND_STRONG` ancak aşağıdakiler birlikte sağlanırsa **WALK-FORWARD PASS CANDIDATE** sayılır:

- toplam OOS CAGR farkı >= 0
- aylık bootstrap %5 alt sınırı > 0
- pozitif fold oranı >= %60
- primary cost stress'te >= 4/5 senaryo pozitif CAGR farkı

Aksi durumda:

**WALK-FORWARD FAIL / DO NOT PROMOTE**

Bu karar production'a otomatik entegrasyon anlamına gelmez. Başarılı olsa bile bir sonraki adım ayrı production-parity / integration validation olmalıdır.

## Kurulum

ZIP'i repository köküne açın. Bu paket tek seferlik manuel kurulum için aşağıdakileri birlikte içerir:

- `gmq_alpha_quality_walk_forward_validation_v1.py`
- `gmq_alpha_quality_conditional_allocation_v1.py`
- `gmq_tail_risk_alpha_conditional_sizing.py`
- `gmq_alpha_quality_requirements.txt`
- `.github/workflows/gmq_alpha_quality_walk_forward_validation_v1.yml`
- bu README

GitHub Actions:

`Actions → GMQ Alpha-Quality Conditional Allocation V1 - Walk-Forward Validation → Run workflow`

Workflow parametreleri kullanıcı tarafından değiştirilmeyecek şekilde sabitlenmiştir:

- start `2018-01-01`
- refit `6 months`
- years `14`

## Lokal smoke / gerçek test

```bash
pip install -r gmq_alpha_quality_requirements.txt
python gmq_alpha_quality_walk_forward_validation_v1.py --synthetic
python gmq_alpha_quality_walk_forward_validation_v1.py --years 14 --start 2018-01-01 --refit-months 6
```

**CANLI SİSTEME DEĞİŞİKLİK YAPILMAZ.**
