# GMQ — Momentum Entry Timing / Chase Avoidance V1

## Amaç

Önceki BASE Failure Analysis, iki entry-timing mekanizmasını araştırma için öne çıkardı:

- `momentum_extension_z > 2.5`
- `gap_atr > 2.0`

Önceki diagnostik raporda `late_extension` 1,470 trade ve toplam kayıpların %14.56'sını; `gap_risk` 537 trade ve toplam kayıpların %3.86'sını oluşturuyordu. Bu eşikler önceki diagnostik bucket'lardan alınmış olup V1 içinde yeniden optimize edilmez.

Bu çalışma BASE'in hisse seçimini değiştirmez. Sadece giriş anında chase koşulu görüldüğünde yeni BUY amount'ını azaltır.

## Zamanlama / leakage

`momentum_extension_z` sinyal kapanışında bilinir.

`gap_atr`, sinyal kapanışından sonraki gerçek giriş açılışında bilinir. Araştırma wrapper'ı bu koşulu pending BUY emrine uygular; dolayısıyla kararın ekonomik anlamı **entry open'da karar verme** şeklindedir. Çıkış fiyatı, MFE/MAE, erken adverse veya başka geleceğe dönük path bilgileri giriş kararına verilmez.

Eksik timing verisi varsa işlem cezalandırılmaz; BASE allocation korunur.

## Sabit policy family

| Policy | Koşul | BUY multiplier |
|---|---|---:|
| BASE | yok | 1.00x |
| EXT_SOFT | extension_z > 2.5 | 0.75x |
| GAP_SOFT | gap_atr > 2.0 | 0.75x |
| **CHASE_AND_SOFT** | **extension_z > 2.5 AND gap_atr > 2.0** | **0.75x** |
| CHASE_OR_SOFT | extension_z > 2.5 OR gap_atr > 2.0 | 0.75x |
| CHASE_AND_STRONG | AND | 0.50x |
| GAP_SKIP | gap_atr > 2.0 | 0.00x |
| EXT_SKIP | extension_z > 2.5 | 0.00x |

Primary policy: `CHASE_AND_SOFT`.

Strong/skip policy'ler sensitivity check'tir; optimizer değildir.

## Production protection

Araştırma:

- `engine.py` değiştirmez.
- `portfolio.py` değiştirmez.
- `config.py` değiştirmez.
- `signals.py` değiştirmez.
- state/order/NAV/trade canlı dosyalarını yazmaz.
- Aynı `engine.MarketData` + `portfolio.process_day` production path'ini kullanır.
- Sadece research runtime'da BUY amount'ını değiştirir.

Sell path, stop, catastrophe stop, tranche schedule, hold/refresh, monthly risk parity ve cash/FX mechanics korunur.

## Testler

Ana testler:

1. 2018+ full OOS-style historical evaluation
2. 6 aylık robustness blocks
3. Eşleştirilmiş aylık bootstrap
4. `%0.35 / 0.50 / 0.75 / 1.00 / 1.25` RT cost stress
5. BIST / US diagnostik kırılımı
6. Chase-flag'li işlemlerin realized return / tail-loss davranışı

## Karar standardı

Primary `CHASE_AND_SOFT` ancak şu dört koşul birlikte sağlanırsa `PROMISING — NEEDS FINAL HOLDOUT / PARITY` sayılır:

- OOS CAGR delta >= 0
- bootstrap %5 alt sınır > 0
- pozitif 6 aylık fold oranı >= %60
- 5 maliyet senaryosunun en az 4'ünde CAGR delta > 0

Aksi durumda:

**REJECT FOR PROMOTION**

Başarılı olsa dahi otomatik production entegrasyonu yapılmaz; ardından ayrı bir final holdout / production parity testi gerekir.

## Kurulum

ZIP'i repo köküne çıkarın.

Repo kökünde mevcut olması gereken production/research bağımlılığı:

- `data.py`
- `engine.py`
- `portfolio.py`
- `config.py`
- `gmq_tail_risk_alpha_conditional_sizing.py`
- bu helper'ın kullandığı mevcut `gmq_early_adverse_predictor.py`

Bu test için eski bir optimizer veya başka bir sizing policy dosyasının çalıştırılması gerekmez.

## Lokal

```bash
pip install -r gmq_momentum_entry_timing_requirements.txt
python gmq_momentum_entry_timing_chase_avoidance_v1.py --synthetic
python gmq_momentum_entry_timing_chase_avoidance_v1.py --years 14 --start 2018-01-01
```

## GitHub Actions

`Actions → GMQ Momentum Entry Timing / Chase Avoidance V1 → Run workflow`

Workflow yalnızca manuel (`workflow_dispatch`) çalışır.

## Outputs

`research/results_momentum_entry_timing_chase_avoidance_v1/` altında:

- `gmq_momentum_entry_timing_chase_avoidance_v1_raporu.md`
- `gmq_momentum_entry_timing_chase_avoidance_v1_metrics.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v1_folds.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v1_monthly.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v1_stress.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v1_diagnostics.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v1_policy_usage.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v1_predictions.csv`
- `gmq_momentum_entry_timing_chase_avoidance_v1_trades.csv`

**Canlı sistemi değiştirmez.**
