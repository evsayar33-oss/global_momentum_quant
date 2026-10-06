# GMQ Tail Risk Adaptive Position Sizing V1

Bu paket **Global Momentum Quant** için araştırma amaçlı bir sonraki testtir.

## Amaç

Önceki Full Trade Failure / Tail Loss araştırmasında ordinary loss (`<=0%`) tahmini zayıf, fakat tail-loss tahmini özellikle `<=-10%` ve `<=-15%` için anlamlı OOS ayrışma göstermişti. Bu nedenle yeni test işlemi iptal etmek yerine, tail-risk yükseldiğinde yalnızca **yeni alım notionalini** monoton biçimde küçültür.

Bu paket canlı sistemi değiştirmez.

## Kurulum

ZIP'i repo köküne aç. İçindeki dosyalar doğrudan:

- `gmq_tail_risk_adaptive_sizing.py`
- `.github/workflows/gmq_tail_risk_adaptive_sizing.yml`
- `research_inputs/gmq_full_trade_failure_tail_holdout_predictions.csv`
- `README_GMQ_TAIL_RISK_ADAPTIVE_SIZING.md`

şeklinde yerleşir.

Production `config.py`, `engine.py`, `portfolio.py`, `signals.py`, state/order/NAV dosyaları değiştirilmez.

## Test mantığı

OOS tahminlerde yalnızca `strict_signal` kullanılır. `tail_5`, `tail_10`, `tail_15` tahminleri aynı `market + signal_date` kesitinde percentile rank'e dönüştürülür ve sabit blend ile tek bir risk skoru oluşturulur:

`risk = 0.25*rank(tail_5) + 0.35*rank(tail_10) + 0.40*rank(tail_15)`

Bu ağırlıklar test başlamadan sabittir; geçmişte optimize edilmez.

### Boyutlandırma politikaları

| Policy | Formula | Minimum size |
|---|---|---:|
| BASE | `1.00` | 100% |
| GENTLE | `max(0.70, 1 - 0.30*risk)` | 70% |
| MODERATE | `max(0.50, 1 - 0.50*risk)` | 50% |
| DEFENSIVE | `max(0.35, 1 - 0.65*risk)` | 35% |

Hiçbir politika BASE boyutunu artırmaz.

Sadece `engine.step_market()` çağrısında yeni oluşturulan `BUY` pending order'ları ölçeklenir. `amount` ve varsa `full` birlikte ölçeklenir; böylece portföy transferi sonradan risk sınırını geri açmaz.

Mevcut pozisyonlar yeniden yazılmaz. Çıkış, süre dolumu, felaket stopu, sigorta, seçim, aylık risk-parity ve diğer üretim mekanikleri aynıdır.

## Ana ölçüm

Primary dönem **2022-01-01 sonrası** OOS holdout'tur. Bu, önceki tail predictor'ın TRAIN ile aynı örneklemde yeniden kullanılmasını önlemek içindir.

Maliyet stresleri:

`0.35 / 0.50 / 0.75 / 1.00 / 1.25 % RT`

Önceki GMQ full-system araştırmalarındaki konvansiyona paralel olarak her senaryoda BIST ve US maliyeti aynı RT senaryosuna çekilir.

## Çıktılar

`local_output/gmq_tail_risk_adaptive_sizing/` altında:

- `gmq_tail_risk_adaptive_sizing_raporu.md`
- `gmq_tail_risk_adaptive_sizing_metrics.csv`
- `gmq_tail_risk_adaptive_sizing_stress.csv`
- `gmq_tail_risk_adaptive_sizing_orders.csv`
- `gmq_tail_risk_adaptive_sizing_trades.csv`
- `gmq_tail_risk_oos_scores.csv`
- `gmq_tail_risk_adaptive_sizing_primary_review.csv`

## GitHub Actions

`Actions → GMQ Tail Risk Adaptive Position Sizing → Run workflow`

varsayılan değerlerle çalıştırılır.

İlk olarak smoke test çalışır; sonra gerçek OOS full-system A/B başlar.

## Karar disiplini

Bu test otomatik olarak canlı `config.py` veya `engine.py` değiştirmez.

Öncelikli bakılacak metrikler:

1. CAGR ve MaxDD birlikte.
2. Profit Factor.
3. `Tail15_loss_share_pct` azalıyor mu?
4. Aday politika hangi kârlı işlemleri gereksiz küçültüyor?
5. Aylık paired delta ve bootstrap CI.
6. 5 maliyet senaryosunda davranış korunuyor mu?

Trade-level olumlu görünse bile doğrudan production'a alınmaz. Production adımı ancak ayrı, tam portföy teyidi ve holdout/robustness kanıtı sonrasında değerlendirilir.
