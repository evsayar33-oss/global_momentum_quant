# GMQ — Cross-Sectional Alpha Quality / Capital Efficiency V1

## Amaç

Bu paket, Global Momentum Quant BASE stratejisinin zaten seçtiği hisseler arasındaki **göreli seçim kalitesinin** sermaye tahsisine çevrilebilip çevrilemeyeceğini test eder.

Bu bir gelecekteki getiriyi tahmin eden ML modeli değildir. Amaç:

> Aynı market + aynı signal_date içinde BASE'in seçtiği hisselerden hangileri sermayenin daha azını hak ediyor?

## Neden bu eksen?

Önceki araştırmalarda Ridge alpha tahmininin mutlak tahmin gücünün zayıf, tail-risk tahmininin ise zaman içinde kararsız olduğu görüldü. Bu V1 model yerine **point-in-time cross-sectional ranking** kullanır.

## Sabit Capital Efficiency Score

Her market/sinyal gününde seçilmiş hisseler için:

- 50% `score_pctile`
- 20% `ret_21d_pct`
- 15% `stock_vs_market_21d_pp`
- 10% `ret_63d_pct`
- 5% inverse `momentum_extension_z`

Her bileşen aynı **market + signal_date** cohortu içinde rank edilir. Bu nedenle geleceğe bakılmaz ve absolute threshold drift problemi azaltılır.

## Policy family

| Policy | Koşul | BUY multiplier |
|---|---|---:|
| BASE | yok | 1.00x |
| CE_GENTLE | günlük market cohortunun bottom 20% | 0.90x |
| CE_BALANCED | günlük market cohortunun bottom 20% | **0.80x** |
| CE_STRONG | günlük market cohortunun bottom 20% | 0.70x |
| CE_BOTTOM30_GENTLE | bottom 30% | 0.90x |

Primary policy: **CE_BALANCED**.

Bu policy'ler test başlamadan sabittir. OOS'a bakarak seçilmez.

## Backtest kuralları

- Gerçek `engine.MarketData` / `portfolio.process_day` production path kullanılır.
- 4 tranche, 5-day stagger, 21-day refresh, risk parity, cash mechanics, catastrophe stop ve maliyet mekanikleri korunur.
- Yalnızca BUY amount araştırma wrapper'ında küçültülür.
- Sells, stops, selection, ranking ve tranche timing değiştirilmez.
- Production `config.py`, `engine.py`, `portfolio.py`, `signals.py` ve state/order/NAV dosyaları değiştirilmez.
- Primary OOS başlangıcı: `2018-01-01`.
- Altı aylık bloklar descriptive robustness olarak raporlanır.
- Cost stress: `0.35 / 0.50 / 0.75 / 1.00 / 1.25% RT`.

## Önemli metodoloji notu

Cross-sectional score yalnızca aynı gün zaten seçilmiş isimler üzerinde hesaplanır. Böylece araştırma, BASE'in seçim mekanizmasının üstüne çok hafif bir **capital-efficiency overlay** ekler. Bu çalışma bir optimizer değildir.

## Gerekli dosyalar

Paket şu dosyaları içerir:

- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1.py`
- `gmq_tail_risk_alpha_conditional_sizing.py` (research helper)
- `gmq_cross_sectional_alpha_quality_requirements.txt`
- `.github/workflows/gmq_cross_sectional_alpha_quality_capital_efficiency_v1.yml`
- bu README
- `INSTALL_MAP.txt`

## Manuel kurulum

1. ZIP içeriğini repo köküne çıkarın.
2. Production dosyalarına dokunmayın.
3. GitHub Actions → **GMQ Cross-Sectional Alpha Quality / Capital Efficiency V1** → **Run workflow**.
4. Sonuç artifact'ini indirin.

## Çıktılar

- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_raporu.md`
- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_metrics.csv`
- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_folds.csv`
- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_monthly.csv`
- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_stress.csv`
- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_quintiles.csv`
- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_policy_usage.csv`
- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_predictions.csv`
- `gmq_cross_sectional_alpha_quality_capital_efficiency_v1_trades.csv`

## Promotion gate

Primary policy ancak şu dört koşulun tamamında aday sayılır:

1. Overall CAGR delta >= 0
2. Matched-month bootstrap 5% bound > 0
3. Positive six-month block ratio >= 60%
4. 5 cost scenario içinde en az 4'ünde CAGR delta > 0

Aksi durumda: **REJECT FOR PROMOTION**.

## Çalıştırma

```bash
python gmq_cross_sectional_alpha_quality_capital_efficiency_v1.py --synthetic
python gmq_cross_sectional_alpha_quality_capital_efficiency_v1.py --years 14 --start 2018-01-01
```
