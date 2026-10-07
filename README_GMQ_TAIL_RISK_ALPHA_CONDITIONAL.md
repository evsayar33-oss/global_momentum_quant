# GMQ — Tail-Risk × Alpha Conditional Sizing V1

## Amaç

Önceki testte tail-risk tek başına pozisyon küçültünce CAGR kaybı oluştu. Bu araştırma, yüksek tail-risk'in **yüksek alpha ile birlikte** geldiği işlemleri koruyup, yüksek tail-risk + düşük alpha kombinasyonunu küçültmenin daha iyi çalışıp çalışmadığını test eder.

## Kurulum

Production dosyalarına dokunulmaz. Aşağıdaki yeni dosyalar repository root'a ve workflow `.github/workflows/` altına eklenir:

- `gmq_tail_risk_alpha_conditional_sizing.py`
- `gmq_tail_risk_alpha_requirements.txt`
- `.github/workflows/gmq_tail_risk_alpha_conditional_sizing.yml`

Bu araştırma mevcut `engine.py`, `portfolio.py`, `data.py`, `config.py`, `signals.py` ve mevcut `gmq_early_adverse_predictor.py` modülünü import eder.

## Kritik metodoloji

Model fit cutoff: **2022-01-01**.

FIT'e yalnızca hem `signal_date < 2022-01-01` hem de `exit_date < 2022-01-01` olan gerçekleşmiş işlemler girer. Böylece 2022'ye taşan 2021 sonu işlemleri model eğitiminden çıkarılır.

Model feature'ları yalnızca signal-date strict setidir. `gap_pct`, `gap_atr`, early-MAE, early-MFE, early-adverse ve benzeri geleceğe dönük path değişkenleri sizing modeline verilmez.

## Model

### Alpha

`Ridge(alpha=10)` ile clipped `ret_pct [-25%, +25%]` tahmin edilir.

### Tail risk

Üç sabit lojistik model:

- `P(ret <= -5%)`
- `P(ret <= -10%)`
- `P(ret <= -15%)`

Birleşik tail-risk skoru:

`0.25 × P(-5) + 0.35 × P(-10) + 0.40 × P(-15)`

Ağırlıklar önceden sabittir ve OOS'a göre optimize edilmez.

## Conditional sizing

Alpha ve risk percentilleri **market bazında**, yalnızca FIT dağılımından alınır.

Temel kural:

`gap = max(risk_percentile - alpha_percentile, 0)`

`multiplier = max(floor, 1 - strength × gap)`

Test edilen önceden tanımlı politikalar:

| Politika | Strength | Floor |
|---|---:|---:|
| BASE | 0.00 | 1.00 |
| CONDITIONAL_GENTLE | 0.35 | 0.65 |
| CONDITIONAL_BALANCED | 0.60 | 0.40 |
| CONDITIONAL_STRONG | 0.80 | 0.25 |

Alpha riskten daha iyiyse pozisyon küçültülmez. Böylece önceki testte görülen **"yüksek tail-risk ama yüksek alpha"** işlemlerinin tamamen bastırılması önlenir.

## Production-path entegrasyonu

Araştırma backtest'i önce gerçek production engine ile BASE emirlerini üretir. Sonra yalnızca araştırma runtime'ında `engine.step_market` wrapper'ı ile BUY amount'ları ölçekler.

Şunlar değiştirilmez:

- hisse seçimi
- score/ranking
- tranche zamanlaması
- satış emirleri
- catastrophe stop
- aylık risk parity
- cash mechanics
- production state dosyaları

Ölçeklenen emirlerden artan tutar nakit olarak kalır; inter-market transfer mantığı bu kasıtlı eksik yatırım nedeniyle yeniden tetiklenmez.

## Test çıktıları

- `gmq_tail_risk_alpha_conditional_sizing_raporu.md`
- `gmq_tail_risk_alpha_conditional_metrics.csv`
- `gmq_tail_risk_alpha_conditional_stress.csv`
- `gmq_tail_risk_alpha_conditional_predictions.csv`
- `gmq_tail_risk_alpha_conditional_trades.csv`
- `gmq_tail_risk_alpha_conditional_policy_usage.csv`
- `gmq_tail_risk_alpha_conditional_model_metrics.csv`
- `gmq_tail_risk_alpha_conditional_alpha_bins.csv`
- `gmq_tail_risk_alpha_conditional_risk_bins.csv`

## Karar kuralı

Bu test bir optimizer değildir. Bir aday ancak OOS'ta:

1. BASE'e göre kabul edilebilir CAGR korurken,
2. MaxDD / tail-loss katkısını gerçekten düşürürken,
3. Profit Factor ve worst-12M performansını bozmazken,
4. bootstrap aylık farkı istatistiksel olarak desteklerken,
5. 0.35–1.25% maliyet streslerinde yönünü korurken

aşamaya aday olur.

**Canlı entegrasyon otomatik yapılmaz.**
