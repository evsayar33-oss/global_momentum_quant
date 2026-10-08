# GMQ — Asymmetric Tail-Risk Allocation V1

Bu paket **yalnızca araştırma** katmanıdır. `config.py`, `engine.py`, `portfolio.py`, `signals.py`, state/order/NAV/trade dosyalarına üretim değişikliği yapmaz.

## Amaç

Önceki Tail-Risk × Alpha Conditional Sizing testinin temel problemi, risk yüksek olduğunda pozisyonu geniş biçimde küçültmenin yüksek alpha taşıyan kazananları da azaltmasıydı. Bu yeni test daha dar bir hipotez sınar:

> **Asıl kaçınılması gereken kombinasyon `LOW ALPHA + HIGH TAIL RISK` olabilir.**

Bu nedenle yüksek tail-risk tek başına cezalandırılmaz.

### Sabit müdahale bölgesi

Market-bazlı, yalnızca TRAIN dağılımından türetilen percentiller kullanılır:

`alpha_pctile < 0.40 AND risk_pctile >= 0.60`

Diğer üç bölge tam pozisyonda bırakılır:

- yüksek alpha + yüksek risk → `1.00`
- yüksek alpha + düşük risk → `1.00`
- düşük alpha + düşük risk → `1.00`
- **düşük alpha + yüksek risk → küçültme**

### Test edilen politikalar

| Policy | Danger multiplier |
|---|---:|
| BASE | 1.00 |
| ASYM_GENTLE | 0.85 |
| ASYM_BALANCED | 0.70 |
| ASYM_STRONG | 0.50 |

Bu değerler OOS sonuçlarına göre seçilmez. Optimizer yoktur.

## Ön koşullar

Bu V1, önceki doğrulanmış araştırma modülünü yeniden kullanır. Repo kökünde şunlar bulunmalıdır:

- `gmq_tail_risk_alpha_conditional_sizing.py`
- `gmq_early_adverse_predictor.py`
- `engine.py`
- `portfolio.py`
- `data.py`
- `config.py`

Production dosyaları değiştirmeyin.

## Kurulum

Repo köküne şu iki dosyayı ekleyin:

- `gmq_asymmetric_tail_risk_allocation_v1.py`
- `gmq_asymmetric_tail_risk_requirements.txt`

Workflow'u ekleyin:

- `.github/workflows/gmq_asymmetric_tail_risk_allocation_v1.yml`

## Lokal test

Önce hızlı smoke test:

```bash
python gmq_asymmetric_tail_risk_allocation_v1.py --synthetic
```

Gerçek production-path araştırması:

```bash
python gmq_asymmetric_tail_risk_allocation_v1.py --years 14 --start 2022-01-01
```

Beklenen süre veri indirme + engine replay nedeniyle makineye ve veri kaynağına göre değişir.

## GitHub Actions

Workflow yalnızca `workflow_dispatch` ile çalışır. Böylece her commit'te veya her gün gereksiz uzun araştırma koşmaz.

GitHub → **Actions** → **GMQ - Asymmetric Tail-Risk Allocation V1 Research** → **Run workflow**.

Varsayılan:

- years: `14`
- OOS start: `2022-01-01`

## Test metodolojisi

### Model

Önceki Tail-Risk × Alpha Conditional Sizing V1 ile aynı model stack'i kullanılır:

- Alpha: `Ridge(alpha=10)`; hedef `ret_pct` clipped `[-25,+25]`
- Tail risk: üç `LogisticRegression(C=0.25, class_weight=balanced)`
- Risk score: `0.25×tail5 + 0.35×tail10 + 0.40×tail15`
- Feature set: yalnızca signal-date strict değişkenler

### Zaman bölmesi

- FIT: `signal_date < 2022-01-01` ve `exit_date < 2022-01-01`
- OOS: `signal_date >= 2022-01-01`

Holdout'a göre threshold/model tuning yapılmaz.

### Portfolio replay

OOS test gerçek production mekanikleriyle yeniden yürütülür:

- 4 tranche
- 5 günlük stagger
- 21 günlük refresh
- mevcut maliyet mantığı
- risk parity
- cash mechanics
- catastrophe stop

Araştırma wrapper'ı yalnızca yeni BUY amount'ı küçültür. Sells, exits, stops ve seçim sıralaması değiştirilmez.

## Çıktılar

`research/results_asymmetric_tail_risk_v1/` altında:

- `gmq_asymmetric_tail_risk_allocation_raporu.md` — ana rapor
- `..._metrics.csv` — BASE ve asymmetric OOS portföy sonuçları
- `..._stress.csv` — `%0.35 / 0.50 / 0.75 / 1.00 / 1.25` RT cost stress
- `..._monthly.csv` — eşleştirilmiş aylık farklar
- `..._trades.csv` — OOS trade + alpha/risk + quadrant + multiplier
- `..._predictions.csv` — model tahminleri
- `..._model_metrics.csv` — model metrikleri
- `..._policy_usage.csv` — multiplier kullanım oranları
- `..._quadrants.csv` — dört alpha/risk bölgesinin ekonomik davranışı
- `..._alpha_risk_matrix.csv` — 5×5 alpha/risk matrisi
- `..._market.csv` — BIST / US kırılımı

## Karar kuralı

Tek başına MaxDD'nin düşmesi yeterli değildir.

Öncelik sırası:

1. OOS CAGR ve toplam ekonomik büyüme
2. MaxDD / worst-12M
3. Profit Factor
4. Eşleştirilmiş aylık bootstrap
5. Maliyet stresinde yönün korunması
6. Danger quadrant'ın gerçek loss-capture değeri ve kaçırılan pozitif getiriler

Özellikle şu kontrol kritiktir:

`LOW_ALPHA_HIGH_RISK` grubu yüksek pozitif getiri de taşıyorsa, agresif asymmetric haircut ekonomik olarak yanlış olabilir.

## Çok önemli

Bu araştırma başarılı çıksa bile **otomatik canlı entegrasyon yapmaz**. Önce sonuç dosyası incelenir; ardından ayrı bir doğrulama kararı verilir.

**RESEARCH ONLY — PRODUCTION CHANGE YOK.**
