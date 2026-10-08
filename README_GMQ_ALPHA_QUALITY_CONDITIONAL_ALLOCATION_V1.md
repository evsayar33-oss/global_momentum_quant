# GMQ — Alpha-Quality Conditional Allocation V1

## Amaç

Önceki araştırma dizisinde tail-risk sinyalinin gerçek bir sol-kuyruk sıralama gücü olduğu görüldü; ancak tail-risk'i doğrudan pozisyon küçültmeye bağlamak CAGR'ı sistematik olarak düşürdü. Ayrıca OOS alpha×risk matrisinde **LOW_ALPHA + LOW_RISK** bölgesi en zayıf gözlenen bölge iken **HIGH_ALPHA + HIGH_RISK** ve **LOW_ALPHA + HIGH_RISK** bölgeleri pozitif kalabildi.

Bu nedenle bu paket tail-risk'i cezalandırmak yerine **alpha kalitesine dayalı sermaye tahsisini** test eder.

## Test edilen iki hipotez

### A) PURE_ALPHA
Yalnızca market içindeki **bottom 25% alpha percentile** işlemleri küçültülür.

### B) ALPHA_X_LOW_RISK
Yalnızca:

`alpha percentile <= 25%` **VE** `tail-risk percentile <= 50%`

olan işlemler küçültülür.

B ailesi özellikle daha önce zayıf gözlenen LOW_ALPHA + LOW_RISK bölgesini hedefler; LOW_ALPHA + HIGH_RISK işlemlerine dokunmaz.

## Sabit politikalar

| Policy | Family | Koşul | BUY multiplier |
|---|---|---|---:|
| BASE | BASE | yok | 1.00 |
| ALPHA_GENTLE | PURE_ALPHA | alpha <= Q25 | 0.85 |
| ALPHA_BALANCED | PURE_ALPHA | alpha <= Q25 | 0.70 |
| ALPHA_STRONG | PURE_ALPHA | alpha <= Q25 | 0.50 |
| COND_GENTLE | ALPHA_X_LOW_RISK | alpha <= Q25 & risk <= Q50 | 0.85 |
| COND_BALANCED | ALPHA_X_LOW_RISK | alpha <= Q25 & risk <= Q50 | 0.70 |
| COND_STRONG | ALPHA_X_LOW_RISK | alpha <= Q25 & risk <= Q50 | 0.50 |

Bunlar önceden tanımlıdır. OOS üzerinde yeni threshold veya multiplier araması yapılmaz.

## Model ve leakage kontrolü

- FIT cutoff: `2022-01-01`
- FIT'e yalnızca `signal_date < 2022-01-01` ve `exit_date < 2022-01-01` işlemleri girer.
- OOS: `signal_date >= 2022-01-01`
- Alpha: `Ridge(alpha=10)`; target `ret_pct` clipped to `[-25%, +25%]`.
- Tail risk: sabit `LogisticRegression(C=0.25, class_weight='balanced')` ile `<= -5%`, `<= -10%`, `<= -15%` hedefleri.
- Tail-risk skoru: `0.25*p_tail5 + 0.35*p_tail10 + 0.40*p_tail15`.
- Alpha ve risk percentile'ları market bazında ve yalnızca FIT tahmin dağılımından hesaplanır.
- Allocation feature seti strict signal-date feature'larıyla sınırlıdır.
- Future/path değişkenleri sizing kararına verilmez.

## Production-path kuralları

Araştırma önce gerçek `engine.MarketData` + `portfolio.process_day` yolu ile BASE trade history üretir. Sonra OOS'ta yalnızca research wrapper ile yeni BUY amount'ları ölçekler.

Değiştirilmez:

- hisse seçimi
- ranking / score
- tranche zamanlaması
- hold / refresh mekanizması
- satışlar
- stop mekanikleri
- catastrophe stop
- aylık risk parity
- cash / FX mekanikleri
- production state/order/NAV dosyaları
- canlı `config.py`, `engine.py`, `portfolio.py`, `signals.py`

## Gerekli dosyalar

Repo kökünde aşağıdakiler bulunmalıdır:

- `gmq_alpha_quality_conditional_allocation_v1.py`
- `gmq_alpha_quality_requirements.txt`
- `gmq_tail_risk_alpha_conditional_sizing.py`  ← production-path research helper
- `.github/workflows/gmq_alpha_quality_conditional_allocation_v1.yml`

Bu test production dosyalarının yerine geçmez.

## Çalıştırma

### GitHub Actions

`Actions → GMQ Alpha-Quality Conditional Allocation V1 → Run workflow`

Workflow yalnızca `workflow_dispatch` ile çalışır; otomatik schedule oluşturmaz.

### Lokal

```bash
pip install -r gmq_alpha_quality_requirements.txt
python gmq_alpha_quality_conditional_allocation_v1.py --synthetic
python gmq_alpha_quality_conditional_allocation_v1.py --years 14
```

## Çıktılar

`research/results_alpha_quality_conditional_allocation_v1/` altında:

- `gmq_alpha_quality_conditional_allocation_v1_raporu.md`
- `gmq_alpha_quality_conditional_allocation_v1_metrics.csv`
- `gmq_alpha_quality_conditional_allocation_v1_monthly.csv`
- `gmq_alpha_quality_conditional_allocation_v1_trades.csv`
- `gmq_alpha_quality_conditional_allocation_v1_predictions.csv`
- `gmq_alpha_quality_conditional_allocation_v1_model_metrics.csv`
- `gmq_alpha_quality_conditional_allocation_v1_policy_usage.csv`
- `gmq_alpha_quality_conditional_allocation_v1_stress.csv`
- `gmq_alpha_quality_conditional_allocation_v1_alpha_bins.csv`
- `gmq_alpha_quality_conditional_allocation_v1_risk_bins.csv`
- `gmq_alpha_quality_conditional_allocation_v1_quadrants.csv`
- `gmq_alpha_quality_conditional_allocation_v1_market.csv`

## Karar standardı

Bir aday yalnızca MaxDD'yi iyileştirdi diye canlıya alınmaz.

Öncelikli kontroller:

1. OOS CAGR değişimi
2. MaxDD ve worst-12M
3. Profit Factor
4. Eşleştirilmiş aylık bootstrap
5. `%0.35 → %1.25` maliyet stresi
6. BIST / US davranışının tutarlılığı
7. Weak quadrant gerçekten iyileşiyor mu?
8. Allocation değişimi yeterince küçük ve ekonomik olarak anlamlı mı?

### Beklenen araştırma kararı

- CAGR anlamlı düşüyor ve bootstrap aleyhteyse: **REJECT**
- CAGR yaklaşık korunurken DD/PF/worst-12M iyileşiyorsa: **CONDITIONAL PASS adayı**
- Sadece tek bir küçük alt grupta iyileşme varsa: production entegrasyonu yapılmaz; yeni validation gerekir.

**CANLI SİSTEME DEĞİŞİKLİK YAPILMAZ.**
