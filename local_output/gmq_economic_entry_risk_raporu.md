# Global Momentum Quant — Economic Entry Risk / Adverse Severity
_Üretim: 2026-10-06 19:49 · source: prior Early-Adverse trade dataset · RT benchmark: 0.35%_

## 1. Amaç
Önceki Early-Adverse Predictor'ın yakaladığı olayları `healthy`, `temporary_adverse` ve `harmful_adverse` olarak ayırıp, sinyal günündeki bilgilerin ekonomik açıdan zararlı adverse path'i öngörüp öngöremediği test edildi.
Bu çalışma canlı sistemi değiştirmez ve tam portföy A/B backtesti değildir.

## 2. Path dağılımı
- Toplam işlem: **12944**
- Healthy: **59.02%**
- Temporary adverse: **17.71%**
- Harmful adverse: **23.26%**
- Tail harmful (<= -5%): **13.23%**

- BASE tüm işlemler win rate: **55.60%**
- BASE tüm işlemler ortalama getiri: **2.73%**
- Temporary adverse ortalama getiri: **9.32%**
- Harmful adverse ortalama getiri: **-8.46%**
- Oracle olarak harmful işlemlerin tamamı önceden atlatılabilse teorik trade-level negatif getiri kaçınması: **25477.08 puan**

## 3. Zararlı adverse predictor doğrulaması
TRAIN harmful AUC: **0.5719**
HOLDOUT harmful AUC: **0.5619**
TRAIN harmful PR-AUC: **0.2881**
HOLDOUT harmful PR-AUC: **0.2616**
HOLDOUT Brier: **0.1848**

## 4. Holdout risk filtresi

|   threshold |   skipped_pct |   kept_n |   skipped_n |   kept_win_pct |   skipped_win_pct |   kept_mean_ret_pct |   skipped_mean_ret_pct |   kept_mean_after_0p35_cost_pct |   skipped_positive_pct |   temporary_skipped_pct |   harmful_total_n |   harmful_skipped_n |   harmful_capture_pct |   harmful_loss_total_pp |   harmful_loss_skipped_pp |   harmful_loss_capture_pct |   kept_harmful_pct |   skipped_harmful_pct |   selection_delta_if_skipped_return_0_pp | scenario   |
|------------:|--------------:|---------:|------------:|---------------:|------------------:|--------------------:|-----------------------:|--------------------------------:|-----------------------:|------------------------:|------------------:|--------------------:|----------------------:|------------------------:|--------------------------:|---------------------------:|-------------------:|----------------------:|-----------------------------------------:|:-----------|
|      0.3575 |       25.8085 |     3533 |        1229 |        57.5432 |           55.0854 |              3.7101 |                 3.3703 |                          3.3601 |                55.0854 |                 20.5858 |              1076 |                 330 |               30.6691 |              10375.5500 |                 3446.6300 |                    33.2188 |            21.1152 |               26.8511 |                                  -0.8698 | Q75        |
|      0.3833 |       14.4057 |     4076 |         686 |        57.4828 |           53.4985 |              3.6729 |                 3.3221 |                          3.3229 |                53.4985 |                 18.9504 |              1076 |                 196 |               18.2156 |              10375.5500 |                 2286.2000 |                    22.0345 |            21.5898 |               28.5714 |                                  -0.4786 | Q90        |

Yorum: `selection_delta_if_skipped_return_0_pp` pozitifse, yalnızca trade-level ve sıfır getirili nakit varsayımı altında seçimin ekonomik yönde iyileşme işareti vardır. Bu portföy CAGR testi değildir.

## 5. Holdout risk binleri

|   risk_bin |   n |   pred_harmful_mean |   harmful_rate_pct |   temporary_rate_pct |   win_rate_pct |   mean_ret_pct |   mean_loss_pct |   tail_harmful_rate_pct |
|-----------:|----:|--------------------:|-------------------:|---------------------:|---------------:|---------------:|----------------:|------------------------:|
|          0 | 832 |              0.2220 |            17.0673 |              15.2644 |        58.8942 |         5.6088 |         -9.0707 |                 11.2981 |
|          1 | 585 |              0.2855 |            19.8291 |              15.8974 |        58.2906 |         3.4825 |         -8.1259 |                 12.8205 |
|          2 | 528 |              0.3049 |            17.9924 |              15.9091 |        60.4167 |         4.0149 |         -7.3622 |                 10.7955 |
|          3 | 425 |              0.3177 |            24.9412 |              18.8235 |        55.7647 |         3.6604 |         -7.2215 |                 14.8235 |
|          4 | 344 |              0.3283 |            23.2558 |              18.0233 |        56.3953 |         2.9063 |         -7.8807 |                 13.9535 |
|          5 | 346 |              0.3377 |            25.4335 |              17.9191 |        54.6243 |         1.7745 |         -7.9388 |                 17.0520 |
|          6 | 306 |              0.3472 |            23.5294 |              16.9935 |        55.5556 |         2.5303 |         -8.2646 |                 14.3791 |
|          7 | 334 |              0.3578 |            24.5509 |              20.6587 |        57.4850 |         2.5793 |         -7.6311 |                 14.9701 |
|          8 | 376 |              0.3729 |            26.3298 |              23.9362 |        56.1170 |         3.5693 |         -8.3899 |                 16.4894 |
|          9 | 686 |              0.4388 |            28.5714 |              18.9504 |        53.4985 |         3.3221 |        -12.3232 |                 17.6385 |

## 6. BIST / US ayrı doğrulama

| market   |    n |   harmful_auc |   harmful_pr_auc |   harmful_rate_pct |   temporary_rate_pct |   win_rate_pct |   mean_ret_pct |
|:---------|-----:|--------------:|-----------------:|-------------------:|---------------------:|---------------:|---------------:|
| bist     | 2288 |        0.5283 |           0.2530 |            23.8636 |              16.3899 |        56.5997 |         4.3957 |
| us       | 2474 |        0.5851 |           0.2778 |            21.4228 |              19.1593 |        57.1948 |         2.9073 |

## 7. Model katsayıları

| class             | feature                  |   coefficient_z |   abs_coefficient |
|:------------------|:-------------------------|----------------:|------------------:|
| harmful_adverse   | vol20_ann_pct            |         -0.1386 |            0.1386 |
| harmful_adverse   | ret_63d_pct              |          0.0957 |            0.0957 |
| harmful_adverse   | gap_atr                  |          0.0566 |            0.0566 |
| harmful_adverse   | gap_pct                  |          0.0523 |            0.0523 |
| harmful_adverse   | market_median_21d_pct    |         -0.0517 |            0.0517 |
| harmful_adverse   | score_pctile             |         -0.0511 |            0.0511 |
| harmful_adverse   | stock_vs_market_21d_pp   |          0.0318 |            0.0318 |
| harmful_adverse   | market_breadth_21d       |          0.0299 |            0.0299 |
| harmful_adverse   | ret_5d_pct               |          0.0278 |            0.0278 |
| harmful_adverse   | momentum_extension_z     |         -0.0233 |            0.0233 |
| harmful_adverse   | market_breadth_5d        |         -0.0229 |            0.0229 |
| harmful_adverse   | atr14_pct                |          0.0198 |            0.0198 |
| harmful_adverse   | market_median_63d_pct    |         -0.0155 |            0.0155 |
| harmful_adverse   | market_dispersion_1d_pct |          0.0098 |            0.0098 |
| harmful_adverse   | market_median_1d_pct     |          0.0033 |            0.0033 |
| harmful_adverse   | ret_21d_pct              |          0.0015 |            0.0015 |
| healthy           | gap_atr                  |         -0.1809 |            0.1809 |
| healthy           | ret_5d_pct               |         -0.1295 |            0.1295 |
| healthy           | market_breadth_21d       |         -0.1245 |            0.1245 |
| healthy           | vol20_ann_pct            |          0.1145 |            0.1145 |
| healthy           | momentum_extension_z     |          0.1085 |            0.1085 |
| healthy           | ret_63d_pct              |         -0.0908 |            0.0908 |
| healthy           | market_median_1d_pct     |          0.0763 |            0.0763 |
| healthy           | gap_pct                  |         -0.0742 |            0.0742 |
| healthy           | market_breadth_5d        |          0.0717 |            0.0717 |
| healthy           | market_median_21d_pct    |          0.0656 |            0.0656 |
| healthy           | atr14_pct                |          0.0641 |            0.0641 |
| healthy           | market_median_63d_pct    |          0.0604 |            0.0604 |
| healthy           | market_dispersion_1d_pct |         -0.0522 |            0.0522 |
| healthy           | stock_vs_market_21d_pp   |         -0.0371 |            0.0371 |
| healthy           | score_pctile             |          0.0334 |            0.0334 |
| healthy           | ret_21d_pct              |          0.0010 |            0.0010 |
| temporary_adverse | gap_atr                  |          0.1243 |            0.1243 |
| temporary_adverse | ret_5d_pct               |          0.1017 |            0.1017 |
| temporary_adverse | market_breadth_21d       |          0.0946 |            0.0946 |
| temporary_adverse | momentum_extension_z     |         -0.0852 |            0.0852 |
| temporary_adverse | atr14_pct                |         -0.0839 |            0.0839 |
| temporary_adverse | market_median_1d_pct     |         -0.0796 |            0.0796 |
| temporary_adverse | market_breadth_5d        |         -0.0488 |            0.0488 |
| temporary_adverse | market_median_63d_pct    |         -0.0449 |            0.0449 |
| temporary_adverse | market_dispersion_1d_pct |          0.0424 |            0.0424 |
| temporary_adverse | vol20_ann_pct            |          0.0241 |            0.0241 |
| temporary_adverse | gap_pct                  |          0.0220 |            0.0220 |
| temporary_adverse | score_pctile             |          0.0177 |            0.0177 |
| temporary_adverse | market_median_21d_pct    |         -0.0139 |            0.0139 |
| temporary_adverse | stock_vs_market_21d_pp   |          0.0053 |            0.0053 |
| temporary_adverse | ret_63d_pct              |         -0.0049 |            0.0049 |
| temporary_adverse | ret_21d_pct              |         -0.0025 |            0.0025 |

## 8. Karar çerçevesi
Canlı Entry Risk Quality filtresi ancak aynı anda şu kanıtlar oluşursa düşünülebilir:
1. Holdout harmful AUC anlamlı ve train'e göre makul ölçüde korunuyor.
2. Risk yükseldikçe holdout harmful rate belirgin biçimde artıyor.
3. Q75/Q90 ile atlanan grubun ortalama getirisi açık biçimde daha kötü.
4. Pozitif temporary-adverse işlemlerin aşırı büyük kısmı filtreyle atılmıyor.
5. Harmful loss capture ekonomik olarak anlamlı.
6. BIST ve US sonuçları aynı yönde.
7. Daha sonra tam üretim portföyünde bağımsız A/B + 0.35/0.50/0.75/1.00/1.25 maliyet stresi + holdout doğrulaması yapılmalı.

**Bu test pozitif çıksa bile doğrudan canlı entegrasyon yapılmaz.**
