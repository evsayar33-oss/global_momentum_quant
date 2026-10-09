# Önceki V4.2 artifact incelemesi

## Kaynak doğrulaması

- GitHub Actions run: https://github.com/evsayar33-oss/global_momentum_quant/actions/runs/37974041171
- Artifact: `gmq-momentum-entry-timing-chase-avoidance-v4-2-diagnostic-only`
- Artifact SHA-256: `b899dc8e512686b1ccfc04ad283211b8065656b28fb6810231f297afae21cd70`
- İndirilen ZIP üzerinde yapılan yerel SHA-256 hesaplaması aynı değeri verdi.
- Workflow adımları ve manifest: artifact audit başarılı; yeni BASE replay yapılmadı (`historical_backtest_run: false`).

## Tarihsel örneklem sayıları

| Ölçüm | V2 | V3 | V4.1 |
|---|---:|---:|---:|
| Tahmin satırı, tam arşiv | 8.842 | 8.842 | 3.432 |
| Sinyal tarihi aralığı | 2018-01-03–2026-09-30 | 2018-01-03–2026-09-30 | 2018-01-02–2026-09-15 |
| Tam arşivde benzersiz kimlik | 8.330 | 8.330 | 3.432 |
| Tekrarlanan kimlik satırı | 512 | 512 | 0 |
| Chase adayı | 75 | 75 | 16 |
| Tehlikeli / sağlıklı | arşivde kayıtlı değil | 31 / 44 | 13 / 3 |
| Ham closed-cycle sayısı | arşivde kayıtlı değil | arşivde kayıtlı değil | 4.994 (tarih filtresi öncesi raporlanan) |
| Feature satırı, start filtresi öncesi | arşivde kayıtlı değil | arşivde kayıtlı değil | 4.994 |

Ortak sinyal-tarihi penceresi 2018-01-03–2026-09-15 olarak hesaplandığında V2 ve V3'te 8.825'er, V4.1'de 3.430 satır bulunuyor. Tarih penceresini eşitlemek farkı açıklamıyor.

V2 ve V3'te 512 dört alanlı kimlik grubunun her birinde iki satır var. Bu satırlar aynı satırın birebir kopyası değil: aynı `market,ticker,signal_date,entry_date` altında `exit_date` ve `ret_pct` değişebiliyor. Eski CSV'de `tranche` alanı olmadığı için bunların tam ekonomik kimliği geriye dönük olarak doğrulanamıyor.

Ortak tarih penceresinde V2/V3 eski dört alanlı kimlikleri ile V4.1'in aynı dört alana indirgenmiş kimlikleri arasındaki kesişim 6. Bu yalnızca kimlik kesişimi duyarlılık ölçümüdür; V4.1 `tranche` alanını eklediği ve sinyal tarihi gerçek BUY emrinden alındığı için tam eşdeğerlik testi değildir.

## Koddan görülen tanım farkı

- V2/V3'ün kullandığı eski normalize etme yolu `tranche` alanını korumuyor. `signal_date` ham trade kaydında yoksa giriş tarihinden önceki piyasa barını tahmin ediyor.
- V4.1, in-memory BASE replay sırasında BUY emir sinyalini ve ertesi fill olayını yakalıyor; sınıflandırılan döngüde `market,ticker,signal_date,entry_date,tranche` alanları var.
- `engine.py`, aynı pozisyon yeni bir dilimde devam ettiğinde `döngü yenilendi (devam)` gerekçesiyle bir cycle/trade kaydı üretebiliyor. Bu satırın yeni bir BUY fill'i olmadan legacy trade evrenine girmesi mümkündür.

Bu bulgular örneklemlerin tanım olarak farklı olduğunu gösterir. Bu farkların her birinin kaç satır ürettiği, yalnızca önceki V4.2 artifact audit'inden kesinleştirilemez. Bunun için aynı güncel BASE replay üzerinde legacy ve execution-verified datasetleri üretmek gerekir; yeni V4.2.1 replay paketi bunu yapar.

## Maliyet stresi ve sınıflandırma

- V3 `DANGEROUS_SOFT`: eski raporda beş maliyet seviyesinin CAGR farkı 0.000 pp.
- V4.1 `DANGEROUS_SOFT`: beş maliyet seviyesinin CAGR farkı da BASE'e karşı negatif; farklar yaklaşık -0.115475 ile -0.021644 yüzde puan aralığında.
- V4.1 tehlikeli/sağlıklı sınıfları 13/3. Sınıflar arası karşılaştırma için örneklem yetersiz.
- V4.1 `execution_audit.csv` beklenen ve uygulanmış sizing olaylarını eşleştiriyor; ham emir → fill → closed-cycle defteri içermiyor. %100 sizing eşleşme oranı ham işlem eşleştirmesinin %100 doğruluğunu kanıtlamaz.

## Sonuç

V4.2 artifact kendi amacında başarılıdır: önceki sonuçları toplar ve kanıtlanmayan sayıları sıfır olarak doldurmaz. Ancak örneklem farkının tam sayısal kök nedenini henüz belirlemez; çünkü yeni BASE replay çalıştırmamıştır. Yeni V4.2.1 workflow'u önce bu eksik replay'i yapacak ve policy sizing uygulamadan aynı BASE evrenindeki iki dataset tanımını karşılaştıracaktır.
