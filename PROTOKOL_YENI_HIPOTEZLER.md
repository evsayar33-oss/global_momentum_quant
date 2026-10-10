# GMQ — Kapsamlı Yeni Hipotezler Test Protokolü (v1.4)

**Belge türü:** Önceden tanımlı araştırma ve doğrulama protokolü. `gmq_new_hypotheses_test.py` bu belgeyi uygular.
**v1.0 → v1.1:** Orijinal plandaki (GMQ_Kapsamli_Yeni_Hipotezler_Test_Plani.pdf) bütün kurallar geçerlidir.
Bu sürümde 4 ek madde (§A–§D) ve bunların gerektirdiği somut tanımlar eklendi.
**Karar ilkesi (değişmedi):** Bağımsız kilitli testte doğrulanmış ve risk sınırlarına uyan iyileşme yoksa hiçbir aday üretime alınmaz.

---

## 1. Değişmeyen çerçeve (v1.0'dan özet)
- Üretim dosyaları değiştirilmez. Test, GMQ'nun **kendi motoru** (`engine.step_market` + `portfolio.process_day`) ile
  **tam portföy yeniden oynatımı** olarak yürütülür: emir → ertesi açılış fill'i → nakit → 4 dilim → felaket stopu → risk paritesi.
  Adaylar motora yalnızca bellekte sarmalanarak (hisse seçimi veya yeni dilim bütçesi) etki eder.
- Her özellik yalnızca sinyal günü kapanışında bilinen veriyle hesaplanır. Kur verisi BIST için bir gün gecikmeli (t−1) kullanılır
  (Yahoo kur kapanışı BIST kapanışından sonra oluşur).
- Eşikler yalnızca eğitim penceresinden öğrenilir ve dondurulur. Aynı hipotezin eşik varyantları üretilmez.
- Risk paritesi için kullanılan gölge seriler tüm adaylarda ortak (BASE) tutulur; fark yalnızca adayın kendi etkisidir.

## 2. Zaman bölmesi (v1.1'de genişletildi)
| Bölüm | Süre | Kullanım |
|---|---|---|
| Eğitim | başlangıç → doğrulama başı (≈7 yıl) | eşik ve model öğrenimi |
| Doğrulama | 30 ay | aday karşılaştırma ve en fazla 1 aday seçimi |
| **Kilitli test** | **son 36 ay** | seçilen tek aday ile BASE, bir kez |

- Aday taraması kilitli test başlangıcında **durdurulur**; kilitli pencere verisi hiçbir aday için işlenmez.
- Doğrulamada hiçbir aday seçim ölçütlerini geçemezse **kilitli pencere açılmaz** ve ileride yapılacak testler için korunur.
- Pencere sınırlarında 32 takvim günü (21 işlem günü tutma + giriş gecikmesi) embargo uygulanır.

## 3. Aday havuzu (her hipotez için tek, önceden sabit tanım)
| Kod | Tip | Tanım |
|---|---|---|
| H1 | piyasa kapısı | Evrende 200 günlük ortalamanın üstündeki hisse payı < eğitim %20'si → yeni dilim bütçesi ×0.5 |
| H2 | skor | BIST skoruna piyasadan arındırılmış momentum (resid_mom) 4. bileşen olarak eklenir. ABD zaten resid_mom kullandığı için ABD'de **uygulanmaz** (mevcut mantıkla özdeş) |
| H3 | filtre | 21g/126g oynaklık oranı kesitsel %80 üstü aday elenir, sıradaki isimle doldurulur |
| H4 | sıralama | 0.75 × üretim skoru + 0.25 × katılım (ciro medyan/MAD robust z + 10 günlük süreklilik) |
| H5 | filtre | Gece boşluğu riski (63g gap oynaklığı + toplam oynaklığa oranı) kesitsel %80 üstü aday elenir |
| H6 | model filtresi | Yarışan risk: girişten sonra önce −1.5 ATR mi, +1.5 ATR mi (10 gün). Lojistik model yalnızca eğitimde kurulur; tahmini risk eğitim %80 üstü aday elenir |
| H7 | piyasa kapısı | BIST: USD/TRY 21g değişimi eğitim %90 üstü (kur şoku) → ×0.5. ABD için anlamlı çapraz bağlam yok (**uygulanmaz**) |
| H8 | piyasa kapısı (§E) | Momentum çöküş durumu (Daniel–Moskowitz): piyasa 24 ay negatif VE 126g oynaklık eğitim %70 üstü → ×0.5 |
| H9 | boyut ölçekleme (§E) | Oynaklığa göre ölçekleme (Barroso–Santa-Clara): kodda bulunan ama kapalı `VOL_TARGET`, üretim parametreleriyle |

Filtrelerde elenen isimler sıradaki adaylarla doldurulur; işlem sayısı yapay olarak azaltılmaz.

---

## §A. Güç analizi ve ölçülebilir en küçük etki (v1.1 madde 1)
- Test başlamadan önce, BASE'in eğitim dönemi işlemlerinden ve her adayın doğrulamadaki **eşleşik** (aynı aylar) fark
  oynaklığından, kilitli pencere uzunluğunda **ölçülebilir en küçük etki (MDE)** hesaplanır:
  `MDE = (z₀.₉₅ + z₀.₈₀) × SE_fark ≈ 2.49 × SE_fark` (tek yönlü %5, güç %80).
- MDE, raporda her adayın yanında gösterilir. Gözlenen fark MDE'nin altındaysa kilitli testin bu farkı güvenle ayırt
  edemeyeceği baştan bilinir; böyle bir sonuç "iyileşme yok" değil, "ölçülemez kadar küçük" olarak yorumlanır.
- Kilitli pencere 36 aya genişletildi (v1.0'da verinin son %20'si ≈ 2.5 yıl).

## §B. Çoklu karşılaştırma düzeltmesi (v1.1 madde 2)
- Doğrulamada her aday için tek yönlü bootstrap p-değeri (3 aylık hareketli blok, BASE ile **ortak** ay örneklemesi, B = 2000).
- **Holm–Bonferroni** düzeltmesi uygulanır; seçim için Holm-düzeltilmiş p ≤ 0.20 gerekir. Eşik bilinçli olarak gevşektir,
  çünkü kilitli test ikinci ve bağımsız sınavdır; aile bazında hata kontrolü Holm ile sağlanır.
- Ek olarak **White Reality Check** p-değeri raporlanır: "en iyi görünen adayın, denenen aday sayısı düşünüldüğünde
  şans eseri bu kadar iyi görünme olasılığı".

## §C. Plasebo kontrolü (v1.1 madde 3)
- Her aday tipi için aynı motor ve aynı pencerelerde rastgele politikalar çalıştırılır (varsayılan tip başına 6):
  - **filtre plasebosu:** adayların rastgele %20'si elenir ve doldurulur
  - **sıralama plasebosu:** 0.25 ağırlıklı rastgele bileşen eklenir (H2 ve H4 ile eşleşir)
  - **kapı plasebosu:** H1/H8'in gerçek aktiflik payıyla, ay bazında rastgele ×0.5 (H1, H7, H8, H9 ile eşleşir)
- Bir adayın seçilebilmesi için doğrulama farkı kendi tipindeki plasebo dağılımının **%90'ını aşmalıdır**.
- Plasebo sonuçları, gürültünün bu sistemde ne kadar iyi görünebileceğinin doğrudan ölçüsü olarak raporlanır.

## §D. Hayatta kalma yanlılığı: ölç, azalt, raporla (v1.1 madde 4)
- v1.0'daki "yanlılık varsa kalite kapısı başarısız" kuralı, ücretsiz veriyle her zaman INCONCLUSIVE sonucu doğuracağından
  şu şekilde değiştirildi:
  - **ABD:** Wikipedia S&P 500 değişiklik tablosundan **geçmiş üyelik** çıkarılır; her tarihte yalnızca o gün üye olan
    hisseler seçilebilir. Endeksten çıkmış hisseler de indirilmeye çalışılır. Yahoo'da verisi olmayanlar eksik kalır (raporlanır).
    Tablo alınamazsa yanlılık DISCLOSED olarak işaretlenir.
  - **BIST:** Ücretsiz geçmiş liste yoktur. İşlemden kalkmış hisse sayısı ve yıllara göre kapsama ölçülür ve DISCLOSED olarak raporlanır.
  - Aday ve BASE **aynı evrende** kıyaslandığı için fark testi, seviye sonuçlarına göre yanlılıktan çok daha az etkilenir.
- Kalite kapısı yalnızca gerçek veri/oynatma hatalarında başarısız olur: tekrar eden kayıt, geçersiz OHLC (>%1),
  tarih sırası, kur kapsaması (<%95), look-ahead testi, motor belirlenimciliği, işlem uzlaştırma hatası.

## §E. Literatür eklemeleri
- H8 ve H9, momentum stratejilerinde kanıtı en güçlü iki risk konusu (momentum çöküşleri ve oynaklığa göre ölçekleme) için eklendi.

---

## 4. Seçim ölçütleri (doğrulama, hepsi gerekli)
1. Holm-düzeltilmiş p ≤ 0.20 (işlem başı net beklenti farkı > 0)
2. Eğitim penceresinde de fark > 0 (yön tutarlılığı)
3. Doğrulama farkı, kendi tipindeki plasebo dağılımının %90'ından büyük
4. En büyük düşüş BASE'ten %5'ten fazla kötü değil
5. İşlem sayısı BASE'in en az %70'i
6. Sharpe BASE'ten 0.05'ten fazla düşük değil

Geçenler arasından doğrulama t-değeri en yüksek **tek** aday kilitli teste gider.

## 5. PASS kapısı (kilitli test, hepsi gerekli; v1.0 §8 ile aynı)
1. İşlem başı net fark > 0 ve blok-bootstrap %95 güven aralığının alt sınırı > 0
2. Ekonomik anlam: BASE beklentisine göre en az %10 göreli artış
3. Win rate en az +2 puan **veya** Sharpe +0.10 ve yıllık getiri BASE'ten düşük değil
4. En büyük düşüş %5'ten fazla kötüleşmez; CVaR %10'dan, kâr faktörü %5'ten fazla bozulmaz
5. 1.5× maliyette iyileşme yönü korunur (2× ayrıca raporlanır)
6. Örneklem: kilitli dönemde ≥300 işlem, adayın uygulandığı her piyasada ≥100 işlem (yoksa INCONCLUSIVE)
7. Tek piyasada geçerli aday "yalnızca o piyasa" kapsamıyla raporlanır

**PASS** bile doğrudan üretim değildir; önce gölge çalışma gerekir.

## 6. Çıktılar
`final_decision_report.md`, `candidate_comparison.csv`, `locked_test_base_vs_winner.csv`, `market_regime_breakdown.csv`,
`execution_reconciliation.csv`, `cost_stress.csv`, `data_quality_report.csv`, `manifest.json` (commit, bölme, eşikler,
SHA-256), `reproducibility_notes.md` + v1.1 ekleri: `power_analysis.csv`, `multiple_testing.csv`, `placebo_results.csv`,
`yearly_stability_pretest.csv`, `h6_model_diagnostics.csv`.

## 7. Bilinen sınırlamalar
- Yahoo verisi sonradan revize olabilir; Wikipedia değişiklik tablosu eksik olabilir.
- Üretim stratejisi 2014–2026'nın tamamı görülerek tasarlandığı için kilitli pencere BASE için tamamen "görülmemiş" değildir; adaylar için öyledir.
- Gerçekçi beklenti: çoğu adayın FAIL ya da INCONCLUSIVE çıkması. Bu, canlı sistemi korumuş olmak demektir.

---

## §F. v1.2 — ABD kolu modu (`--mode us`, USD bazlı)

**Gerekçe:** v1.1 sonucu (9 Eki 2026): 9 adayın hiçbiri karma portföyde BASE'i geçemedi; kilitli pencere açılmadı.
Dolar bazında referans backtestte yalnızca ABD kolu yıllık %17.8, karma portföy %16.2 verdi (TL rakamları kur nedeniyle yanıltıcı).

**Değişen tanımlar (diğer tüm kurallar v1.1 ile aynı):**
- **BASE = %100 ABD** (üretim ABD kolu, `FIXED_WEIGHTS = {bist: 0, us: 1}`). Adaylar yalnızca ABD koluna uygulanır;
  H2 ve H7 (yalnızca BIST'e özgü) bu modda yoktur.
- **Portföy ölçümleri USD bazındadır** (yıllık getiri, oynaklık, Sharpe, en büyük düşüş). İşlem başı net beklenti zaten USD'dir.
- **Hedef:** 2014–2019'un zayıf dönemini "düzeltmek" değil, **hem eğitim (zayıf dönem dahil) hem doğrulama dönemlerinde**
  BASE'i geçmektir. Aday seçim ölçütü 2 (eğitimde de fark > 0) bunu zorunlu kılar. Belirli bir dönemi düzeltmeye yönelik ayar yapılmaz.
- Bölme ve kilitli pencere v1.1 ile aynı. Kilitli pencere v1.1 çalışmasında **açılmadığı** için temizdir.

**Soru 0 — %100 ABD mi, karma (BIST + ABD) mi?** (önceden kayıtlı, adaylardan ayrı soru)
- Ölçüt: USD bazlı portföy Sharpe'ı. Hem eğitimde hem doğrulamada %100 ABD Sharpe ≥ karma Sharpe ise soru kilitli teste gider;
  aksi halde karma portföy korunur ve kilitli pencere bu soru için açılmaz.
- Kilitli test kapıları (hepsi gerekli): %100 ABD Sharpe ≥ karma, yıllık getiri ≥ karma, en büyük düşüş karmadan 5 puandan fazla kötü değil.
  Aylık getiri farkının blok-bootstrap %95 güven aralığı raporlanır.
- Sonuç olumlu olsa bile önce gölge çalışma yapılır; üretim ağırlığı doğrudan değiştirilmez.
- Bu soru ile (varsa) seçilen tek aday aynı kilitli çalıştırmada, birer kez değerlendirilir.

**Ek çıktı:** `q0_us100_vs_karma.csv`.

---

## §G. v1.3 — Kazananların anatomisi (`--mode anatomy`, karma portföy, USD bazlı)

**Gerekçe:** v1.1 (karma) ve v1.2 (ABD kolu) sonuçları: 16 literatür adayından hiçbiri BASE'i geçemedi; v1.2'de karma
portföy %100 ABD'den daha iyi çıktı (USD Sharpe eğitim 1.06 vs 0.93, doğrulama 1.05 vs 0.56) → karma yapı korunur.
v1.3, adayları literatürden değil **sistemin kendi kazanan/kaybeden işlemlerinden** türetir.

**Aşama 1 — Anatomi (yalnızca eğitim dönemi, BASE'in gerçek işlemleri):**
- Her işlem, karar günü (giriş öncesi kapanış) bilinen 17 özellikle eşleştirilir: 11 hisse düzeyi (momentum skoru, 52 hafta
  zirvesine yakınlık, oynaklık, oynaklık geçişi, gap riski, ciro katılımı, ciro sıçraması, likidite, son gün şoku, 5g ve 21g getiri)
  ve 6 piyasa düzeyi (genişlik, 21g piyasa getirisi, 126g piyasa oynaklığı, 200g trend, 24 ay ayı piyasası, kur 21g — yalnız BIST).
- Her özellik için 5'li dilim tablosu (kazanma oranı, işlem başı net) ve **her eğitim yılı için ayrı** sıra korelasyonu (özellik ↔ getiri).
- **Seçim ölçütü (önceden sabit):** yıllık korelasyonların yıllar arası t ≥ 2.0 VE yılların ≥ %75'inde aynı yön. Tek güçlü yıldan
  gelen etki bu ölçütü geçemez. En güçlü en fazla 3 (özellik, piyasa) çifti aday olur.

**Aşama 2 — Adaya çevirme (mekanik, ayar yok):**
- Hisse düzeyi → en kötü dilime (eğitim işlemlerinin %20/%80 sınırı) düşen aday elenir, sıradakiyle doldurulur.
- Piyasa düzeyi → piyasa en kötü dilimdeyken o kolun yeni dilim bütçesi ×0.5.
- Aday yalnızca bulunduğu piyasaya uygulanır.

**Aşama 3 — Doğrulama ve kilitli test:** v1.1 ile aynı (Holm, White RC, plasebo %90, seçim ölçütleri, PASS kapısı).
Portföy ölçümleri USD bazındadır. Kilitli pencere yalnızca doğrulamayı geçen tek aday için açılır.

**Ek çıktılar:** `anatomy_summary.csv` (özellik başına yıllık IC, t, tutarlılık, en iyi/en kötü dilim), `anatomy_quintiles.csv`,
`anatomy_yearly_ic.csv`.

**Not (duman testinden):** Tamamen rastgele sentetik veride bile 3 özellik eğitim ölçütünü geçti, biri doğrulamayı da geçti ve
kilitli testte elendi. Anatomi tek başına kanıt değildir; kanıt doğrulama + plasebo + kilitli testtir.

---

## §H. v1.4 — SEC temel veri sinyalleri (`--mode fundamentals`, karma portföy, USD bazlı)

**Gerekçe:** v1.1–v1.3'te fiyat/hacimden türetilen 28 adayın hiçbiri kanıt eşiğini geçemedi. v1.4, GMQ'nun hiç kullanmadığı
bir bilgi türünü ekler: ABD şirketlerinin SEC'e dosyaladığı finansal tablolar (data.sec.gov companyfacts API, ücretsiz).

**Zaman doğruluğu (kalite kapısının parçası):**
- Her değer SEC'e **ilk dosyalandığı** tarihten sonraki günden itibaren kullanılır; aynı dönem için sonradan dosyalanan
  düzeltmeler yok sayılır.
- Q4 kazancı yalnızca yıllık raporda varsa Q4 = yıllık − 3 çeyrek; bilinme tarihi = yıllık raporun dosyalanması.
- SUE yalnızca dosyalamadan sonraki 91 gün, GP/A 15 ay geçerlidir. Eksik veri nötr kabul edilir (sıfırla doldurulmaz).
- Eğitim döneminde ABD evren-günlerinin en az %30'unda veri yoksa ilgili aday NOT TESTABLE olur.

**Adaylar (literatürden önceden sabit; eğitim tanısı seçimi etkilemez):**
| Kod | Tanım | Kaynak |
|---|---|---|
| F1 | Son bilançoda kazanç sürprizi (SUE) evrenin en kötü %20'sindeki aday elenir | Bernard & Thomas (1989), bilanço sonrası kayma |
| F2 | Brüt kârlılığı (GP/A) evrenin en kötü %20'sindeki aday elenir | Novy-Marx (2013), kalite |
| F3 | Sıralama = 0.75 × üretim skoru + 0.125 × SUE + 0.125 × GP/A yüzdeliği | birleşik |

Doğrulama, plasebo (yalnız ABD kolunda), Holm, White RC ve kilitli test kuralları v1.1 ile aynıdır.
**Ek çıktılar:** `fundamentals_train_diagnostic.csv` (eğitim döneminde her sinyalin yıllık IC'si), `fundamentals_coverage.csv`.
