# Global Momentum Quant

İki piyasayı (BIST ve ABD S&P 500) tek bir TL portföyü olarak yöneten, kendini denetleyen momentum sistemi.

## Strateji (v1.4 — BIST + ABD karma)

| Katman | Kural |
|---|---|
| BIST kolu | 12 ay momentum + 52 hafta zirvesine yakınlık + düşük aşırı sıçrama → en iyi 10 hisse |
| ABD kolu | S&P 500 kalıntı (piyasadan arındırılmış) momentum → en iyi 10 hisse |
| Kademeli giriş | Her kol 4 dilim. Her hafta bir dilim yenilenir, her işlem **en fazla 21 işlem günü** tutulur |
| Felaket stopu | Kapanış girişin %25 altına inerse ertesi açılışta satılır |
| Kollar arası ağırlık | Risk paritesi (63 günlük oynaklığın tersi, ayda bir; BIST %30–90 arası) |
| Portföy sigortası | Kapalı |
| %100 ABD büyükler | `config.py` içinde `FIXED_WEIGHTS = {"bist": 0.0, "us": 1.0}` ve ABD `liq_min_pct = 0.75` yapılırsa açılır |

## 13 yıllık test (canlı motorla, geçmiş S&P 500 üyelikleriyle, 2014-01 → 2026-09)

| | 1 TL ne oldu | Yıllık | En büyük düşüş | 12 aylık dönemlerde SPY'ı geçme |
|---|---|---|---|---|
| **Sistem: BIST+ABD karma (TL)** | **≈153 TL** | **%48,4** | **−%27,0** | %53 |
| Sistem, dolar bazında | | %16,1 | −%37,2 | |
| Sistem, reel (TÜFE üstü) | | %18,1 | −%28,8 | |
| %100 ABD büyükler (TL) | ≈175 TL | %50,6 | −%42,2 | %70 |
| SPY (TL) | ≈95 TL | %43,0 | −%37,0 | – |

- **İşlem düzeyi:** BIST'te kârla kapanma %55,7, işlem başı net +%3,4, kâr faktörü 1,90. ABD'de kârla kapanma %55,9, işlem başı net +%1,5, kâr faktörü 1,58.
- **Kalan iyimserlik:** Endeksten çıkmış şirketlerin fiyat verisi yok; küçük bir iyimserlik payı kaldı.
- **SPY karnesi:** Her ABD çalışmasında sistem SPY ile karşılaştırılır. Son 24 ayda SPY'ın %10'dan fazla gerisinde kalırsa uyarı gelir.
- **Denenip kullanılmayanlar:** Sektör sınırı, momentum çöküş freni ve portföy sigortası test edildi; kapalı tutuluyor.

## Bot için emir dosyası

Her çalışmada `data/orders_bist.json` ve `data/orders_us.json` dosyaları yazılır. Bunlar her zaman o pazarın **en son** emir listesini içerir; o gün emir yoksa liste boştur. Tüm emirlerin geçmişi `data/orders_history.jsonl` dosyasında satır satır tutulur.

Her emirde şu alanlar bulunur:
- `id`: benzersiz kimlik; aynı emri iki kez işlememek için kullanılır.
- `action`: `BUY` / `SELL` / `REDUCE` / `INCREASE` / `HOLD`.
- `symbol` ve `yahoo_symbol`.
- `quantity`: BIST'te tam sayı, ABD'de kesirli.
- `sell_all` ve `amount`.
- `stop_price`.
- `order_type = MARKET_ON_OPEN` ve `execute_date`.
- `exit_date_est` ve `reason`.

Dosyada ayrıca `transfer` (hesaplar arası aktarım), `portfolio` ve `positions` (açık pozisyonlar, stop fiyatları, kalan gün) bölümleri var.

Telegram'a da her emir mesajının ardından sabit biçimli ikinci bir mesaj gelir. Örnek:
```
BUY THYAO.IS QTY=12 AMT=3500.0 STOP=215.81 MARKET_ON_OPEN 2026-10-05
SELL AAPL QTY=ALL MARKET_ON_OPEN 2026-10-05
TRANSFER BIST->US TRY=5000.0 USD=102.04
```

## Kurulum (yalnızca telefon + GitHub web)

1. **Repo oluştur:** GitHub'da `global_momentum_quant` adında yeni bir repo aç.
2. **Dosyaları yükle:** "Add file → Upload files" ile zip'teki dosyaları yükle (`data/` ve `backtest_ref/` klasörleri dahil).
3. **İş akışı dosyası:** Telefondan gizli klasör yüklenemeyebilir. Bu durumda "Add file → Create new file" ile dosya adına tam olarak `.github/workflows/run.yml` yaz ve zip'teki `kurulum/run.yml` içeriğini yapıştır. Aynı şekilde `.gitignore` adlı dosyayı `kurulum/gitignore.txt` içeriğiyle oluştur.
4. **Gizli bilgiler:** Settings → Secrets and variables → Actions → New repository secret:
   - `TELEGRAM_BOT_TOKEN` = bot token
   - `CHAT_ID` = sohbet kimliği
   - GitHub isimleri otomatik büyük harfe çevirir; `telegram_bot_token` yazman da olur.
5. **Yazma izni:** Settings → Actions → General → Workflow permissions → **Read and write permissions** → Save.
6. **Sermaye:** `config.py` içindeki `CAPITAL_TL = 100_000` değerini kendi sermayenle değiştir.
7. **İlk çalıştırma:** Actions → Global Momentum Quant → Run workflow → `bist` → çalıştır. Sonra aynısını `us` ile yap. Telegram'a hoş geldin mesajı ve hesaplara ne kadar para koyacağın gelir.
8. **Panel:** share.streamlit.io → New app → bu repo → `app.py`.

## Günlük kullanım

- **Ne zaman mesaj gelir:**
  - BIST mesajları ≈18:45'te gelir.
  - ABD mesajları ≈00:40'ta gelir.
  - Mesaj yalnızca yapılacak iş, uyarı ya da Cuma özeti varsa gelir.
- **Mesajda ne var:**
  - 🔴 SAT, 🟢 AL, ⚪ TUT, 🟠 AZALT ve ⛔ FELAKET STOPU satırlarını **ertesi günün açılışında** uygula.
  - Hisse adedi ve felaket stopu fiyatı mesajda yazar.
- **Kesirli hisse:** ABD tarafında pozisyonlar küçük olabilir. Kesirli hisse alabilen bir aracı kurum gerekir; ya da sermayeyi büyüt.
- **🔁 AKTARIM:** Söylenen tutarı iki hesap arasında aktar.
- **Sistemle eşitleme:** Bir emri uygulayamazsan sorun değil, ama gerçek hesabın sistemden sapar. Panelden açık pozisyonları kontrol ederek eşitle.

## Öz denetim (yavaş ve kanıta dayalı)

- **Canlı denetim:**
  - Her kapanan dilim, aynı günlerde evrenin ortalamasıyla (rastgele seçim beklentisi) karşılaştırılır.
  - Son 24 dilimde sistem rastgeleden anlamlı derecede kötüyse (t < −2), o kolda yeni alımlar durur ve haber verilir.
  - Toparlanınca alımlar yeniden başlar.
- **Aylık derin denetim** (her ayın 2'si):
  - Son 8 yıllık veriyle önceden kayıtlı rakip stratejiler test edilir.
  - Bir rakip, aktif stratejiyi son 36 ayda t>2 farkla **3 ay üst üste** geçerse strateji değişir.
  - Değişiklik yılda en fazla 1 kez olur.
  - Tek bir iyi ay hiçbir şeyi değiştirmez.

## Dosyalar

| Dosya | Görev |
|---|---|
| `config.py` | Tüm ayarlar |
| `signals.py` | Sinyaller (araştırmadakiyle birebir) |
| `engine.py` | Dilim motoru, felaket stopu, bölünme/bedelsiz düzeltmesi |
| `portfolio.py` | Risk paritesi, sigorta, aktarım, geçmiş test |
| `data.py` | Ücretsiz veri (yfinance, Wikipedia, TradingView) |
| `auditor.py` | Öz denetim |
| `telegram_bot.py` | Mesajlar |
| `orders.py` | Bot için emir dosyası |
| `main.py` | Giriş noktası |
| `app.py` | Streamlit paneli |
| `selftest.py` | Sentetik veriyle doğruluk testleri; `./data`'ya yazmaz |

`data/` klasöründe sistemin durumu (`state.json`), işlemler, değer geçmişi ve mesaj günlüğü tutulur. Elle değiştirme.
