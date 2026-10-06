# GMQ Early-Adverse Predictor — Hardened v2

Bu paket, önceki run'da görülen:

`❌ HATA: False`

durumunu teşhis edilebilir hale getirir ve bağımlılık sürüm kaymasını sınırlar.

## Neden?

Repo'nun `requirements.txt` dosyası `yfinance>=0.2.54` dediği için GitHub Actions bugün daha yeni 1.x sürümlerini kurabilir. PyPI'da yfinance 1.7.0, 26 Ağustos 2026'da yayınlandı; 0.2.66 ise 17 Eylül 2025 sürümüdür. Bu araştırmanın tekrarlanabilirliğini bozabilecek bir major-version drift oluşturur.

Bu workflow araştırma ortamında `yfinance==0.2.66` ve `scikit-learn==1.7.2` sabitler.

Not: Yahoo tarafındaki `possibly delisted; no timezone found` uyarıları halen bazı geçersiz/erişilemeyen ticker'larda görülebilir. Mevcut data.py bunları atlayarak devam edecek şekilde tasarlanmıştır.

## Kritik

Bu paket `gmq_early_adverse_predictor.py` matematiğini değiştirmez.

Eklenen:

- hardened runner
- bağımlılık pinleme
- `DOWNLOAD_CHUNK=30` araştırma-only override
- tam exception type/repr/traceback
- faulthandler
- `ubuntu-24.04`
- sadece manuel workflow

## Kurulum

Repo kökünde:

1. `gmq_early_adverse_predictor_hardened_runner.py` dosyasını ekle.
2. `.github/workflows/gmq_early_adverse_predictor.yml` dosyasını paketteki sürümle değiştir.

Eski `gmq_early_adverse_predictor.py` dosyasını değiştirme.

Ardından:

Actions → Global Momentum Quant - Early-Adverse Predictor (Hardened v2) → Run workflow

Değerler:

- years: `14`
- cost: `0.35`

## Başarı ölçütü

Job artık başarısız olursa yalnızca `False` yazmayacak. Şu blok kesin olarak görünecek:

TYPE
STR
REPR
TRACEBACK

Böylece hatanın tam kaynağı tek run'da ayrıştırılabilir.

Gerçek araştırma başarılı olursa önceki 6 artifact dosyası üretilecektir.
