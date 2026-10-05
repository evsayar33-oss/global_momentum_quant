# GMQ — Early-Adverse Predictor / Entry Risk Quality

## Manuel çalışma sürümü

Bu workflow **yalnızca `workflow_dispatch` ile manuel başlatılır**.

Otomatik `schedule` kaldırılmıştır. Böylece haftalık zamanlanmış çalışma başlamaz.

### Çalıştırma

GitHub → **Actions** → **Global Momentum Quant - Early-Adverse Predictor** → **Run workflow**

Değerleri:

- `years = 14`
- `cost = 0.35`

olarak bırak.

### Güvenlik / teşhis

- Timeout: 360 dakika.
- `PYTHONUNBUFFERED=1`: loglar tamponlanmadan GitHub Actions'a akar.
- Önce `py_compile` çalışır.
- Sonra synthetic smoke test çalışır.
- Smoke test geçmeden gerçek araştırma çalışmaz.
- Araştırma çıktıları yalnızca başarılı gerçek koşudan sonra artifact olarak yüklenir.
- Canlı `config.py`, `state`, `orders`, `NAV`, `trades` dosyaları değiştirilmez.
