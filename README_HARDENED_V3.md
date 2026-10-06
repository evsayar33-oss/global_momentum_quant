# GMQ Early-Adverse Predictor — Hardened v3

Bu sürüm önceki run'daki somut yeni bulguya göre hazırlandı.

## Neden v3?

Son GitHub logunda araştırma ortamı şu şekildeydi:

- Python 3.11.16
- pandas 3.0.6
- numpy 2.4.6
- yfinance 0.2.66
- scikit-learn 1.7.2

Buradaki problem yfinance 0.2.66'ın 2025 sürümü olması, buna karşılık pandas 3.x'in 2026 ortamında kurulmuş olmasıdır. yfinance upstream changelog'unda pandas 3 desteğinin daha sonraki 1.x geliştirme hattında geldiği görülüyor. Bu nedenle araştırmada yfinance 0.2.66 kullanırken pandas 2.3.3'e dönmek daha güvenli ve tekrarlanabilir bir kombinasyondur.

Ayrıca bu sürüm gerçek araştırmadan önce dört küçük Yahoo veri bağlantısını test eder:

- BIST
- US
- USD/TRY
- ^IRX

Bunlardan biri boşsa 14 yıllık indirme başlamadan anlaşılır.

## Kurulum

Repo'da mevcut:

`.github/workflows/gmq_early_adverse_predictor.yml`

dosyasını ZIP'teki dosyayla değiştir.

`gmq_early_adverse_predictor.py` dosyasına dokunma.

`gmq_early_adverse_predictor_hardened_runner.py` dosyasını repo köküne ekle.

Sonra:

Actions → Global Momentum Quant - Early-Adverse Predictor (Hardened v3) → Run workflow

Parametre:

- years = 14
- cost = 0.35

## Beklenen sıra

Repo
→ Python
→ Uyumlu araştırma ortamı
→ Sürüm/sözdizimi
→ Yahoo preflight
→ Synthetic smoke
→ Gerçek araştırma
→ Artifact

Eğer gerçek araştırma tekrar hata verirse workflow ek debug özeti de yazacaktır.

Canlı `config.py`, state, order, NAV veya trade dosyaları değiştirilmez.
