# GMQ Momentum Entry Timing / Chase Avoidance V4

## Amaç
V3'te görülen execution-mapping problemini research boundary'de kapatır.
V4, sınıflandırılmış bir chase olayını yalnızca gerçek production engine BUY order -> BUY fill -> closed cycle zinciriyle eşleştirdikten sonra sizing lookup'a alır.

## Dosyalar
- `gmq_momentum_entry_timing_chase_avoidance_v4.py`
- `gmq_momentum_entry_timing_chase_avoidance_v4.yml`

## Kurulum
1. Python dosyasını repository root'a yükleyin.
2. YAML dosyasını `.github/workflows/` içine yükleyin.
3. Production `engine.py`, `portfolio.py`, `config.py`, `signals.py`, state/order/NAV dosyalarını değiştirmeyin.

## V4'ün önemli farkı
Sizing identity artık:

`market + ticker + signal_date + tranche`

Ayrıca workflow şu koşullarda araştırmayı geçersiz sayar:
- execution-verified dataset yoksa,
- duplicate BUY-cycle identity oluşursa,
- classified dangerous event sayısı ile gerçekten ölçeklenen BUY event sayısı eşleşmezse.

## Smoke test
`py_compile` ve synthetic smoke testi başarıyla geçmiştir.

## Production
Bu paket production entegrasyonu yapmaz; yalnızca research/backtest çalıştırır.
