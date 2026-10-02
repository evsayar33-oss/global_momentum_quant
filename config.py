"""Global Momentum Quant — tüm ayarlar tek yerde.

Strateji (13 yıllık veride, rastgele seçim kıyaslı, eğitim 2014-19 / test 2020-26 ile seçildi):
  * BIST kolu : 12 ay momentum + 52 hafta zirvesine yakınlık + düşük aşırı-sıçrama, 10 hisse, 21 işlem günü
  * ABD kolu  : S&P 500 kalıntı (piyasadan arındırılmış) momentum, 10 hisse, 21 işlem günü
  * Sermaye   : BIST + ABD, risk paritesiyle (FIXED_WEIGHTS = None)
  * Her kol 4 dilim: her dilim 21 günde bir yenilenir, dilimler 5'er gün kaydırılır (kademeli giriş)
  * Felaket stopu: kapanış girişin %25 altına inerse ertesi açılışta sat
  * Kollar arası ağırlık: risk paritesi (63 günlük oynaklığın tersi, ayda bir)
  * Portföy sigortası: KAPALI (13 yıllık testte reel ve dolar bazında sigortasız sürüm daha iyi çıktı);
    INS_ENABLED = True yapılırsa: zirveden %8 düşüşte pozisyonlar %25'e iner
Kullanıcının değiştirebileceği değerler en üstte.
"""
import os

# ---------------------------------------------------------------- kullanıcı ayarları
CAPITAL_TL = 100_000          # başlangıç sermayesi (TL). Mesajlardaki tutarlar buna göre ölçeklenir.
START_DATE = None             # None: ilk çalıştırma günü başlar. "2026-10-05" gibi bir tarih de verilebilir.

# ---------------------------------------------------------------- strateji (araştırmayla birebir)
ENGINE_VERSION = "1.4.0"
HOLD_DAYS = 21                # dilim tutma süresi (işlem günü)
N_TRANCHES = 4                # kademeli giriş dilim sayısı
TRANCHE_STEP = 5              # dilimler arası gün kaydırması
N_PICKS = 10                  # dilim başına hisse
CAT_STOP = 0.25               # felaket stopu: girişten %25 düşüş
MIN_TRADE = {"bist": 250.0, "us": 10.0}   # bundan küçük yeniden boyutlandırma emri üretilmez
SECTOR_CAP = None             # sektör başına en fazla hisse (None = sınırsız)
LIQ_MIN_PCT = 0.40            # likidite: 20 gün medyan işlem hacmine göre alttaki %40 dışarıda

MARKETS = {
    "bist": {
        "name": "BIST", "ccy": "TL", "suffix": ".IS", "max_move": 0.105,
        "cost_rt_pct": 0.35,                       # gidiş-dönüş maliyet (komisyon + kayma)
        "spec": [["mom_12_1", 1], ["hi52", 1], ["low_max", 1]],
        "universe_file": "data/universe_bist.json",
    },
    "us": {
        "name": "ABD (S&P 500)", "ccy": "$", "suffix": "", "max_move": 0.40,
        "cost_rt_pct": 0.10,
        "liq_min_pct": 0.40,                       # %100 ABD büyükler sürümü için 0.75
        "spec": [["resid_mom", 1]],
        "universe_file": "data/universe_us.json",
    },
}

# ---------------------------------------------------------------- portföy düzeyi
RP_WINDOW = 63                # risk paritesi oynaklık penceresi
FIXED_WEIGHTS = None          # None: BIST+ABD karma (risk paritesi). {"bist": 0.0, "us": 1.0}: %100 ABD
RP_DEFAULT = {"bist": 0.64, "us": 0.36}   # yeterli geçmiş yokken (13 yıllık ortalama)
RP_BOUNDS = (0.30, 0.90)      # BIST ağırlığı sınırları (aşırı uçlara karşı)
REBAL_TOL = 0.05              # hedeften 5 puan sapınca aktarım önerisi
INS_ENABLED = False           # portföy sigortası (TL nakde çekilir; reel bazda zarar verdiği için kapalı)
INS_TRIGGER = 0.08            # sigorta: zirveden düşüş
INS_RELEASE = 0.04            # sigorta kapanış eşiği
INS_EXPOSURE = 0.25           # sigorta devredeyken pozisyon boyutu (%25)
INS_RESTORE_NOW = True        # sigorta kapanınca pozisyonları hemen tam boyuta tamamla
VOL_TARGET = False            # momentum çöküş freni (oynaklık ölçekleme)
VOL_FLOOR = 0.25
VOL_TARGET_FIXED = None       # None: stratejinin kendi geçmiş medyan oynaklığı
CASH_RATE = {"bist": 0.30, "us": 0.035}   # nakitte bekleyen paranın yıllık net getirisi (para piyasası fonu / T-bill)

# ---------------------------------------------------------------- öz denetim (yavaş, kanıta dayalı)
AUDIT_MIN_TRANCHES = 12       # canlı karar için en az kapanmış dilim
AUDIT_WARN_T = -1.0           # rastgeleye göre fazla getiri t-değeri bunun altındaysa uyarı
AUDIT_PAUSE_T = -2.0          # bunun altındaysa o kolda yeni alımlar durur
AUDIT_PAUSE_DD = 0.30         # kol kendi zirvesinden %30 düşerse (testte en kötü ~%28) uyarı + inceleme
SWITCH_T_MARGIN = 2.0         # rakip strateji, son 36 ayda mevcut olanı t>2 farkla geçmeli
SWITCH_CONFIRM_MONTHS = 3     # üst üste kaç aylık denetimde
SWITCH_COOLDOWN_DAYS = 365    # iki değişiklik arası en az süre
CANDIDATES = {                # önceden kayıtlı rakipler (araştırmada sağlam çıkanlar)
    "bist": {
        "A_mom12_hi52_lowmax": [["mom_12_1", 1], ["hi52", 1], ["low_max", 1]],
        "B_mom12_hi52": [["mom_12_1", 1], ["hi52", 1]],
        "C_resid_hi52_clv": [["resid_mom", 1], ["hi52", 1], ["clv", 1]],
        "D_upvol_posdays": [["upvol_ratio", 1], ["pos_days", 1]],
    },
    "us": {
        "A_resid_mom": [["resid_mom", 1]],
        "E_resid_mom12": [["resid_mom", 1], ["mom_12_1", 1]],
        "B_overnight_mom": [["overnight_mom", 1]],
        "C_mom_12_1": [["mom_12_1", 1]],
        "D_resid_overnight": [["resid_mom", 1], ["overnight_mom", 1]],
    },
}

# ---------------------------------------------------------------- veri
HISTORY_YEARS = 3             # günlük çalışmada indirilen geçmiş (sinyaller ~2 yıl ister)
AUDIT_YEARS = 8               # aylık denetimde indirilen geçmiş
DOWNLOAD_CHUNK = 60
FX_TICKER = "TRY=X"           # USD/TRY
INDEX_TICKERS = {"bist": "XU100.IS", "us": "SPY"}

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRADES_FILE = os.path.join(DATA_DIR, "trades.csv")
NAV_FILE = os.path.join(DATA_DIR, "nav.csv")
AUDIT_FILE = os.path.join(DATA_DIR, "audit.json")
LOG_FILE = os.path.join(DATA_DIR, "messages.log")
REF_DIR = os.path.join(BASE, "backtest_ref")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.environ.get("CHAT_ID", "")
