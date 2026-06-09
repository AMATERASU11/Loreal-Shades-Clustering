"""
Configuration — Pipeline ShadeNail (Vernis à Ongles)
═══════════════════════════════════════════════════════════════════
Architecture modulaire en tiers :
  Tier 0 : Gatekeeper   (kit / incolore / nail_accessory / volume_only)
  Tier 1 : Texture      (glitter / holo / ombre / gradient / cat_eye)
  Tier 2 : NLP Prior    (COLOR_WORDS → Lab centroid → 6 features NLP)
  Tier 3 : Vision Core  (cache NPZ → KMeans K=5 → 139 features)
  Tier 4 : LightGBM     (139 features → classe luminosité 0-4 → couleur)
"""
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
DATA_DIR      = PROJECT_ROOT / "data"
IMAGE_DIR     = DATA_DIR / "raw" / "images"
RECOVERED_DIR = DATA_DIR / "raw" / "recovered_images_v6"

LABELED_PARQUET = DATA_DIR / "labeled" / "nail_all_labeled_1261.parquet"

CACHE_DIR  = PROJECT_ROOT / "outputs" / "cache"
MODEL_DIR  = PROJECT_ROOT / "outputs" / "models"
RESULT_DIR = PROJECT_ROOT / "outputs" / "reports"

for _d in [CACHE_DIR, MODEL_DIR, RESULT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Preprocessing ─────────────────────────────────────────────────
ALPHA_THRESHOLD = 10

CENTER_CROP_TOP    = 0.30
CENTER_CROP_BOTTOM = 0.05

# ── KMeans ────────────────────────────────────────────────────────
N_CLUSTERS          = 5
KMEANS_N_INIT       = 15
KMEANS_RANDOM_STATE = 42
MIN_PIXELS          = N_CLUSTERS * 20

# ── Features (identique methode_5) ───────────────────────────────
# Vision blocs A-P : 105
# HSV              :  19
# NLP              :   6
# Vernis score     :   9
N_VISION_FEATURES = 105
N_HSV_FEATURES    = 19
N_NLP_FEATURES    = 6
N_VERNIS_FEATURES = 9
N_TOTAL_FEATURES  = N_VISION_FEATURES + N_HSV_FEATURES + N_NLP_FEATURES + N_VERNIS_FEATURES  # 139

# ── Confiance & Tags ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60

TAG_OK       = "OK"
TAG_KIT      = "Kit - Exclu"
TAG_INCOLORE = "Incolore - Exclu"
TAG_TEXTURE  = "Texture Complexe - Non traitable"
TAG_CAT_EYE  = "Cat Eye - Couleur Base"
TAG_ANOMALIE = "Anomalie - Extraction Impossible"

# ── Détection kit hybride ─────────────────────────────────────────
KIT_DE_THRESHOLD = 20

# ── Entraînement LightGBM (identique methode_5) ───────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 10

LGBM_PARAMS = dict(
    n_estimators=1200, max_depth=4, learning_rate=0.02,
    subsample=0.70, colsample_bytree=0.75,
    min_child_samples=8, reg_alpha=0.1, reg_lambda=2.0,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, num_leaves=35,
    class_weight="balanced",
)

# ── Évaluation ────────────────────────────────────────────────────
DELTA_E_THRESHOLDS = [5, 10, 15, 20, 25]
