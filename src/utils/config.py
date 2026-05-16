"""
Configuration — Pipeline ShadeNail (Vernis à Ongles)
═══════════════════════════════════════════════════════════════════
Architecture modulaire en tiers :
  Tier 0 : Gatekeeper   (kit / incolore)
  Tier 1 : Texture      (glitter / shimmer / holographique)
  Tier 2 : NLP Prior    (COLOR_WORDS → Lab centroid → ΔE features)
  Tier 3 : Vision Core  (cache rembg → KMeans K=5 → 35 features)
  Tier 4 : ShadeNail Ranking (34 features × 5 clusters → ΔE regression → argmin)
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

# Cache M4 — fallback pour run_full_extraction sur les ~8000 produits
# sans cache complet (lab seulement, vpos neutre)
M4_CACHE_DIR = DATA_DIR / "raw" / "rembg_cache_m4"

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

# ── Features ──────────────────────────────────────────────────────
# BLOC A : L, a, b, weight × K  = 20
# BLOC B : chroma C* × K        =  5
# BLOC C : dispersion std ΔE × K =  5
# BLOC D : position verticale × K =  5
# NLP    : has_prior + ΔE×K     =  6
N_VISION_FEATURES = 35
N_NLP_FEATURES    = 6
N_TOTAL_FEATURES  = N_VISION_FEATURES + N_NLP_FEATURES  # 41

# ── Confiance & Tags ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60

TAG_OK       = "OK"
TAG_KIT      = "Kit - Exclu"
TAG_INCOLORE = "Incolore - Exclu"
TAG_TEXTURE  = "Texture Complexe - Non traitable"
TAG_ANOMALIE = "Anomalie - Extraction Impossible"

# ── Détection kit hybride ─────────────────────────────────────────
KIT_DE_THRESHOLD = 20

# ── Entraînement ──────────────────────────────────────────────────
TEST_SIZE    = 0.20
CV_FOLDS     = 5
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators":      400,
    "max_depth":         4,
    "learning_rate":     0.05,
    "subsample":         0.80,
    "colsample_bytree":  0.80,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "use_label_encoder": False,
    "eval_metric":       "mlogloss",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
}

# ── Évaluation ────────────────────────────────────────────────────
DELTA_E_THRESHOLDS = [5, 10, 15, 20, 25]

# ── ShadeNail Ranking ─────────────────────────────────────────────
# Modèle : XGBoost régresseur, régression ΔE
N_RANKING_FEATURES = 34
SHADENAIL_MODEL_PATH = MODEL_DIR / "shadenail_xgb.pkl"
SHADENAIL_META_PATH = MODEL_DIR / "shadenail_meta.json"
