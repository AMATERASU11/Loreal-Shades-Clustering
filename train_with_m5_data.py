"""
Entraînement LightGBM (139 features) depuis le cache methode_5.

Utilise features_cache.pkl de methode_5 (139 features déjà calculées)
+ les corrections de labels (label_class_fixes.json)
+ les quarantines (dataset_decisions.json)

Usage :
    python train_with_m5_data.py
"""
import sys, json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

EQUIPE_ROOT  = Path(__file__).parent
PROJECT_ROOT = EQUIPE_ROOT.parent
M5_DIR       = PROJECT_ROOT / "methode_5"
M5_RESULT_DIR = M5_DIR / "results"
M5_MODEL_DIR  = M5_DIR / "models"

sys.path.insert(0, str(EQUIPE_ROOT))
from src.utils.config import MODEL_DIR, RESULT_DIR, RANDOM_STATE, CV_FOLDS, LGBM_PARAMS
import src.feature_engineering.tier0_gatekeeper as tier0
import src.feature_engineering.tier1_texture    as tier1

# ── Chargement features_cache.pkl (methode_5) ─────────────────────
print("Chargement features_cache.pkl de methode_5...")
with open(M5_MODEL_DIR / "features_cache.pkl", "rb") as f:
    cache = pickle.load(f)
print(f"  {len(cache)} produits, {len(cache[0]['X'])} features")

# ── Appliquer quarantines ─────────────────────────────────────────
dec_path = M5_RESULT_DIR / "dataset_decisions.json"
decisions = json.load(open(dec_path, encoding="utf-8"))
cache = [r for r in cache if decisions.get(r.get("image", ""), "train") == "train"]
print(f"  Apres quarantines : {len(cache)} produits")

# ── Appliquer filtres tier0/tier1 (meme logique que run_cv_full.py) ──
df_labeled = pd.read_parquet(EQUIPE_ROOT / "data" / "labeled" / "nail_all_labeled_1261.parquet")
df_map = df_labeled.set_index("image_filename")

valid_imgs = set()
for r in cache:
    img = r.get("image", "")
    if img not in df_map.index:
        continue
    row = df_map.loc[img]
    t0 = tier0.classify(row)
    if t0 in ("kit_multicolor", "incolore", "nail_accessory", "volume_only"):
        continue
    if t0 in ("normal", "kit_monocolor"):
        t1 = tier1.classify(row)
        if t1 in ("texture_complexe", "cat_eye"):
            continue
    valid_imgs.add(img)

cache = [r for r in cache if r.get("image", "") in valid_imgs]
print(f"  Apres filtres tier0/tier1 : {len(cache)} produits")

# ── Appliquer corrections de labels ──────────────────────────────
fix_path = M5_RESULT_DIR / "label_class_fixes.json"
label_fixes = json.load(open(fix_path, encoding="utf-8"))
n_patched = 0
for r in cache:
    img = r.get("image", "")
    if img in label_fixes:
        r["y_true"] = int(label_fixes[img])
        n_patched += 1
print(f"  Labels corriges : {n_patched}")

# ── Construction X, y ─────────────────────────────────────────────
X = np.array([r["X"] for r in cache], dtype=np.float32)
y = np.array([r["y_true"] for r in cache], dtype=np.int32)

unique, counts = np.unique(y, return_counts=True)
print(f"  Distribution classes : {dict(zip(unique.tolist(), counts.tolist()))}")
print(f"  Dataset : {len(X)} produits, {X.shape[1]} features")

# ── Cross-validation 10-fold ──────────────────────────────────────
print(f"\nCross-validation {CV_FOLDS}-fold LightGBM...")
skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
model_cv = LGBMClassifier(**LGBM_PARAMS)
cv_scores = cross_val_score(model_cv, X, y, cv=skf, scoring="accuracy")

print(f"\nCV {CV_FOLDS}-fold : {cv_scores.mean()*100:.1f}% +/- {cv_scores.std()*100:.1f}%")
for i, s in enumerate(cv_scores):
    n_err = int(round((1 - s) * (len(X) // CV_FOLDS)))
    print(f"  Fold {i+1:2d}/{CV_FOLDS} : {s*100:.1f}%  (~{n_err} erreurs)")

# ── Entraînement final sur tout le dataset ────────────────────────
print("\nEntrainement final sur tout le dataset...")
model_final = LGBMClassifier(**LGBM_PARAMS)
model_final.fit(X, y)

# ── Sauvegarde ────────────────────────────────────────────────────
model_path = MODEL_DIR / "best_model.pkl"
meta_path  = MODEL_DIR / "train_meta.json"

with open(model_path, "wb") as f:
    pickle.dump(model_final, f)

train_meta = {
    "model":        "LightGBM",
    "n_features":   int(X.shape[1]),
    "n_train":      int(len(X)),
    "cv_folds":     CV_FOLDS,
    "cv_mean":      round(float(cv_scores.mean()), 4),
    "cv_std":       round(float(cv_scores.std()), 4),
    "cv_per_fold":  [round(float(s), 4) for s in cv_scores],
    "lgbm_params":  {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"},
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(train_meta, f, indent=2)

print(f"\n{'='*55}")
print(f"CV {CV_FOLDS}-fold : {cv_scores.mean()*100:.1f}% +/- {cv_scores.std()*100:.1f}%")
print(f"Modele sauvegarde : {model_path}")
print(f"Metadata          : {meta_path}")
