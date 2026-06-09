"""
Tier 4 — XGBoost Arbitrateur
═══════════════════════════════════════════════════════════════════
Entraîne et applique le sélecteur de cluster XGBoost.

Input  : 41 features (35 vision + 6 NLP)
Target : index du cluster (0..4) le plus proche du manual_label
Output : couleur Lab/RGB + confidence + tag

Règle de confiance :
  max(predict_proba) >= 0.60 → couleur prédite
  max(predict_proba) <  0.60 → [Anomalie - Extraction Impossible]

Usage :
  python -m src.clustering.tier4_xgboost --train     # entraîner + sauvegarder
  python -m src.clustering.tier4_xgboost --eval       # évaluer sur test set
"""
import argparse
import ast
import json
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from skimage import color as skcolor
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError:
    print("ERREUR : xgboost non installé. Lancer : pip install xgboost")
    sys.exit(1)

from src.utils.config import (
    LABELED_PARQUET, MODEL_DIR, RESULT_DIR,
    N_CLUSTERS, CONFIDENCE_THRESHOLD,
    TEST_SIZE, CV_FOLDS, RANDOM_STATE,
    XGB_PARAMS, DELTA_E_THRESHOLDS,
    TAG_OK, TAG_KIT, TAG_INCOLORE, TAG_TEXTURE, TAG_ANOMALIE,
    KIT_DE_THRESHOLD,
)
import src.feature_engineering.tier0_gatekeeper as tier0
import src.feature_engineering.tier1_texture    as tier1
import src.feature_engineering.tier2_nlp_prior  as tier2
import src.feature_engineering.tier3_vision     as tier3


# ═══════════════════════════════════════════════════════════════
#  UTILITAIRES COULEUR
# ═══════════════════════════════════════════════════════════════

def rgb_to_lab(rgb) -> np.ndarray:
    arr = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
    return skcolor.rgb2lab(arr / 255.0)[0, 0]


def delta_e(lab1, lab2) -> float:
    return float(np.sqrt(np.sum((np.array(lab1) - np.array(lab2)) ** 2)))


def parse_label(label) -> Optional[list]:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    try:
        parsed = ast.literal_eval(label) if isinstance(label, str) else list(label)
        return [int(x) for x in parsed[:3]]
    except Exception:
        return None


def lab_to_rgb(lab) -> list:
    arr = np.array(lab, dtype=np.float64).reshape(1, 1, 3)
    rgb = skcolor.lab2rgb(arr) * 255
    return np.clip(rgb[0, 0], 0, 255).astype(int).tolist()


def _noise_free_clusters(centers_lab: np.ndarray) -> list[int]:
    """
    Retourne les indices des clusters non-bruit (ordre conservé = poids décroissant).

    Exclus :
      - Noir    : L < 15  (capuchon, fond non supprimé)
      - Blanc   : L > 90  (arrière-plan résiduel)
      - Peau    : 40 < L < 78  ET  5 < a < 25  ET  5 < b < 30
    """
    valid = []
    for i, lab in enumerate(centers_lab):
        L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
        if L < 15:
            continue
        if L > 90:
            continue
        if 40 < L < 78 and 5 < a < 25 and 5 < b < 30:
            continue
        valid.append(i)
    return valid


# ═══════════════════════════════════════════════════════════════
#  CONSTRUCTION DU DATASET
# ═══════════════════════════════════════════════════════════════

def build_dataset(df: pd.DataFrame, split_name: str = "") -> tuple:
    """
    Construit X (41 features) et y (index cluster 0..4) depuis un DataFrame.

    Filtre automatiquement Tier 0 + Tier 1.
    Ignore les produits sans cache NailShadeDetector ou sans label valide.

    Returns:
        X        : np.ndarray (N, 41)
        y        : np.ndarray (N,)  entiers 0..4
        meta     : list[dict]   métadonnées (image, manual_rgb, de_oracle)
        skipped  : int          produits ignorés
    """
    X_list, y_list, meta_list = [], [], []
    skipped = 0
    no_cache = 0

    label = f"[{split_name}] " if split_name else ""
    print(f"\n{label}Construction dataset sur {len(df)} produits...")

    # ── Filtres Tier 0 + Tier 1 ──────────────────────────────
    df = tier0.filter_normal(df)
    df = tier1.filter_normal(df)
    print(f"{label}Après filtrage : {len(df)} produits normaux")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{label}Features"):

        # Parse label manuel
        manual_rgb = parse_label(row.get("manual_label"))
        if manual_rgb is None:
            skipped += 1
            continue

        # Tier 3 : extraction features vision
        vision = tier3.extract(row["image_filename"])
        if vision is None:
            no_cache += 1
            skipped += 1
            continue

        centers_lab  = vision["centers_lab"]   # (K, 3) trié par poids ↓
        features_vis = vision["features"]       # (35,)

        # Tier 2 : features NLP
        features_nlp = tier2.extract_features(row, centers_lab)  # (6,)

        # Vecteur complet : 41 features
        X_vec = np.concatenate([features_vis, features_nlp]).astype(np.float32)

        # Label : cluster le plus proche du manual_label
        manual_lab = rgb_to_lab(manual_rgb)
        best_idx = int(np.argmin([
            delta_e(manual_lab, centers_lab[i]) for i in range(N_CLUSTERS)
        ]))
        best_de = delta_e(manual_lab, centers_lab[best_idx])

        X_list.append(X_vec)
        y_list.append(best_idx)
        meta_list.append({
            "image":      row["image_filename"],
            "manual_rgb": manual_rgb,
            "brand":      str(row.get("brand_name", "")),
            "shade":      str(row.get("shade_name", "")),
            "de_oracle":  round(best_de, 2),
            "centers_rgb": vision["centers_rgb"],
        })

    print(f"{label}Skipped : {skipped} ({no_cache} sans cache NailShadeDetector)")
    print(f"{label}Dataset : {len(X_list)} échantillons")

    if not X_list:
        return np.array([]), np.array([]), [], 0

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Distribution des labels
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"{label}Distribution classes : {dist}")

    return X, y, meta_list, skipped


# ═══════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

def train(save: bool = True) -> dict:
    """
    Entraîne XGBoost sur les produits normaux des 1261 labels.

    Pipeline :
      1. Build dataset complet
      2. Split train/test 80/20 stratifié
      3. Cross-validation 5-fold sur train
      4. Fit sur train complet
      5. Évaluation sur test
      6. Sauvegarde modèle + metadata

    Returns dict avec les résultats.
    """
    print("\n╔" + "═" * 64 + "╗")
    print("║" + "  TIER 4 — ENTRAÎNEMENT XGBOOST".center(64) + "║")
    print("╚" + "═" * 64 + "╝")

    # ── Charger les données labellisées ──────────────────────
    df = pd.read_parquet(LABELED_PARQUET)
    print(f"\nDataset chargé : {len(df)} lignes")

    # ── Construire X, y ──────────────────────────────────────
    X, y, meta, _ = build_dataset(df)

    if len(X) == 0:
        print("ERREUR : aucun échantillon. Vérifier le cache NailShadeDetector.")
        sys.exit(1)

    # ── Split train / test ────────────────────────────────────
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

    # ── Cross-validation 5-fold ───────────────────────────────
    print(f"\nCross-validation {CV_FOLDS}-fold sur train...")
    model_cv = XGBClassifier(**XGB_PARAMS)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"CV par fold : {[f'{s:.3f}' for s in cv_scores]}")

    # ── Fit sur train complet ─────────────────────────────────
    print("\nEntraînement final sur train complet...")
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    # ── Feature importance ────────────────────────────────────
    importance = model.feature_importances_
    feature_names = _build_feature_names()
    top_idx = np.argsort(-importance)[:10]
    print("\nTop 10 features :")
    for i in top_idx:
        print(f"  [{i:2d}] {feature_names[i]:<30} {importance[i]:.4f}")

    # ── Évaluation sur test ───────────────────────────────────
    results = evaluate_predictions(model, X_test, y_test, meta_test)

    # ── Sauvegarde ────────────────────────────────────────────
    if save:
        model_path = MODEL_DIR / "best_model.pkl"
        meta_path  = MODEL_DIR / "train_meta.json"
        test_path  = RESULT_DIR / "test_predictions.json"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        train_meta = {
            "n_train":      int(len(X_train)),
            "n_test":       int(len(X_test)),
            "cv_mean":      float(cv_scores.mean()),
            "cv_std":       float(cv_scores.std()),
            "feature_names": feature_names,
            "top_features": [
                {"name": feature_names[i], "importance": float(importance[i])}
                for i in top_idx
            ],
        }
        with open(meta_path, "w") as f:
            json.dump({**train_meta, **results}, f, indent=2)

        # Sauvegarde des prédictions test
        with open(test_path, "w") as f:
            json.dump(meta_test, f, indent=2)

        print(f"\nModèle sauvegardé → {model_path}")

    return results


# ═══════════════════════════════════════════════════════════════
#  ÉVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate_predictions(
    model: "XGBClassifier",
    X_test: np.ndarray,
    y_test: np.ndarray,
    meta_test: list,
) -> dict:
    """
    Évalue le modèle sur le test set.
    Calcule accuracy de sélection + ΔE colorimétrique.
    """
    probas = model.predict_proba(X_test)         # (N, K)
    preds  = probas.argmax(axis=1)               # (N,)
    confs  = probas.max(axis=1)                  # (N,)

    # ── Accuracy sélection cluster ────────────────────────────
    acc_selection = float(accuracy_score(y_test, preds))

    # ── ΔE colorimétrique ────────────────────────────────────
    de_all    = []
    de_above  = []  # produits avec confidence >= seuil
    quarantined = 0

    for i, (pred_idx, conf, m) in enumerate(zip(preds, confs, meta_test)):
        centers_rgb = m["centers_rgb"]
        manual_rgb  = m["manual_rgb"]

        pred_rgb = centers_rgb[pred_idx]
        pred_lab = rgb_to_lab(pred_rgb)
        man_lab  = rgb_to_lab(manual_rgb)
        de = delta_e(pred_lab, man_lab)

        de_all.append(de)
        m["de_predicted"] = round(de, 2)
        m["confidence"]   = round(float(conf), 3)
        m["pred_cluster"] = int(pred_idx)
        m["quarantined"]  = bool(conf < CONFIDENCE_THRESHOLD)

        if conf >= CONFIDENCE_THRESHOLD:
            de_above.append(de)
        else:
            quarantined += 1

    de_all   = np.array(de_all)
    de_above = np.array(de_above) if de_above else np.array([])

    print(f"\n{'─'*66}")
    print(f"  RÉSULTATS TEST ({len(X_test)} produits)")
    print(f"{'─'*66}")
    print(f"  Accuracy sélection cluster : {acc_selection*100:.2f}%")
    print(f"  Quarantinés (conf < {CONFIDENCE_THRESHOLD}) : "
          f"{quarantined} ({quarantined/len(X_test)*100:.1f}%)")

    print(f"\n  ── Tous les produits (y compris conf faible) ──")
    _print_de_stats(de_all)

    if len(de_above) > 0:
        print(f"\n  ── Produits prédits (conf >= {CONFIDENCE_THRESHOLD}) : "
              f"{len(de_above)} ──")
        _print_de_stats(de_above)

    results = {
        "acc_selection":    round(acc_selection * 100, 2),
        "n_test":           int(len(X_test)),
        "n_quarantined":    int(quarantined),
        "de_mean_all":      round(float(de_all.mean()), 2),
        "de_median_all":    round(float(np.median(de_all)), 2),
    }
    for thr in DELTA_E_THRESHOLDS:
        results[f"acc_de{thr}_all"] = round(float((de_all <= thr).mean() * 100), 2)
        if len(de_above) > 0:
            results[f"acc_de{thr}_predicted"] = round(
                float((de_above <= thr).mean() * 100), 2
            )

    return results


def _print_de_stats(de_arr: np.ndarray):
    print(f"    ΔE moyen  : {de_arr.mean():.2f}")
    print(f"    ΔE médian : {np.median(de_arr):.2f}")
    for thr in DELTA_E_THRESHOLDS:
        print(f"    @ΔE≤{thr:2d}   : {(de_arr <= thr).mean()*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
#  INFÉRENCE SUR UN PRODUIT
# ═══════════════════════════════════════════════════════════════

def load_model() -> Optional["XGBClassifier"]:
    """Charge le modèle sauvegardé."""
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(row: pd.Series, model: "XGBClassifier") -> dict:
    """
    Prédit la couleur d'un produit.

    Args:
        row   : Ligne du DataFrame (tous les champs requis)
        model : XGBClassifier entraîné

    Returns dict :
        tag            : TAG_OK ou TAG_* de quarantaine
        shade_lab      : [L, a, b] ou None
        shade_rgb      : [R, G, B] ou None
        confidence     : float [0, 1]
        color_word     : str | None  (mot détecté par NLP)
    """
    _none = {"shade_lab": None, "shade_rgb": None, "confidence": 0.0, "color_word": None}

    # ── Tier 0 ───────────────────────────────────────────────
    t0 = tier0.classify(row)

    if t0 == "kit_multicolor":
        return {**_none, "tag": TAG_KIT}

    if t0 == "incolore":
        return {**_none, "tag": TAG_INCOLORE}

    if t0 in ("nail_accessory", "volume_only"):
        return {**_none, "tag": TAG_KIT}

    # ── Tier 1 ───────────────────────────────────────────────
    t1 = tier1.classify(row)
    if t1 != "normal":
        return {**_none, "tag": TAG_TEXTURE}

    # ── Tier 3 ───────────────────────────────────────────────
    vision = tier3.extract(row["image_filename"])
    if vision is None:
        return {**_none, "tag": TAG_ANOMALIE}

    centers_lab = vision["centers_lab"]

    # ── Kit ambigu : arbitrage ΔE sur clusters non-bruit ────────
    # Filtre noir (L<15), blanc (L>90), peau avant comparaison.
    # Si < 2 clusters exploitables → monocolor par défaut (on garde).
    if t0 == "kit_ambiguous":
        valid = _noise_free_clusters(centers_lab)
        if len(valid) >= 2:
            de_top2 = delta_e(centers_lab[valid[0]], centers_lab[valid[1]])
            if de_top2 >= KIT_DE_THRESHOLD:
                return {**_none, "tag": TAG_KIT}
        # < 2 clusters nail valides OU ΔE faible → même couleur → continuer

    # ── Tier 2 ───────────────────────────────────────────────
    features_vis = vision["features"]
    features_nlp = tier2.extract_features(row, centers_lab)
    color_word   = tier2.get_color_word(row)

    # ── Tier 4 ───────────────────────────────────────────────
    X = np.concatenate([features_vis, features_nlp]).reshape(1, -1).astype(np.float32)
    proba = model.predict_proba(X)[0]
    best_idx   = int(proba.argmax())
    confidence = float(proba[best_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return {**_none, "tag": TAG_ANOMALIE, "confidence": confidence,
                "color_word": color_word}

    shade_lab = centers_lab[best_idx].tolist()
    shade_rgb = vision["centers_rgb"][best_idx]

    return {
        "tag":        TAG_OK,
        "shade_lab":  shade_lab,
        "shade_rgb":  shade_rgb,
        "confidence": confidence,
        "color_word": color_word,
    }


# ═══════════════════════════════════════════════════════════════
#  NOMS DES FEATURES (pour debug / importance)
# ═══════════════════════════════════════════════════════════════

def _build_feature_names() -> list[str]:
    names = []
    # BLOC A
    for i in range(N_CLUSTERS):
        for feat in ["L", "a", "b", "weight"]:
            names.append(f"C{i}_{feat}")
    # BLOC B
    for i in range(N_CLUSTERS):
        names.append(f"C{i}_chroma")
    # BLOC C
    for i in range(N_CLUSTERS):
        names.append(f"C{i}_disp")
    # BLOC D
    for i in range(N_CLUSTERS):
        names.append(f"C{i}_vpos")
    # NLP
    names.append("nlp_has_prior")
    for i in range(N_CLUSTERS):
        names.append(f"nlp_de_c{i}")
    assert len(names) == 41, f"Expected 41 feature names, got {len(names)}"
    return names


# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Entraîner et sauvegarder")
    parser.add_argument("--eval",  action="store_true", help="Évaluer le modèle sauvegardé")
    args = parser.parse_args()

    if args.train:
        train(save=True)
    elif args.eval:
        model = load_model()
        if model is None:
            print("Aucun modèle trouvé. Lancer d'abord --train")
            sys.exit(1)
        df = pd.read_parquet(LABELED_PARQUET)
        X, y, meta, _ = build_dataset(df)
        _, X_test, _, y_test, _, meta_test = train_test_split(
            X, y, meta,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
        )
        evaluate_predictions(model, X_test, y_test, meta_test)
    else:
        print("Usage : python -m src.clustering.tier4_xgboost --train | --eval")
