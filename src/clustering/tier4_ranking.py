"""
Tier 4 — LightGBM (139 features)
═══════════════════════════════════════════════════════════════════
Prédit la classe de luminosité (0-4) qui sélectionne le cluster KMeans
correspondant à la couleur du vernis.

139 features : 105 vision (blocs A-P) + 19 HSV + 6 NLP + 9 vernis_score
Modèle : LightGBM classifier (identique Méthode 5, CV 89.6%)

Usage (inférence) :
  from src.clustering.tier4_ranking import load_model, predict
  model = load_model()
  result = predict(row, model)
"""
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from skimage import color as skcolor

from src.utils.config import (
    MODEL_DIR, N_CLUSTERS,
    CONFIDENCE_THRESHOLD,
    TAG_OK, TAG_KIT, TAG_INCOLORE, TAG_TEXTURE, TAG_CAT_EYE, TAG_ANOMALIE,
    KIT_DE_THRESHOLD,
)
import src.feature_engineering.tier0_gatekeeper as tier0
import src.feature_engineering.tier1_texture    as tier1
import src.feature_engineering.tier2_nlp_prior  as tier2
import src.feature_engineering.tier3_vision     as tier3


# ═══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def lab_to_rgb(lab) -> list:
    arr = np.array(lab, dtype=np.float64).reshape(1, 1, 3)
    rgb = skcolor.lab2rgb(arr) * 255
    return np.clip(rgb[0, 0], 0, 255).astype(int).tolist()

def delta_e(lab1, lab2) -> float:
    return float(np.sqrt(np.sum((np.array(lab1) - np.array(lab2))**2)))

def _noise_free_clusters(centers_lab: np.ndarray) -> list:
    valid = []
    for i, lab in enumerate(centers_lab):
        L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
        if L < 15 or L > 90:
            continue
        if 40 < L < 78 and 5 < a < 25 and 5 < b < 30:
            continue
        valid.append(i)
    return valid


# ═══════════════════════════════════════════════════════════════
#  CHARGEMENT MODELE
# ═══════════════════════════════════════════════════════════════

def load_model(path: Optional[Path] = None):
    model_path = path or MODEL_DIR / "best_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════
#  PRÉDICTION COMPLÈTE
# ═══════════════════════════════════════════════════════════════

def predict(row: pd.Series, model) -> dict:
    """
    Prédit la couleur d'un produit via LightGBM 139 features.

    Returns dict :
        tag        : TAG_OK ou TAG_* de quarantaine
        shade_lab  : [L, a, b] ou None
        shade_rgb  : [R, G, B] ou None
        confidence : float [0, 1]
        color_word : str | None
    """
    _none = {"shade_lab": None, "shade_rgb": None, "confidence": 0.0, "color_word": None}

    # ── Tier 0 ────────────────────────────────────────────────
    t0 = tier0.classify(row)
    if t0 == "kit_multicolor":
        return {**_none, "tag": TAG_KIT}
    if t0 == "incolore":
        return {**_none, "tag": TAG_INCOLORE}
    if t0 in ("nail_accessory", "volume_only"):
        return {**_none, "tag": TAG_KIT}

    # ── Tier 1 ────────────────────────────────────────────────
    t1 = tier1.classify(row)
    if t1 == "texture_complexe":
        return {**_none, "tag": TAG_TEXTURE}

    # ── Cat eye : NLP prior d'abord ───────────────────────────
    if t1 == "cat_eye":
        color_word = tier2.get_color_word(row)
        if color_word and color_word in tier2.COLOR_TO_LAB:
            prior_lab = np.array(tier2.COLOR_TO_LAB[color_word])
            prior_rgb = lab_to_rgb(prior_lab.tolist())
            return {
                "tag": TAG_CAT_EYE, "confidence": 1.0,
                "shade_lab": prior_lab.tolist(),
                "shade_rgb": prior_rgb,
                "color_word": color_word,
            }

    # ── Tier 2 — NLP prior ────────────────────────────────────
    color_word   = tier2.get_color_word(row)
    nlp_features = tier2.extract_features(row, np.zeros((N_CLUSTERS, 3)))

    # ── Tier 3 — Vision + 139 features ───────────────────────
    vision = tier3.extract(row["image_filename"], nlp_features=nlp_features)
    if vision is None:
        return {**_none, "tag": TAG_ANOMALIE, "color_word": color_word}

    centers_lab  = vision["centers_lab"]
    centers_rgb  = vision["centers_rgb"]
    proportions  = vision["proportions"]
    pixel_labels = vision["pixel_labels"]
    lab_crop     = vision["lab_crop"]
    y_crop       = vision["y_crop"]

    # Recalcul features NLP avec vrais centres (tier2 a besoin des centres)
    nlp_features = tier2.extract_features(row, centers_lab)
    # Recalcule features complètes avec NLP correct
    from src.feature_engineering.tier3_vision import compute_features
    X_vec = compute_features(
        centers_lab, proportions, pixel_labels,
        lab_crop, y_crop, centers_rgb, nlp_features,
    )

    # ── Kit ambigu : arbitrage ΔE ─────────────────────────────
    if t0 == "kit_ambiguous":
        valid = _noise_free_clusters(centers_lab)
        if len(valid) >= 2:
            de_top2 = delta_e(centers_lab[valid[0]], centers_lab[valid[1]])
            if de_top2 >= KIT_DE_THRESHOLD:
                return {**_none, "tag": TAG_KIT, "color_word": color_word}

    # ── Tier 4 — LightGBM ────────────────────────────────────
    proba      = model.predict_proba(X_vec.reshape(1, -1))[0]
    best_idx   = int(proba.argmax())
    confidence = float(proba[best_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return {**_none, "tag": TAG_ANOMALIE, "confidence": confidence,
                "color_word": color_word}

    shade_lab = centers_lab[best_idx].tolist()
    shade_rgb = centers_rgb[best_idx]

    return {
        "tag":        TAG_OK,
        "shade_lab":  shade_lab,
        "shade_rgb":  shade_rgb,
        "confidence": round(confidence, 4),
        "color_word": color_word,
    }
