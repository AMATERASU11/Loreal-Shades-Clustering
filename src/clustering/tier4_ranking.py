"""
Tier 4 — ShadeNail Ranking (XGBoost Regressor)
═══════════════════════════════════════════════════════════════════
Approche RANKING : pour chaque image, on génère 5 lignes (1 par cluster),
et on prédit ΔE(cluster_i, vrai_label) par régression. Le cluster avec
le ΔE prédit le plus faible est sélectionné.

34 features par ligne (focal cluster + contexte + image-level + NLP).

Modèle : XGBoost Regressor.

Usage (inférence uniquement dans ce repo) :
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
    TAG_OK, TAG_KIT, TAG_INCOLORE, TAG_TEXTURE, TAG_ANOMALIE,
    KIT_DE_THRESHOLD,
)
import src.feature_engineering.tier0_gatekeeper as tier0
import src.feature_engineering.tier1_texture    as tier1
import src.feature_engineering.tier2_nlp_prior  as tier2
import src.feature_engineering.tier3_vision     as tier3


# ═══════════════════════════════════════════════════════════════
#  CONSTANTES
# ═══════════════════════════════════════════════════════════════

N_RANKING_FEATURES = 34

FEATURE_NAMES = [
    "focal_L", "focal_a", "focal_b", "focal_chroma", "focal_weight",
    "focal_disp", "focal_vpos", "focal_is_skin", "focal_is_dark", "focal_is_light",
    "focal_rank_weight", "focal_rank_chroma",
    "others_L", "others_a", "others_b", "others_max_chroma",
    "n_nail_clusters", "de_to_dominant", "de_to_most_chromatic",
    "img_L_mean", "img_a_mean", "img_b_mean", "img_weight_entropy", "img_chroma_dominant",
    "nlp_has_prior", "nlp_de_focal", "nlp_rank_focal",
    "focal_is_bg", "de_to_img_mean", "chroma_pct", "disp_norm",
    "vpos_centered", "nlp_de_norm", "is_nlp_best",
]


# ═══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def rgb_to_lab(rgb) -> np.ndarray:
    arr = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
    return skcolor.rgb2lab(arr / 255.0)[0, 0]


def lab_to_rgb(lab) -> list:
    arr = np.array(lab, dtype=np.float64).reshape(1, 1, 3)
    rgb = skcolor.lab2rgb(arr) * 255
    return np.clip(rgb[0, 0], 0, 255).astype(int).tolist()


def delta_e(lab1, lab2) -> float:
    return float(np.sqrt(np.sum((np.array(lab1) - np.array(lab2)) ** 2)))


def _is_skin(L, a, b):
    return (40 < L < 78) and (5 < a < 25) and (5 < b < 30)


def _entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-8]
    return float(-(p * np.log(p)).sum())


def _noise_free_clusters(centers_lab: np.ndarray) -> list[int]:
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
#  CONSTRUCTION DES FEATURES RANKING (34 par cluster)
# ═══════════════════════════════════════════════════════════════

def build_cluster_rows(
    centers_lab: np.ndarray,
    proportions: np.ndarray,
    pixel_labels: np.ndarray,
    lab_pixels: np.ndarray,
    y_crop: np.ndarray,
    nlp_prior_lab: Optional[np.ndarray],
    has_nlp_prior: float,
) -> np.ndarray:
    """
    Pour chaque cluster i (0..K-1), produit une ligne de 34 features.

    Returns:
        np.ndarray (K, 34)
    """
    K = len(centers_lab)
    weights = proportions
    L_arr = centers_lab[:, 0]
    a_arr = centers_lab[:, 1]
    b_arr = centers_lab[:, 2]
    chroma_arr = np.sqrt(a_arr**2 + b_arr**2)

    # Moyennes globales pondérées
    L_mean_w = float(np.sum(L_arr * weights))
    a_mean_w = float(np.sum(a_arr * weights))
    b_mean_w = float(np.sum(b_arr * weights))
    weight_entropy = _entropy(weights)
    image_chroma_dominant = float(chroma_arr[0])

    # Rangs
    rank_weight = np.argsort(-weights).argsort()
    rank_chroma = np.argsort(-chroma_arr).argsort()

    # Clusters non-bruit
    is_noise = np.array([
        _is_skin(L_arr[i], a_arr[i], b_arr[i]) or L_arr[i] < 15 or L_arr[i] > 90
        for i in range(K)
    ], dtype=bool)
    n_nail_clusters = int((~is_noise).sum())

    # ΔE prior NLP
    if nlp_prior_lab is not None:
        nlp_de_per_cluster = np.array([
            delta_e(nlp_prior_lab, centers_lab[i]) for i in range(K)
        ])
        nlp_rank = np.argsort(nlp_de_per_cluster).argsort()
    else:
        nlp_de_per_cluster = np.zeros(K)
        nlp_rank = np.zeros(K, dtype=int)

    # Dispersions par cluster
    disp_per_cluster = np.zeros(K)
    for i in range(K):
        mask_i = pixel_labels == i
        if mask_i.sum() > 1:
            pixels_i = lab_pixels[mask_i]
            de_i = np.sqrt(np.sum((pixels_i - centers_lab[i]) ** 2, axis=1))
            disp_per_cluster[i] = float(de_i.std())

    # vpos par cluster
    vpos_per_cluster = np.full(K, 0.5)
    for i in range(K):
        mask_i = pixel_labels == i
        if mask_i.sum() > 0:
            vpos_per_cluster[i] = float(y_crop[mask_i].mean())

    # Pré-calculs pour nouvelles features
    chroma_max_image = max(float(chroma_arr.max()), 1e-6)
    disp_max_image = max(float(disp_per_cluster.max()), 1e-6)
    nlp_best_idx = int(np.argmin(nlp_de_per_cluster)) if nlp_prior_lab is not None else -1
    nlp_de_min = float(nlp_de_per_cluster.min()) if nlp_prior_lab is not None else 0.0
    img_mean_lab = np.array([L_mean_w, a_mean_w, b_mean_w])

    rows = []
    for i in range(K):
        L_i, a_i, b_i = float(L_arr[i]), float(a_arr[i]), float(b_arr[i])
        c_i = float(chroma_arr[i])

        # Autres clusters
        others_mask = np.arange(K) != i
        L_others = float(L_arr[others_mask].mean())
        a_others = float(a_arr[others_mask].mean())
        b_others = float(b_arr[others_mask].mean())
        chroma_others_max = float(chroma_arr[others_mask].max())

        # Dominant non-focal
        for j in range(K):
            if j != i:
                dominant_non_focal = j
                break
        de_to_dominant = delta_e(centers_lab[i], centers_lab[dominant_non_focal])

        # Plus chromatique non-focal
        chroma_others_arr = chroma_arr.copy()
        chroma_others_arr[i] = -1
        most_chromatic_non_focal = int(np.argmax(chroma_others_arr))
        de_to_most_chromatic = delta_e(centers_lab[i], centers_lab[most_chromatic_non_focal])

        # Nouvelles features
        is_bg = 1.0 if (L_i > 92 and c_i < 8) else 0.0
        de_to_img_mean = delta_e(centers_lab[i], img_mean_lab)
        chroma_pct = 1.0 - (c_i / chroma_max_image)
        disp_norm = float(disp_per_cluster[i]) / disp_max_image
        vpos_centered = float(vpos_per_cluster[i]) - 0.5
        nlp_de_norm = (float(nlp_de_per_cluster[i]) - nlp_de_min) / 100.0 if has_nlp_prior else 0.0
        is_nlp_best = 1.0 if (has_nlp_prior and i == nlp_best_idx) else 0.0

        row = [
            L_i / 100.0,
            (a_i + 128.0) / 255.0,
            (b_i + 128.0) / 255.0,
            c_i / 100.0,
            float(weights[i]),
            float(disp_per_cluster[i]) / 50.0,
            float(vpos_per_cluster[i]),
            1.0 if _is_skin(L_i, a_i, b_i) else 0.0,
            1.0 if L_i < 15 else 0.0,
            1.0 if L_i > 90 else 0.0,
            float(rank_weight[i]) / max(K - 1, 1),
            float(rank_chroma[i]) / max(K - 1, 1),
            L_others / 100.0,
            (a_others + 128.0) / 255.0,
            (b_others + 128.0) / 255.0,
            chroma_others_max / 100.0,
            float(n_nail_clusters) / K,
            float(de_to_dominant) / 100.0,
            float(de_to_most_chromatic) / 100.0,
            L_mean_w / 100.0,
            (a_mean_w + 128.0) / 255.0,
            (b_mean_w + 128.0) / 255.0,
            weight_entropy / np.log(K),
            image_chroma_dominant / 100.0,
            has_nlp_prior,
            float(nlp_de_per_cluster[i]) / 100.0,
            float(nlp_rank[i]) / max(K - 1, 1),
            is_bg,
            float(de_to_img_mean) / 100.0,
            chroma_pct,
            disp_norm,
            vpos_centered,
            nlp_de_norm,
            is_nlp_best,
        ]
        rows.append(row)

    return np.array(rows, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  CHARGEMENT MODÈLES
# ═══════════════════════════════════════════════════════════════

def load_model():
    """
    Charge le modèle ShadeNail (XGBoost).

    Returns XGBRegressor ou None si absent.
    """
    xgb_path = MODEL_DIR / "shadenail_xgb.pkl"

    if not xgb_path.exists():
        return None

    with open(xgb_path, "rb") as f:
        return pickle.load(f)


# ═══════════════════════════════════════════════════════════════
#  PRÉDICTION RANKING
# ═══════════════════════════════════════════════════════════════

def predict_ranking(
    centers_lab: np.ndarray,
    proportions: np.ndarray,
    pixel_labels: np.ndarray,
    lab_pixels: np.ndarray,
    y_crop: np.ndarray,
    nlp_prior_lab: Optional[np.ndarray],
    has_nlp: float,
    model,
) -> dict:
    """
    Prédiction par ranking : pour chaque cluster, prédit ΔE, choisit argmin.

    Returns:
        best_idx   : int — index du cluster sélectionné
        de_pred    : np.ndarray (K,) — ΔE prédit par cluster
        confidence : float — écart normalisé entre best et 2nd best
    """
    X_rows = build_cluster_rows(
        centers_lab, proportions, pixel_labels,
        lab_pixels, y_crop, nlp_prior_lab, has_nlp,
    )  # (K, 34)

    de_pred = model.predict(X_rows)  # (K,)

    best_idx = int(de_pred.argmin())

    # Confidence : écart entre le meilleur et le 2ème meilleur
    sorted_de = np.sort(de_pred)
    gap = float(sorted_de[1] - sorted_de[0]) if len(sorted_de) > 1 else 0.0
    # Normaliser : gap > 10 → confidence ~1.0, gap = 0 → confidence ~0.5
    confidence = min(1.0, 0.5 + gap / 20.0)

    return {
        "best_idx": best_idx,
        "de_pred": de_pred,
        "confidence": confidence,
    }


# ═══════════════════════════════════════════════════════════════
#  PRÉDICTION COMPLÈTE (comme tier4_xgboost.predict)
# ═══════════════════════════════════════════════════════════════

def predict(row: pd.Series, model) -> dict:
    """
    Prédit la couleur d'un produit via ShadeNail ranking.

    Args:
        row   : Ligne du DataFrame (product_id, brand_name, shade_name, image_filename)
        model : XGBRegressor entraîné

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

    # ── Tier 1 ────────────────────────────────────────────────
    t1 = tier1.classify(row)
    if t1 != "normal":
        return {**_none, "tag": TAG_TEXTURE}

    # ── Tier 3 — Vision (KMeans K=5) ─────────────────────────
    cache = tier3.load_cache(row["image_filename"])
    if cache is None:
        return {**_none, "tag": TAG_ANOMALIE}

    lab_full = cache["lab"]
    y_norm = cache["y_norm"]
    lab_crop, rgb_crop, y_crop = tier3.apply_crop(lab_full, cache["rgb"], y_norm)

    if len(lab_crop) < 50:
        return {**_none, "tag": TAG_ANOMALIE}

    centers_lab, proportions, pixel_labels = tier3.run_kmeans(lab_crop)

    # ── Kit ambigu : arbitrage ΔE sur clusters non-bruit ─────
    if t0 == "kit_ambiguous":
        valid = _noise_free_clusters(centers_lab)
        if len(valid) >= 2:
            de_top2 = delta_e(centers_lab[valid[0]], centers_lab[valid[1]])
            if de_top2 >= KIT_DE_THRESHOLD:
                return {**_none, "tag": TAG_KIT}

    # ── Tier 2 — NLP Prior ───────────────────────────────────
    color_word = tier2.get_color_word(row)
    if color_word and color_word in tier2.COLOR_TO_LAB:
        nlp_prior_lab = np.array(tier2.COLOR_TO_LAB[color_word])
        has_nlp = 1.0
    else:
        nlp_prior_lab = None
        has_nlp = 0.0

    # ── Tier 4 — ShadeNail Ranking ───────────────────────────
    ranking = predict_ranking(
        centers_lab, proportions, pixel_labels,
        lab_crop, y_crop, nlp_prior_lab, has_nlp, model,
    )

    best_idx = ranking["best_idx"]
    confidence = ranking["confidence"]

    if confidence < CONFIDENCE_THRESHOLD:
        return {**_none, "tag": TAG_ANOMALIE, "confidence": confidence,
                "color_word": color_word}

    shade_lab = centers_lab[best_idx].tolist()
    shade_rgb = lab_to_rgb(centers_lab[best_idx])

    return {
        "tag":        TAG_OK,
        "shade_lab":  shade_lab,
        "shade_rgb":  shade_rgb,
        "confidence": confidence,
        "color_word": color_word,
    }
