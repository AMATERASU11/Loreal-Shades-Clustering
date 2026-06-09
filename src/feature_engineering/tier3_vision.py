"""
Tier 3 — Vision Core
═══════════════════════════════════════════════════════════════════
Charge le cache NPZ (lab + rgb + y_norm), applique le center-crop,
lance KMeans K=5 en Lab, extrait 139 features.

Features (139 total) :
  Vision blocs A-P (105) :
    A  (20) L, a, b, weight × 5
    B  ( 5) chroma C*
    C  ( 5) dispersion std ΔE
    D  ( 5) vpos mean
    E  ( 5) skin flag strict
    F  (10) pairwise ΔE entre clusters
    G  ( 5) isolation (min ΔE vers autre cluster)
    H  ( 5) chroma bis
    I  ( 5) skin flag loose
    J  ( 5) chroma rank
    K  ( 5) proportion rank
    L  ( 5) background flag
    M  ( 5) vpos mean (absolu)
    N  ( 5) vpos std
    O  (10) dist noir / dist blanc
    P  ( 5) x_norm (neutre 0.5)
  HSV     ( 19) : H, S, V × 5 clusters + 4 dérivées
  NLP     (  6) : has_prior + ΔE × 5 clusters
  Vernis  (  9) : 5 scores + Lab gagnant + index
"""
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from skimage import color as skcolor

from src.utils.config import (
    CACHE_DIR,
    N_CLUSTERS,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    CENTER_CROP_TOP,
    CENTER_CROP_BOTTOM,
    MIN_PIXELS,
    N_TOTAL_FEATURES,
)


# ═══════════════════════════════════════════════════════════════
#  CHARGEMENT DU CACHE
# ═══════════════════════════════════════════════════════════════

def load_cache(image_filename: str) -> Optional[dict]:
    stem = Path(str(image_filename)).stem
    path = CACHE_DIR / f"{stem}.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path)
        return {
            "lab":    data["lab"].astype(np.float64),
            "rgb":    data["rgb"],
            "y_norm": data["y_norm"].astype(np.float64),
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  CENTER-CROP
# ═══════════════════════════════════════════════════════════════

def apply_crop(lab, rgb, y_norm):
    mask = (y_norm >= CENTER_CROP_TOP) & (y_norm <= (1.0 - CENTER_CROP_BOTTOM))
    return lab[mask], rgb[mask], y_norm[mask]


# ═══════════════════════════════════════════════════════════════
#  KMEANS + RÉORDONNANCEMENT PAR L* CROISSANT
# ═══════════════════════════════════════════════════════════════

def run_kmeans(lab_pixels: np.ndarray):
    km = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
    )
    km.fit(lab_pixels)
    centers_lab = km.cluster_centers_.copy()
    counts      = np.bincount(km.labels_, minlength=N_CLUSTERS)
    proportions = counts / len(lab_pixels)

    # Tri par L* croissant (identique methode_5)
    sorted_idx  = np.argsort(centers_lab[:, 0])
    centers_lab = centers_lab[sorted_idx]
    proportions = proportions[sorted_idx]
    inv_map     = np.argsort(sorted_idx)
    pixel_labels = inv_map[km.labels_]

    return centers_lab, proportions, pixel_labels


# ═══════════════════════════════════════════════════════════════
#  EXTRACTION 139 FEATURES
# ═══════════════════════════════════════════════════════════════

def compute_features(
    centers_lab: np.ndarray,
    proportions: np.ndarray,
    pixel_labels: np.ndarray,
    lab_pixels: np.ndarray,
    y_crop: np.ndarray,
    centers_rgb: list,
    nlp_features: np.ndarray,
) -> np.ndarray:
    K = N_CLUSTERS
    features = np.zeros(105, dtype=np.float32)
    idx = 0

    # BLOC A : L, a, b, weight
    for i in range(K):
        L, a, b = centers_lab[i]; w = proportions[i]
        features[idx:idx+4] = [L/100., (a+128.)/255., (b+128.)/255., w]; idx += 4

    # BLOC B : chroma
    for i in range(K):
        a, b = float(centers_lab[i, 1]), float(centers_lab[i, 2])
        features[idx] = np.sqrt(a**2 + b**2) / 100.; idx += 1

    # BLOC C : dispersion
    for i in range(K):
        mask_i = pixel_labels == i
        if mask_i.sum() < 2:
            features[idx] = 0.
        else:
            diffs = np.sqrt(np.sum((lab_pixels[mask_i] - centers_lab[i])**2, axis=1))
            features[idx] = float(diffs.std()) / 50.
        idx += 1

    # BLOC D : vpos mean (normalisé dans crop)
    y_min  = CENTER_CROP_TOP
    y_span = (1. - CENTER_CROP_BOTTOM) - CENTER_CROP_TOP
    for i in range(K):
        mask_i = pixel_labels == i
        features[idx] = ((float(y_crop[mask_i].mean()) - y_min) / y_span) if mask_i.sum() > 0 else 0.5
        idx += 1

    # BLOC E : skin flag strict
    for i in range(K):
        L, a, b = float(centers_lab[i, 0]), float(centers_lab[i, 1]), float(centers_lab[i, 2])
        features[idx] = 1. if (40 < L < 78 and 5 < a < 25 and 5 < b < 30) else 0.; idx += 1

    # BLOC F : pairwise ΔE
    for i in range(K):
        for j in range(i+1, K):
            features[idx] = float(np.sqrt(np.sum((centers_lab[i] - centers_lab[j])**2))) / 100.; idx += 1

    # BLOC G : isolation
    for i in range(K):
        others = [j for j in range(K) if j != i]
        features[idx] = min(float(np.sqrt(np.sum((centers_lab[i] - centers_lab[j])**2))) for j in others) / 100.; idx += 1

    # BLOC H : chroma bis
    for i in range(K):
        a, b = centers_lab[i, 1], centers_lab[i, 2]
        features[idx] = float(np.sqrt(a**2 + b**2)) / 100.; idx += 1

    # BLOC I : skin flag loose
    for i in range(K):
        L, a, b = centers_lab[i]
        features[idx] = float(35 < L < 78 and 3 < a < 25 and 3 < b < 30); idx += 1

    # BLOC J : chroma rank
    chromas = [float(np.sqrt(centers_lab[i, 1]**2 + centers_lab[i, 2]**2)) for i in range(K)]
    cr = np.argsort(np.argsort(chromas))
    for i in range(K):
        features[idx] = cr[i] / (K - 1); idx += 1

    # BLOC K : proportion rank
    pr = np.argsort(np.argsort(proportions))
    for i in range(K):
        features[idx] = pr[i] / (K - 1); idx += 1

    # BLOC L : background flag
    for i in range(K):
        L = centers_lab[i][0]
        features[idx] = float(L > 88 or L < 8); idx += 1

    # BLOC M : vpos mean absolu
    for i in range(K):
        mask_i = pixel_labels == i
        features[idx] = float(y_crop[mask_i].mean()) if mask_i.any() else 0.5; idx += 1

    # BLOC N : vpos std
    for i in range(K):
        mask_i = pixel_labels == i
        features[idx] = float(y_crop[mask_i].std()) if (mask_i.any() and mask_i.sum() > 1) else 0.3; idx += 1

    # BLOC O : dist noir / dist blanc
    for i in range(K):
        features[idx] = centers_lab[i][0] / 100.; idx += 1
    for i in range(K):
        features[idx] = (100. - centers_lab[i][0]) / 100.; idx += 1

    # BLOC P : x_norm (neutre 0.5)
    for i in range(K):
        features[idx] = 0.5; idx += 1

    assert idx == 105

    # HSV (19)
    hsv_feats = []
    for i in range(K):
        rgb_i = np.array(centers_rgb[i], dtype=np.float32) / 255.
        hsv_i = skcolor.rgb2hsv(rgb_i.reshape(1, 1, 3)).reshape(3)
        hsv_feats.extend([float(hsv_i[0]), float(hsv_i[1]), float(hsv_i[2])])
    sats = [float(skcolor.rgb2hsv(np.array(centers_rgb[i], dtype=np.float32).reshape(1, 1, 3) / 255.).reshape(3)[1]) for i in range(K)]
    hsv_feats += [sats[0] - sats[1], float(np.argmax(sats)) / 4., float(max(sats)), float(np.std(sats))]
    hsv_array = np.array(hsv_feats, dtype=np.float32)

    # Vernis score (9)
    vernis_scores = np.zeros(K, dtype=np.float32)
    for i in range(K):
        s = 0.
        L_i, a_i, b_i = float(centers_lab[i, 0]), float(centers_lab[i, 1]), float(centers_lab[i, 2])
        if 35 < L_i < 78 and 3 < a_i < 25 and 3 < b_i < 30: s -= 2.
        if L_i > 90: s -= 2.
        if L_i < 6:  s -= 1.5
        mask_i = pixel_labels == i
        if mask_i.any():
            y_mean_i = float(y_crop[mask_i].mean())
            y_std_i  = float(y_crop[mask_i].std()) if mask_i.sum() > 1 else 0.3
            if 0.25 <= y_mean_i <= 0.80: s += 1.
            if y_std_i < 0.20: s += 0.5
        if proportions[i] > 0.10: s += 0.5
        if float(np.sqrt(a_i**2 + b_i**2)) > 10: s += 0.5
        vernis_scores[i] = s

    best_v_idx = int(np.argmax(vernis_scores))
    best_v_L   = float(centers_lab[best_v_idx, 0]) / 100.
    best_v_a   = float(centers_lab[best_v_idx, 1] + 128.) / 255.
    best_v_b   = float(centers_lab[best_v_idx, 2] + 128.) / 255.
    v_min, v_max = vernis_scores.min(), vernis_scores.max()
    v_range = (v_max - v_min) if v_max > v_min else 1.
    vernis_feats = np.concatenate([
        (vernis_scores - v_min) / v_range,
        [best_v_L, best_v_a, best_v_b],
        [float(best_v_idx) / (K - 1)],
    ]).astype(np.float32)

    X_vec = np.concatenate([features, hsv_array, nlp_features, vernis_feats]).astype(np.float32)
    assert len(X_vec) == N_TOTAL_FEATURES, f"Feature count mismatch: {len(X_vec)} != {N_TOTAL_FEATURES}"
    return X_vec


# ═══════════════════════════════════════════════════════════════
#  PIPELINE COMPLET TIER 3
# ═══════════════════════════════════════════════════════════════

def extract(image_filename: str, nlp_features: Optional[np.ndarray] = None) -> Optional[dict]:
    """
    Pipeline complet Tier 3 pour une image.

    Args:
        image_filename : nom du fichier image
        nlp_features   : (6,) features NLP depuis tier2 (zeros si None)

    Returns dict ou None :
        features    : np.ndarray (139,)
        centers_lab : np.ndarray (K, 3)
        centers_rgb : list[list[int]]
        proportions : np.ndarray (K,)
        pixel_labels: np.ndarray (N,)
        lab_crop    : np.ndarray (N, 3)
        y_crop      : np.ndarray (N,)
    """
    cache = load_cache(image_filename)
    if cache is None:
        return None

    lab_crop, rgb_crop, y_crop = apply_crop(cache["lab"], cache["rgb"], cache["y_norm"])
    if len(lab_crop) < MIN_PIXELS:
        return None

    centers_lab, proportions, pixel_labels = run_kmeans(lab_crop)

    # Centres RGB (moyenne pixels du cluster)
    centers_rgb = []
    for i in range(N_CLUSTERS):
        mask_i = pixel_labels == i
        if mask_i.any():
            centers_rgb.append([int(v) for v in rgb_crop[mask_i].mean(axis=0)])
        else:
            arr = centers_lab[i].reshape(1, 1, 3)
            r = (skcolor.lab2rgb(arr).reshape(3) * 255).clip(0, 255).astype(int)
            centers_rgb.append([int(v) for v in r])

    if nlp_features is None:
        nlp_features = np.zeros(6, dtype=np.float32)

    features = compute_features(
        centers_lab, proportions, pixel_labels,
        lab_crop, y_crop, centers_rgb, nlp_features,
    )

    return {
        "features":    features,
        "centers_lab": centers_lab,
        "centers_rgb": centers_rgb,
        "proportions": proportions,
        "pixel_labels": pixel_labels,
        "lab_crop":    lab_crop,
        "y_crop":      y_crop,
    }
