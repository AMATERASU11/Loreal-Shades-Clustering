"""
Tier 3 — Vision Core
═══════════════════════════════════════════════════════════════════
Charge le cache NailShadeDetector (lab + rgb + y_norm), applique le center-crop,
lance KMeans K=5 en Lab, extrait 35 features vision.

Features (35 total) :
  BLOC A (20) : L, a, b, weight × 5 clusters  — couleur + dominance
  BLOC B ( 5) : chroma C* = √(a²+b²) × 5      — saturation
  BLOC C ( 5) : dispersion std ΔE × 5          — texture / uniformité
  BLOC D ( 5) : position verticale vpos × 5    — localisation bouchon/vernis

IMPORTANT — Réordonnancement après KMeans :
  sorted_idx = np.argsort(-proportions)
  Propager sur : centers_lab, proportions, ET pixel_labels
  (correction demandée explicitement par l'utilisateur)
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
    N_VISION_FEATURES,
)


# ═══════════════════════════════════════════════════════════════
#  CHARGEMENT DU CACHE
# ═══════════════════════════════════════════════════════════════

def load_cache(image_filename: str) -> Optional[dict]:
    """
    Charge le cache NailShadeDetector pour une image.

    Returns dict avec 'lab', 'rgb', 'y_norm' ou None si absent.
    """
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
#  CENTER-CROP SUR Y_NORM
# ═══════════════════════════════════════════════════════════════

def apply_crop(lab: np.ndarray, rgb: np.ndarray, y_norm: np.ndarray):
    """
    Filtre les pixels par position verticale.
    Garde uniquement : CENTER_CROP_TOP <= y_norm <= (1 - CENTER_CROP_BOTTOM)

    Args:
        lab    : (N, 3) Lab pixels
        rgb    : (N, 3) RGB pixels
        y_norm : (N,)   positions verticales normalisées [0,1]

    Returns:
        lab_crop, rgb_crop, y_crop (pixels filtrés)
    """
    mask = (y_norm >= CENTER_CROP_TOP) & (y_norm <= (1.0 - CENTER_CROP_BOTTOM))
    return lab[mask], rgb[mask], y_norm[mask]


# ═══════════════════════════════════════════════════════════════
#  KMEANS + RÉORDONNANCEMENT
# ═══════════════════════════════════════════════════════════════

def run_kmeans(lab_pixels: np.ndarray):
    """
    KMeans K=5 en espace Lab + tri par poids décroissant.

    IMPORTANT : pixel_labels remappés correctement après tri
    (np.argsort(sorted_idx)[km.labels_] — correction requise).

    Returns:
        centers_lab    : (K, 3)  centres Lab triés par poids ↓
        proportions    : (K,)    poids de chaque cluster
        pixel_labels   : (N,)    label de chaque pixel (remappé, 0=dominant)
    """
    km = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
    )
    km.fit(lab_pixels)

    centers_lab = km.cluster_centers_.copy()          # (K, 3)
    raw_labels  = km.labels_                          # (N,)  original cluster idx
    counts      = np.bincount(raw_labels, minlength=N_CLUSTERS)
    proportions = counts / len(lab_pixels)            # (K,)

    # ── Tri par poids décroissant ──────────────────────────
    sorted_idx   = np.argsort(-proportions)           # ex: [2, 0, 4, 1, 3]
    centers_lab  = centers_lab[sorted_idx]
    proportions  = proportions[sorted_idx]

    # ── Remappage des pixel_labels ─────────────────────────
    # inv_map[orig_cluster_idx] = new_sorted_position
    # np.argsort(sorted_idx) donne la permutation inverse
    inv_map      = np.argsort(sorted_idx)             # ex: [1, 3, 0, 4, 2]
    pixel_labels = inv_map[raw_labels]                # (N,)  remappé

    return centers_lab, proportions, pixel_labels


# ═══════════════════════════════════════════════════════════════
#  EXTRACTION DES 35 FEATURES
# ═══════════════════════════════════════════════════════════════

def _delta_e_batch(pixels: np.ndarray, center: np.ndarray) -> np.ndarray:
    """ΔE CIE76 entre chaque pixel et un centre Lab."""
    return np.sqrt(np.sum((pixels - center) ** 2, axis=1))


def compute_features(
    centers_lab: np.ndarray,    # (K, 3)
    proportions: np.ndarray,    # (K,)
    pixel_labels: np.ndarray,   # (N,)
    lab_pixels: np.ndarray,     # (N, 3)
    y_crop: np.ndarray,         # (N,)  positions verticales après crop
) -> np.ndarray:
    """
    Calcule les 35 features vision.

    Returns:
        features : np.ndarray (35,)
    """
    K = N_CLUSTERS
    features = np.zeros(N_VISION_FEATURES, dtype=np.float32)
    idx = 0

    for i in range(K):
        L, a, b = centers_lab[i]
        w        = proportions[i]

        # ── BLOC A : base (L, a, b, weight) ──────────────────
        features[idx + 0] = L / 100.0              # L normalisé [0,1]
        features[idx + 1] = (a + 128.0) / 255.0   # a normalisé [0,1]
        features[idx + 2] = (b + 128.0) / 255.0   # b normalisé [0,1]
        features[idx + 3] = w                       # poids [0,1]
        idx += 4

    # ── BLOC B : chroma C* par cluster ───────────────────────
    for i in range(K):
        _, a, b = centers_lab[i]
        chroma = np.sqrt(a**2 + b**2)
        features[idx] = chroma / 100.0             # normalisé (C* max ≈ 128)
        idx += 1

    # ── BLOC C : dispersion intra-cluster ────────────────────
    for i in range(K):
        mask_i = pixel_labels == i
        n_i    = mask_i.sum()
        if n_i < 2:
            features[idx] = 0.0
        else:
            pixels_i = lab_pixels[mask_i]
            # ΔE moyen de chaque pixel par rapport au centre
            de_i = _delta_e_batch(pixels_i, centers_lab[i])
            features[idx] = float(de_i.std()) / 50.0   # normalisé
        idx += 1

    # ── BLOC D : position verticale (vpos) ───────────────────
    # y_crop est dans [CENTER_CROP_TOP, 1 - CENTER_CROP_BOTTOM]
    # On normalise dans [0, 1] sur cet intervalle
    y_min  = CENTER_CROP_TOP
    y_span = (1.0 - CENTER_CROP_BOTTOM) - CENTER_CROP_TOP

    for i in range(K):
        mask_i = pixel_labels == i
        if mask_i.sum() == 0:
            features[idx] = 0.5    # valeur neutre
        else:
            raw_vpos = float(y_crop[mask_i].mean())
            features[idx] = (raw_vpos - y_min) / y_span  # [0, 1]
        idx += 1

    assert idx == N_VISION_FEATURES, f"Feature count mismatch: {idx} != {N_VISION_FEATURES}"
    return features


# ═══════════════════════════════════════════════════════════════
#  PIPELINE COMPLET TIER 3
# ═══════════════════════════════════════════════════════════════

def extract(image_filename: str) -> Optional[dict]:
    """
    Pipeline complet Tier 3 pour une image.

    Args:
        image_filename : nom du fichier image (ex: "abc123.jpeg")

    Returns dict ou None :
        {
          'features'    : np.ndarray (35,)       features vision
          'centers_lab' : np.ndarray (K, 3)      centres Lab triés
          'centers_rgb' : list[list[int]]         centres RGB triés
          'proportions' : np.ndarray (K,)         poids triés
          'n_pixels'    : int                     pixels utilisés
        }
    """
    # ── Chargement cache ──────────────────────────────────────
    cache = load_cache(image_filename)
    if cache is None:
        return None

    lab    = cache["lab"]
    rgb    = cache["rgb"]
    y_norm = cache["y_norm"]

    # ── Center-crop ───────────────────────────────────────────
    lab_crop, rgb_crop, y_crop = apply_crop(lab, rgb, y_norm)

    if len(lab_crop) < MIN_PIXELS:
        return None

    # ── KMeans ────────────────────────────────────────────────
    centers_lab, proportions, pixel_labels = run_kmeans(lab_crop)

    # ── Centres RGB (pour référence) ──────────────────────────
    def lab_to_rgb(lab_vec):
        arr = np.array(lab_vec, dtype=np.float64).reshape(1, 1, 3)
        rgb_arr = skcolor.lab2rgb(arr) * 255
        return np.clip(rgb_arr[0, 0], 0, 255).astype(int).tolist()

    centers_rgb = [lab_to_rgb(c) for c in centers_lab]

    # ── 35 features vision ────────────────────────────────────
    features = compute_features(
        centers_lab, proportions, pixel_labels, lab_crop, y_crop
    )

    return {
        "features":    features,
        "centers_lab": centers_lab,
        "centers_rgb": centers_rgb,
        "proportions": proportions,
        "n_pixels":    len(lab_crop),
    }
