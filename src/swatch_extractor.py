"""
src/swatch_extractor.py
-----------------------
Classe SwatchExtractor : extrait la couleur Lab* représentative de chaque image.

Stratégie (inspirée du notebook) :
  1. Focus centre de l'image (évite packaging bords)
  2. Masque peau HSV ∩ YCrCb (double filtre robuste)
  3. Garde le plus grand composant connexe
  4. Médiane Lab* (résistante aux reflets)
  Fallback : filtre luminosité simple si pas assez de pixels peau
"""
import logging
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
tqdm.pandas()


# ------------------------------------------------------------------
# Fonctions utilitaires pures (pas de state)
# ------------------------------------------------------------------

def _lab_opencv_to_labstar(lab_opencv: np.ndarray) -> np.ndarray:
    """
    Convertit Lab OpenCV (L:0-255, a/b:0-255 centrés à 128)
    vers Lab* standard (L*:0-100, a*,b* centrés à 0).
    """
    lab = lab_opencv.astype(np.float32).copy()
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab[..., 1] - 128.0
    lab[..., 2] = lab[..., 2] - 128.0
    return lab


def _remove_white_black(img_bgr: np.ndarray,
                        white_thresh: int = 245,
                        black_thresh: int = 12) -> np.ndarray:
    """Masque binaire 0/255 : exclut fond blanc et zones trop sombres."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    keep = (L < white_thresh) & (L > black_thresh)
    return (keep.astype(np.uint8) * 255)


def _skin_mask_hsv_ycrcb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Masque peau = intersection HSV + YCrCb.
    Cible les teintes beige/orange/marron des produits teint.
    """
    # HSV : rouge/orange
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = (
        cv2.inRange(hsv,
                    np.array([0, 20, 30], dtype=np.uint8),
                    np.array([35, 255, 255], dtype=np.uint8))
        | cv2.inRange(hsv,
                      np.array([170, 20, 30], dtype=np.uint8),
                      np.array([180, 255, 255], dtype=np.uint8))
    )

    # YCrCb : plage classique teintes peau
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    mask_y = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    mask = cv2.bitwise_and(mask_hsv, mask_y)

    # Nettoyage morphologique
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Garde uniquement le plus grand composant connexe du masque."""
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return (binary * 255).astype(np.uint8)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8) * 255


# ------------------------------------------------------------------
# Classe principale
# ------------------------------------------------------------------

class SwatchExtractor:
    """
    Extrait une couleur Lab* représentative pour chaque image produit.

    Usage :
        extractor = SwatchExtractor(config)
        df = extractor.transform(df)   # ajoute les colonnes L, a, b
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.resize: int = config["preprocessing"].get("image_resize", 220)
        self.min_skin_pixels: int = config["swatch_extraction"].get("min_skin_pixels", 120)
        self.prefer_center: bool = config["swatch_extraction"].get("prefer_center", True)
        self.white_thresh: int = config["preprocessing"].get("white_threshold", 245)
        self.black_thresh: int = config["preprocessing"].get("black_threshold", 12)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applique l'extraction sur toutes les images valides."""
        df = df.copy()
        df["L"] = np.nan
        df["a"] = np.nan
        df["b"] = np.nan

        valid_mask = df["image_load_ok"] == 1
        logger.info("Extraction swatch sur %d images valides...", valid_mask.sum())

        results = df.loc[valid_mask, "image_path"].progress_apply(
            self._extract_shade_labstar
        )

        df.loc[valid_mask, "L"] = results.apply(lambda x: x[0])
        df.loc[valid_mask, "a"] = results.apply(lambda x: x[1])
        df.loc[valid_mask, "b"] = results.apply(lambda x: x[2])

        pct_nan = 100 * df["L"].isna().mean()
        logger.info("%.1f%% des lignes sans couleur extraite.", pct_nan)
        return df

    def _extract_shade_labstar(
        self, path: Optional[str]
    ) -> Tuple[float, float, float]:
        """
        Extraction robuste inspirée du notebook (cell 29 : extract_shade_labstar).
        Retourne (L*, a*, b*) ou (nan, nan, nan).
        """
        _nan = (np.nan, np.nan, np.nan)

        if not isinstance(path, str) or not os.path.exists(path):
            return _nan

        img = cv2.imread(path)
        if img is None:
            return _nan

        # Resize
        h, w = img.shape[:2]
        scale = self.resize / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img,
                             (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

        # Focus centre (évite packaging bords)
        if self.prefer_center:
            h, w = img.shape[:2]
            y1, y2 = int(h * 0.18), int(h * 0.82)
            x1, x2 = int(w * 0.18), int(w * 0.82)
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                img = roi

        # Masque blanc/noir
        keep_bg = _remove_white_black(img, self.white_thresh, self.black_thresh)

        # Masque peau HSV ∩ YCrCb
        skin = _skin_mask_hsv_ycrcb(img)

        # Combinaison
        mask = cv2.bitwise_and(keep_bg, skin)

        # Fallback si pas assez de pixels peau
        if cv2.countNonZero(mask) < self.min_skin_pixels:
            # Fallback : luminosité simple (L entre black_thresh et white_thresh)
            lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            L_channel = lab_img[:, :, 0]
            fallback_mask = (
                (L_channel > self.black_thresh) & (L_channel < self.white_thresh)
            ).astype(np.uint8) * 255
            if cv2.countNonZero(fallback_mask) < self.min_skin_pixels:
                return _nan
            mask = fallback_mask

        # Garder le plus grand composant
        mask = _largest_component(mask)

        # Pixels retenus
        pix_bgr = img[mask > 0]
        if pix_bgr.shape[0] < self.min_skin_pixels:
            return _nan

        # Conversion Lab OpenCV → Lab*
        pix_lab = cv2.cvtColor(
            pix_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2LAB
        ).reshape(-1, 3)
        pix_labstar = _lab_opencv_to_labstar(pix_lab)

        # Médiane (robuste aux reflets)
        med = np.median(pix_labstar, axis=0)
        return (float(med[0]), float(med[1]), float(med[2]))