"""
utils/color_utils.py
--------------------
Fonctions pures de calcul colorimétrique.
"""
import numpy as np
import cv2
from typing import Tuple


def delta_e_cie76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """
    Distance colorimétrique CIE76 entre deux couleurs Lab.
    0 = identique, >10 = très différent.
    """
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def lab_to_rgb_display(lab_color: np.ndarray) -> Tuple[int, int, int]:
    """
    Convertit une couleur Lab OpenCV en RGB uint8 pour affichage.
    """
    pixel_lab = np.uint8([[lab_color]])
    pixel_bgr = cv2.cvtColor(pixel_lab, cv2.COLOR_LAB2BGR)[0][0]
    return (int(pixel_bgr[2]), int(pixel_bgr[1]), int(pixel_bgr[0]))


def is_neutral_color(lab_color: np.ndarray, chroma_threshold: float = 10.0) -> bool:
    """
    Détermine si une couleur est neutre (gris, blanc, noir).
    Basé sur la chrominance dans l'espace OpenCV Lab (a, b centrés sur 128).
    """
    a_centered = float(lab_color[1]) - 128.0
    b_centered = float(lab_color[2]) - 128.0
    chroma = np.sqrt(a_centered ** 2 + b_centered ** 2)
    return chroma < chroma_threshold


def compute_lab_centroid(lab_pixels: np.ndarray) -> np.ndarray:
    """
    Centroïde robuste (médiane) d'un nuage de pixels Lab.
    """
    return np.median(lab_pixels, axis=0)


def normalize_lab_for_clustering(lab_pixels: np.ndarray) -> np.ndarray:
    """
    Pondère les coordonnées Lab pour donner plus de poids
    à la chrominance (a, b) qu'à la luminance (L).
    L est multiplié par 0.5.
    """
    weighted = lab_pixels.copy().astype(np.float32)
    weighted[:, 0] *= 0.5
    return weighted
