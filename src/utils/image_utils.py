"""
utils/image_utils.py
--------------------
Fonctions pures de chargement, validation et prétraitement d'images.
"""
import os
import warnings
import numpy as np
from PIL import Image
import cv2
from typing import Optional


def build_image_path(filename: str, images_dir: str) -> Optional[str]:
    """
    Construit le chemin complet d'une image en testant plusieurs extensions.
    Retourne None si le fichier n'existe pas.
    """
    if not filename or (isinstance(filename, float) and np.isnan(filename)):
        return None

    f = str(filename).strip()
    p = os.path.join(images_dir, f)
    if os.path.exists(p):
        return p

    if "." not in f:
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            p2 = os.path.join(images_dir, f + ext)
            if os.path.exists(p2):
                return p2

    return None


def can_open_image(path: Optional[str]) -> bool:
    """
    Vérifie qu'une image est lisible sans la charger entièrement en RAM.
    """
    if not path or not os.path.exists(path):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                im.verify()
        return True
    except Exception:
        return False


def load_image_bgr(path: str, resize: int = 600) -> Optional[np.ndarray]:
    """
    Charge une image en BGR avec OpenCV, redimensionnée si nécessaire.
    Retourne None si échec.
    """
    img = cv2.imread(path)
    if img is None:
        return None

    h0, w0 = img.shape[:2]
    scale = resize / max(h0, w0)
    if scale < 1:
        img = cv2.resize(
            img,
            (int(w0 * scale), int(h0 * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return img


def build_non_white_black_mask(
    img_rgb: np.ndarray,
    white_thr: int = 245,
    black_thr: int = 10,
    margin_ratio: float = 0.05,
) -> np.ndarray:
    """
    Masque booléen : True sur les pixels ni trop blancs ni trop noirs,
    avec marge sur les bords.
    """
    not_white = (
        (img_rgb[:, :, 0] < white_thr)
        | (img_rgb[:, :, 1] < white_thr)
        | (img_rgb[:, :, 2] < white_thr)
    )
    not_black = (
        (img_rgb[:, :, 0] > black_thr)
        | (img_rgb[:, :, 1] > black_thr)
        | (img_rgb[:, :, 2] > black_thr)
    )
    mask = not_white & not_black

    h, w = img_rgb.shape[:2]
    m = int(margin_ratio * min(h, w))
    if m > 0:
        mask[:m, :] = False
        mask[-m:, :] = False
        mask[:, :m] = False
        mask[:, -m:] = False

    return mask


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    Convertit un tableau de pixels RGB (N, 3) en Lab (N, 3) via OpenCV.
    """
    bgr = rgb[:, ::-1].reshape(-1, 1, 3).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab
