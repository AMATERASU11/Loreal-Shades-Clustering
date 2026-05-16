"""
Précomputation Cache NailShadeDetector
═══════════════════════════════════════════════════════════════════
Génère outputs/cache/<stem>.npz pour chaque image labellisée.

Différence avec M3/M4 :
  - PAS de center-crop (pixels full-image après rembg)
  - Sauvegarde 3 arrays : lab (float32) + rgb (uint8) + y_norm (float32)
  - y_norm = row_index / image_height  (0.0 = haut, 1.0 = bas)

Le center-crop est appliqué à la volée dans Tier 3, ce qui permet
de calculer les features de position verticale (vpos) correctement.

Usage :
  python -m src.utils.precompute_cache              # tous les manquants
  python -m src.utils.precompute_cache --force      # tout recalculer
  python -m src.utils.precompute_cache --check      # vérifier couverture
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color as skcolor
from tqdm import tqdm

from src.utils.config import (
    LABELED_PARQUET, CACHE_DIR, IMAGE_DIR, RECOVERED_DIR,
    ALPHA_THRESHOLD, N_CLUSTERS,
)

try:
    from rembg import remove as rembg_remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("ERREUR : rembg non installé. Lancer : pip install rembg")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  UTILITAIRES
# ═══════════════════════════════════════════════════════════════

def find_image(filename: str) -> Path | None:
    """Cherche une image dans les dossiers connus."""
    for folder in [IMAGE_DIR, RECOVERED_DIR]:
        p = folder / str(filename)
        if p.exists():
            return p
    return None


def process_image(img_path: Path) -> dict | None:
    """
    Rembg → pixels valides (SANS crop) → lab + rgb + y_norm.

    Returns dict avec clés 'lab', 'rgb', 'y_norm', ou None si échec.
    """
    try:
        img_pil = Image.open(img_path).convert("RGB")
        img_rgba = rembg_remove(img_pil)
        img_np   = np.array(img_rgba)               # (H, W, 4)

        rgb_full   = img_np[:, :, :3]               # (H, W, 3)
        alpha_full = img_np[:, :, 3]                # (H, W)
        H, W       = alpha_full.shape

        # Masque des pixels valides (foreground)
        valid_mask = alpha_full > ALPHA_THRESHOLD   # (H, W) bool

        if valid_mask.sum() < N_CLUSTERS * 10:
            return None

        # Coordonnées Y de chaque pixel valide
        row_indices = np.arange(H).reshape(-1, 1).repeat(W, axis=1)  # (H, W)
        y_norm_full = row_indices / H                                  # (H, W) float

        # Extraction des pixels valides
        rgb_pixels   = rgb_full[valid_mask]          # (N, 3) uint8
        y_norm_pixels = y_norm_full[valid_mask]      # (N,)  float64

        # Conversion RGB → Lab
        rgb_norm = rgb_pixels.astype(np.float64) / 255.0
        lab_pixels = skcolor.rgb2lab(
            rgb_norm.reshape(-1, 1, 3)
        ).reshape(-1, 3)                             # (N, 3) float64

        return {
            "lab":    lab_pixels.astype(np.float32),
            "rgb":    rgb_pixels.astype(np.uint8),
            "y_norm": y_norm_pixels.astype(np.float32),
        }

    except Exception as e:
        return None


# ═══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def run(force: bool = False, check_only: bool = False):
    print("\n╔" + "═" * 64 + "╗")
    print("║" + "  PRÉCOMPUTATION CACHE NailShadeDetector".center(64) + "║")
    print("╚" + "═" * 64 + "╝\n")

    # Charger le dataset
    df = pd.read_parquet(LABELED_PARQUET)
    filenames = df["image_filename"].tolist()
    total = len(filenames)
    print(f"Images à traiter : {total}")
    print(f"Cache destination : {CACHE_DIR}\n")

    # Mode vérification uniquement
    if check_only:
        n_cached = 0
        n_missing = []
        for fn in filenames:
            stem = Path(str(fn)).stem
            if (CACHE_DIR / f"{stem}.npz").exists():
                n_cached += 1
            else:
                n_missing.append(fn)
        print(f"Couverture : {n_cached} / {total} ({n_cached/total*100:.1f}%)")
        if n_missing:
            print(f"Manquants ({len(n_missing)}) :")
            for fn in n_missing[:20]:
                print(f"  {fn}")
            if len(n_missing) > 20:
                print(f"  ... et {len(n_missing)-20} autres")
        return

    # Traitement
    t_start = time.time()
    n_ok = n_skip = n_fail = n_no_image = 0

    for fn in tqdm(filenames, desc="Precompute NailShadeDetector", unit="img"):
        stem     = Path(str(fn)).stem
        out_path = CACHE_DIR / f"{stem}.npz"

        # Skip si déjà calculé
        if not force and out_path.exists():
            n_skip += 1
            continue

        img_path = find_image(str(fn))
        if img_path is None:
            n_no_image += 1
            continue

        result = process_image(img_path)
        if result is None:
            n_fail += 1
            continue

        np.savez_compressed(
            out_path,
            lab    = result["lab"],
            rgb    = result["rgb"],
            y_norm = result["y_norm"],
        )
        n_ok += 1

    elapsed = time.time() - t_start
    print(f"\n{'─'*66}")
    print(f"  Terminé en {elapsed/60:.1f} min")
    print(f"  Générés     : {n_ok}")
    print(f"  Déjà cachés : {n_skip}")
    print(f"  Échecs rembg: {n_fail}")
    print(f"  Image absente: {n_no_image}")
    total_cached = len(list(CACHE_DIR.glob("*.npz")))
    print(f"  Cache total  : {total_cached} fichiers")
    print(f"{'─'*66}\n")


# ═══════════════════════════════════════════════════════════════
#  MODE INFÉRENCE (pas de labels requis)
# ═══════════════════════════════════════════════════════════════

def run_inference(input_csv: str, force: bool = False):
    """
    Précompute le cache pour les images d'un CSV d'inférence.
    Ne nécessite PAS le parquet labellisé.
    """
    print("\n╔" + "═" * 64 + "╗")
    print("║" + "  PRÉCOMPUTATION CACHE (INFÉRENCE)".center(64) + "║")
    print("╚" + "═" * 64 + "╝\n")

    df = pd.read_csv(input_csv)
    if "image_filename" not in df.columns:
        print("ERREUR : colonne 'image_filename' requise")
        return

    filenames = df["image_filename"].tolist()
    total = len(filenames)
    print(f"Images à traiter : {total}")
    print(f"Cache destination : {CACHE_DIR}\n")

    t_start = time.time()
    n_ok = n_skip = n_fail = n_no_image = 0

    for fn in tqdm(filenames, desc="Precompute (inférence)", unit="img"):
        stem = Path(str(fn)).stem
        out_path = CACHE_DIR / f"{stem}.npz"

        if not force and out_path.exists():
            n_skip += 1
            continue

        img_path = find_image(str(fn))
        if img_path is None:
            n_no_image += 1
            continue

        result = process_image(img_path)
        if result is None:
            n_fail += 1
            continue

        np.savez_compressed(
            out_path,
            lab=result["lab"],
            rgb=result["rgb"],
            y_norm=result["y_norm"],
        )
        n_ok += 1

    elapsed = time.time() - t_start
    print(f"\n{'─'*66}")
    print(f"  Terminé en {elapsed/60:.1f} min")
    print(f"  Générés      : {n_ok}")
    print(f"  Déjà cachés  : {n_skip}")
    print(f"  Échecs rembg : {n_fail}")
    print(f"  Image absente: {n_no_image}")
    total_cached = len(list(CACHE_DIR.glob("*.npz")))
    print(f"  Cache total  : {total_cached} fichiers")
    print(f"{'─'*66}\n")


# ═══════════════════════════════════════════════════════════════
#  ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Précomputation cache NailShadeDetector")
    parser.add_argument("--force", action="store_true",
                        help="Recalculer même si le cache existe")
    parser.add_argument("--check", action="store_true",
                        help="Vérifier la couverture sans calculer")
    args = parser.parse_args()
    run(force=args.force, check_only=args.check)
