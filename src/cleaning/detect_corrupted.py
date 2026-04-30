"""
Détection d'images corrompues
═══════════════════════════════════════════════════════════════════
Parcourt un dossier d'images et identifie celles qui sont invalides
ou illisibles avec Pillow (PIL).

Trois types de corruption détectés :
  1. Fichier illisible         — Image.open() échoue
  2. Structure invalide        — img.verify() échoue
  3. Image tronquée            — img.load() échoue (données incomplètes)

Sorties dans outputs/reports/ :
  corrupted_images_<timestamp>.csv   — liste des images corrompues
  corruption_summary_<timestamp>.json — résumé chiffré

Usage :
  python -m src.cleaning.detect_corrupted
  python -m src.cleaning.detect_corrupted --images-dir data/raw/images
  python -m src.cleaning.detect_corrupted --images-dir data/raw/images --workers 8
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

# ── Chemins par défaut ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

SUPPORTED_EXTENSIONS = {".jpeg", ".jpg", ".png", ".webp", ".gif", ".bmp"}


# ═══════════════════════════════════════════════════════════════
#  VÉRIFICATION D'UNE IMAGE
# ═══════════════════════════════════════════════════════════════

def check_image(path: Path) -> dict:
    """
    Vérifie qu'une image est valide et lisible.

    Retourne un dict avec :
      - filename  : nom du fichier
      - status    : "valid" ou "corrupted"
      - error     : message d'erreur si corrompue, "" sinon
    """
    result = {"filename": path.name, "status": "valid", "error": ""}

    # Étape 1 : ouverture
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, OSError) as e:
        result["status"] = "corrupted"
        result["error"] = f"open_failed: {e}"
        return result

    # Étape 2 : vérification structure (header + chunks)
    try:
        img.verify()
    except Exception as e:
        result["status"] = "corrupted"
        result["error"] = f"verify_failed: {e}"
        return result

    # Étape 3 : décompression complète (détecte les images tronquées)
    try:
        img = Image.open(path)   # réouverture nécessaire après verify()
        img.load()
    except Exception as e:
        result["status"] = "corrupted"
        result["error"] = f"load_failed: {e}"
        return result

    return result


# ═══════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def run(images_dir: Path, workers: int = 16) -> dict:
    """
    Détecte toutes les images corrompues dans images_dir.

    Args:
        images_dir : dossier contenant les images
        workers    : nombre de threads parallèles

    Returns:
        dict avec les statistiques et chemins des fichiers générés
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable : {images_dir}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Lister les images
    all_files = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    total = len(all_files)

    if total == 0:
        print(f"Aucune image trouvée dans : {images_dir}")
        return {}

    print(f"\nImages à vérifier : {total:,}")
    print(f"Threads           : {workers}")
    print(f"Dossier           : {images_dir}\n")

    # Vérification parallèle
    results = []
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_image, p): p for p in all_files}
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 1000 == 0 or done == total:
                pct = done / total * 100
                print(f"  {done:>6,} / {total:,}  ({pct:.1f}%)", end="\r")

    print()

    # Séparer valides / corrompues
    corrupted = [r for r in results if r["status"] == "corrupted"]
    valid     = [r for r in results if r["status"] == "valid"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Sauvegarde CSV des corrompues ──────────────────────────
    csv_path = REPORTS_DIR / f"corrupted_images_{timestamp}.csv"
    df_corrupted = pd.DataFrame(corrupted)
    df_corrupted["detected_date"] = datetime.now().isoformat()
    df_corrupted.to_csv(csv_path, index=False)

    # ── Sauvegarde résumé JSON ─────────────────────────────────
    summary = {
        "timestamp":              timestamp,
        "images_dir":             str(images_dir),
        "total_images":           total,
        "valid_images":           len(valid),
        "corrupted_images":       len(corrupted),
        "corruption_rate_pct":    round(len(corrupted) / total * 100, 2),
        "csv_report":             str(csv_path),
    }
    json_path = REPORTS_DIR / f"corruption_summary_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Affichage résumé ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Total images    : {total:,}")
    print(f"  Valides         : {len(valid):,}")
    print(f"  Corrompues      : {len(corrupted):,}  ({summary['corruption_rate_pct']}%)")
    print(f"{'─'*60}")
    print(f"  Rapport CSV  → {csv_path.name}")
    print(f"  Résumé JSON  → {json_path.name}")
    print(f"{'─'*60}\n")

    return summary


# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Détecte les images corrompues dans un dossier"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help=f"Dossier des images (défaut : {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Nombre de threads parallèles (défaut : 16)",
    )
    args = parser.parse_args()

    run(images_dir=args.images_dir, workers=args.workers)
