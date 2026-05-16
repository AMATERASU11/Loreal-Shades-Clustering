"""
Pipeline Principal — ShadeNail
═══════════════════════════════════════════════════════════════════
Orchestrateur bout-en-bout.

Modes :
  --precompute   Lance precompute_cache.py (rembg → cache)
  --train        Entraîne le XGBoost Classifier (legacy Tier 4)
  --evaluate     Benchmark complet sur les 1261 labels
  --predict      Prédit pour un product_id donné (--product_id XXX)
  --infer        Mode inférence : CSV + images → predictions.csv (ShadeNail Ranking)
  --full         Enchaîne precompute + train + evaluate

Usage :
  python -m src.clustering.pipeline_main --full
  python -m src.clustering.pipeline_main --infer --input data/raw/products.csv
  python -m src.clustering.pipeline_main --predict --product_id <sha256>
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from src.utils.config import LABELED_PARQUET, CACHE_DIR, PROJECT_ROOT


def step_precompute(force: bool = False):
    """Étape 1 : Générer le cache rembg (lab + rgb + y_norm)."""
    print("\n" + "━" * 68)
    print("  ÉTAPE 1 — PRÉCOMPUTATION CACHE")
    print("━" * 68)
    from src.utils.precompute_cache import run
    run(force=force)


def step_precompute_infer(input_csv: str, force: bool = False):
    """Précompute pour inférence : scan images du CSV d'entrée."""
    print("\n" + "━" * 68)
    print("  ÉTAPE 1 — PRÉCOMPUTATION CACHE (INFÉRENCE)")
    print("━" * 68)
    from src.utils.precompute_cache import run_inference
    run_inference(input_csv, force=force)


def step_train():
    """Étape 2 : Entraîner XGBoost Classifier (legacy)."""
    print("\n" + "━" * 68)
    print("  ÉTAPE 2 — ENTRAÎNEMENT XGBOOST (LEGACY)")
    print("━" * 68)
    from src.clustering.tier4_xgboost import train
    results = train(save=True)
    return results


def step_evaluate():
    """Étape 3 : Benchmark complet sur les 1261 labels."""
    print("\n" + "━" * 68)
    print("  ÉTAPE 3 — ÉVALUATION COMPLÈTE")
    print("━" * 68)
    from src.clustering.evaluate import run_full_benchmark
    run_full_benchmark()


def step_predict(product_id: str):
    """Prédit la couleur d'un produit par son product_id (ShadeNail ranking)."""
    from src.clustering.tier4_ranking import load_model, predict

    df = pd.read_parquet(LABELED_PARQUET)
    row_df = df[df["product_id"] == product_id]

    if len(row_df) == 0:
        print(f"Produit non trouv\u00e9 : {product_id}")
        sys.exit(1)

    row = row_df.iloc[0]
    model = load_model()

    if model is None:
        print("Mod\u00e8le ShadeNail non trouv\u00e9. V\u00e9rifier outputs/models/shadenail_xgb.pkl")
        sys.exit(1)

    result = predict(row, model)

    print(f"\n{'─'*60}")
    print(f"  Produit  : {row.get('product_id', '')[:20]}...")
    print(f"  Shade    : {row.get('shade_name', '')}")
    print(f"  Brand    : {row.get('brand_name', '')}")
    print(f"{'─'*60}")
    print(f"  Tag      : {result['tag']}")

    if result["shade_rgb"]:
        R, G, B = result["shade_rgb"]
        print(f"  RGB      : [{R}, {G}, {B}]")
        L, a, b = result["shade_lab"]
        print(f"  Lab      : [L={L:.1f}, a={a:.1f}, b={b:.1f}]")

    print(f"  Confidence : {result['confidence']:.3f}")
    if result["color_word"]:
        print(f"  NLP word : {result['color_word']}")
    print(f"{'─'*60}\n")


def step_infer(input_csv: str, output_csv: str = None, force_cache: bool = False):
    """
    Mode inférence complet : CSV + images → predictions.csv

    Le CSV d'entrée doit contenir au minimum :
      - image_filename (nom du fichier image dans data/raw/images/)
    Colonnes optionnelles (améliorent la prédiction) :
      - shade_name, brand_name, product_id
    """
    from tqdm import tqdm
    from src.clustering.tier4_ranking import load_model, predict

    print("\n" + "━" * 68)
    print("  SHADENAIL — INFÉRENCE")
    print("━" * 68)

    # ── Charger modèle ───────────────────────────────────────
    model = load_model()
    if model is None:
        print("ERREUR : Modèle ShadeNail non trouvé.")
        print("  Vérifier : outputs/models/shadenail_xgb.pkl")
        sys.exit(1)

    # ── Charger CSV d'entrée ─────────────────────────────────
    input_path = Path(input_csv)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    if not input_path.exists():
        print(f"ERREUR : fichier introuvable → {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"\nCSV chargé : {len(df)} lignes depuis {input_path.name}")

    if "image_filename" not in df.columns:
        print("ERREUR : colonne 'image_filename' requise dans le CSV")
        sys.exit(1)

    # ── Précompute cache si nécessaire ───────────────────────
    n_missing = sum(
        1 for fn in df["image_filename"]
        if not (CACHE_DIR / f"{Path(str(fn)).stem}.npz").exists()
    )
    if n_missing > 0 or force_cache:
        print(f"\n{n_missing} images sans cache → lancement précomputation...")
        step_precompute_infer(str(input_path), force=force_cache)

    # ── Prédictions ──────────────────────────────────────────
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ShadeNail predict"):
        result = predict(row, model)
        rgb = result["shade_rgb"]
        lab = result["shade_lab"]
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb) if rgb else None
        results.append({
            "image_filename": row["image_filename"],
            "product_id":     row.get("product_id", ""),
            "brand_name":     row.get("brand_name", ""),
            "shade_name":     row.get("shade_name", ""),
            "tag":            result["tag"],
            "shade_lab_L":    round(lab[0], 2) if lab else None,
            "shade_lab_a":    round(lab[1], 2) if lab else None,
            "shade_lab_b":    round(lab[2], 2) if lab else None,
            "shade_rgb_R":    rgb[0] if rgb else None,
            "shade_rgb_G":    rgb[1] if rgb else None,
            "shade_rgb_B":    rgb[2] if rgb else None,
            "shade_hex":      hex_color,
            "confidence":     round(result["confidence"], 3),
            "color_word":     result.get("color_word", ""),
        })

    # ── Sauvegarder CSV ──────────────────────────────────────
    if output_csv is None:
        output_csv = str(PROJECT_ROOT / "outputs" / "predictions" / "predictions.csv")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_path, index=False)

    # ── Résumé ───────────────────────────────────────────────
    n_ok = sum(1 for r in results if r["tag"] == "OK")
    n_total = len(results)
    print(f"\n{'━'*68}")
    print(f"  RÉSULTATS INFÉRENCE")
    print(f"{'━'*68}")
    print(f"  Produits traités   : {n_total}")
    print(f"  Couleur détectée   : {n_ok} ({n_ok/n_total*100:.1f}%)")
    print(f"  Quarantinés        : {n_total - n_ok}")
    print(f"  Fichier de sortie  : {out_path}")
    print(f"{'━'*68}\n")


# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ShadeNail — Extraction couleur vernis à ongles"
    )
    parser.add_argument("--precompute", action="store_true",
                        help="Générer le cache rembg (lab + rgb + y_norm)")
    parser.add_argument("--force",      action="store_true",
                        help="Forcer le recalcul du cache")
    parser.add_argument("--train",      action="store_true",
                        help="Entraîner XGBoost (legacy)")
    parser.add_argument("--evaluate",   action="store_true",
                        help="Benchmark complet")
    parser.add_argument("--predict",    action="store_true",
                        help="Prédire pour un produit (par product_id)")
    parser.add_argument("--product_id", type=str, default=None,
                        help="product_id pour --predict")
    parser.add_argument("--infer",      action="store_true",
                        help="Inférence ShadeNail : CSV + images → predictions.csv")
    parser.add_argument("--input",      type=str, default=None,
                        help="CSV d'entrée pour --infer")
    parser.add_argument("--output",     type=str, default=None,
                        help="CSV de sortie pour --infer")
    parser.add_argument("--full",       action="store_true",
                        help="Enchaîner precompute + train + evaluate")
    args = parser.parse_args()

    t0 = time.time()

    print("\n╔" + "═" * 66 + "╗")
    print("║" + "  PIPELINE ShadeNail — EXTRACTION COULEUR VERNIS".center(66) + "║")
    print("╚" + "═" * 66 + "╝")

    if args.infer:
        if not args.input:
            print("ERREUR : --infer nécessite --input <csv>")
            sys.exit(1)
        step_infer(args.input, output_csv=args.output, force_cache=args.force)

    elif args.full:
        if not LABELED_PARQUET.exists():
            print(f"ERREUR : fichier introuvable → {LABELED_PARQUET}")
            sys.exit(1)
        step_precompute(force=args.force)
        step_train()
        step_evaluate()

    elif args.precompute:
        n_cached = len(list(CACHE_DIR.glob("*.npz")))
        print(f"Cache existant : {n_cached} fichiers")
        step_precompute(force=args.force)

    elif args.train:
        n_cached = len(list(CACHE_DIR.glob("*.npz")))
        if n_cached == 0:
            print("ERREUR : cache vide. Lancer d'abord : --precompute")
            sys.exit(1)
        print(f"Cache disponible : {n_cached} fichiers")
        step_train()

    elif args.evaluate:
        step_evaluate()

    elif args.predict:
        if not args.product_id:
            print("ERREUR : --predict nécessite --product_id <sha256>")
            sys.exit(1)
        step_predict(args.product_id)

    else:
        parser.print_help()
        sys.exit(0)

    elapsed = time.time() - t0
    print(f"\nTemps total : {elapsed/60:.1f} min\n")


if __name__ == "__main__":
    main()
