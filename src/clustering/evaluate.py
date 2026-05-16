"""
Évaluation Complète — ShadeNail
═══════════════════════════════════════════════════════════════════
Benchmark sur les 1261 labels avec rapport détaillé par tag.

Supporte deux modes :
  - ShadeNail Ranking (par défaut) — modèles shadenail_*.pkl
  - XGBoost Classifier (legacy)   — modèle best_model.pkl

Métriques :
  - Accuracy de sélection cluster (0..4)
  - ΔE moyen / médian / distribution
  - @ΔE≤5/10/15/20/25 — sur tous les produits ET sur les prédits uniquement
  - Taux de quarantaine par catégorie

Usage :
  python -m src.clustering.evaluate
  python -m src.clustering.evaluate --legacy   # XGBoost classifier
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color as skcolor
from tqdm import tqdm

from src.utils.config import (
    LABELED_PARQUET, RESULT_DIR, MODEL_DIR,
    CONFIDENCE_THRESHOLD, DELTA_E_THRESHOLDS,
    TAG_OK, TAG_KIT, TAG_INCOLORE, TAG_TEXTURE, TAG_ANOMALIE,
)
import src.feature_engineering.tier0_gatekeeper as tier0
import src.feature_engineering.tier1_texture    as tier1


def _rgb_to_lab(rgb):
    arr = np.array(rgb, dtype=np.uint8).reshape(1, 1, 3)
    return skcolor.rgb2lab(arr / 255.0)[0, 0]


def _delta_e(lab1, lab2):
    return float(np.sqrt(np.sum((np.array(lab1) - np.array(lab2)) ** 2)))


def _parse_label(label):
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None
    try:
        parsed = ast.literal_eval(label) if isinstance(label, str) else list(label)
        return [int(x) for x in parsed[:3]]
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  BENCHMARK COMPLET
# ═══════════════════════════════════════════════════════════════

def run_full_benchmark(legacy: bool = False):
    """
    Lance le benchmark sur les 1261 labels complets.
    """
    model_name = "XGBoost Classifier" if legacy else "ShadeNail Ranking"

    print("\n╔" + "═" * 64 + "╗")
    print("║" + f"  BENCHMARK {model_name} — 1261 LABELS".center(64) + "║")
    print("╚" + "═" * 64 + "╝\n")

    # ── Charger modèle ────────────────────────────────────────
    if legacy:
        from src.clustering.tier4_xgboost import load_model, predict
        model = load_model()
        if model is None:
            print("ERREUR : modèle XGBoost non trouvé (best_model.pkl)")
            sys.exit(1)
        predict_fn = lambda row: predict(row, model)
    else:
        from src.clustering.tier4_ranking import load_model, predict
        model = load_model()
        if model is None:
            print("ERREUR : mod\u00e8le ShadeNail non trouv\u00e9 (shadenail_xgb.pkl)")
            sys.exit(1)
        predict_fn = lambda row: predict(row, model)

    df = pd.read_parquet(LABELED_PARQUET)
    print(f"Dataset : {len(df)} produits\n")

    # ── Prédictions ───────────────────────────────────────────
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Prédictions"):
        manual_rgb = _parse_label(row.get("manual_label"))

        result = predict_fn(row)
        tag = result["tag"]

        record = {
            "image":      row["image_filename"],
            "shade":      str(row.get("shade_name", "")),
            "brand":      str(row.get("brand_name", "")),
            "tag":        tag,
            "confidence": result["confidence"],
            "de":         None,
            "predicted":  tag == TAG_OK,
        }

        if tag == TAG_OK and manual_rgb is not None and result["shade_rgb"] is not None:
            pred_lab = _rgb_to_lab(result["shade_rgb"])
            man_lab = _rgb_to_lab(manual_rgb)
            record["de"] = round(_delta_e(pred_lab, man_lab), 2)

        records.append(record)

    # ── Agrégation ────────────────────────────────────────────
    df_res = pd.DataFrame(records)
    _print_report(df_res)

    out_path = RESULT_DIR / f"benchmark_{model_name.replace(' ', '_')}.json"
    df_res.to_json(out_path, orient="records", indent=2)
    print(f"\nRésultats sauvegardés → {out_path}")


# ═══════════════════════════════════════════════════════════════
#  RAPPORT
# ═══════════════════════════════════════════════════════════════

def _print_report(df_res: pd.DataFrame):
    total = len(df_res)

    # ── Comptages par tag ─────────────────────────────────────
    tag_counts = df_res["tag"].value_counts()
    print("─" * 68)
    print("  DISTRIBUTION DES TAGS")
    print("─" * 68)
    for tag, n in tag_counts.items():
        bar = "█" * int(n / total * 40)
        print(f"  {tag:<40} {n:4d}  ({n/total*100:5.1f}%)  {bar}")

    # ── Stats ΔE sur produits prédits (TAG_OK) ────────────────
    df_ok = df_res[df_res["tag"] == TAG_OK].dropna(subset=["de"])
    n_ok  = len(df_ok)

    print(f"\n{'─'*68}")
    print(f"  MÉTRIQUES ΔE — PRODUITS PRÉDITS ({n_ok} produits, conf >= {CONFIDENCE_THRESHOLD})")
    print(f"{'─'*68}")

    if n_ok > 0:
        de_arr = df_ok["de"].values
        print(f"  ΔE moyen   : {de_arr.mean():.2f}")
        print(f"  ΔE médian  : {np.median(de_arr):.2f}")
        print(f"  ΔE max     : {de_arr.max():.1f}")
        print(f"  ΔE std     : {de_arr.std():.2f}")
        print()
        for thr in DELTA_E_THRESHOLDS:
            acc = (de_arr <= thr).mean() * 100
            bar = "█" * int(acc / 2)
            print(f"  @ΔE≤{thr:2d}  : {acc:5.1f}%  {bar}")

    # ── Comparaison sur tous (incluant quarantinés = de=None) ─
    # Baseline M3 incluait les kits et glitters → on compare en mode "all"
    print(f"\n{'─'*68}")
    print(f"  COMPARAISON BASELINE M3 (mode all-inclusive)")
    print(f"{'─'*68}")

    # Pour les quarantinés, on attribue ΔE = +∞ (erreur totale)
    de_all = []
    for _, row in df_res.iterrows():
        if row["de"] is not None:
            de_all.append(row["de"])
        else:
            de_all.append(200.0)  # erreur max

    de_all = np.array(de_all)
    print(f"  Mode all (quarantinés = ΔE 200) :")
    for thr in DELTA_E_THRESHOLDS:
        acc_nsd = (de_all <= thr).mean() * 100
        diff = acc_nsd - {5: 0, 10: 73.2, 15: 81.0, 20: 85.0, 25: 87.0}.get(thr, 0)
        sign = "+" if diff >= 0 else ""
        print(f"  @ΔE≤{thr:2d}  NSD={acc_nsd:5.1f}%  "
              f"vs M3={'N/A':>5}  delta={sign}{diff:.1f}pts"
              if thr != 10 else
              f"  @ΔE≤{thr:2d}  NSD={acc_nsd:5.1f}%  "
              f"vs M3=73.2%  delta={sign}{diff:.1f}pts")

    # ── Stats ΔE par brand (top 10 marques) ──────────────────
    print(f"\n{'─'*68}")
    print(f"  PERFORMANCE PAR MARQUE (top 10 en volume)")
    print(f"{'─'*68}")

    df_ok_brand = df_ok.copy()
    if "brand" in df_ok_brand.columns and len(df_ok_brand) > 0:
        top_brands = (
            df_ok_brand["brand"]
            .value_counts()
            .head(10)
            .index
            .tolist()
        )
        print(f"  {'Marque':<30} {'N':>4}  {'@ΔE≤10':>8}  {'ΔE moy':>8}")
        print(f"  {'─'*30} {'─'*4}  {'─'*8}  {'─'*8}")
        for brand in top_brands:
            df_b = df_ok_brand[df_ok_brand["brand"] == brand]
            de_b = df_b["de"].values
            acc10 = (de_b <= 10).mean() * 100
            print(f"  {brand[:29]:<30} {len(de_b):>4}  {acc10:>7.1f}%  {de_b.mean():>7.2f}")

    # ── Résumé final ──────────────────────────────────────────
    n_quarantine = (df_res["tag"] != TAG_OK).sum()
    acc10_predicted = float((df_ok["de"] <= 10).mean() * 100) if n_ok > 0 else 0

    print(f"\n{'╔' + '═'*64 + '╗'}")
    print(f"{'║' + '  RÉSUMÉ FINAL NailShadeDetector'.center(64) + '║'}")
    print(f"{'╠' + '═'*64 + '╣'}")
    lines = [
        f"Produits totaux         : {total}",
        f"Quarantinés (Tier 0+1+conf) : {n_quarantine} ({n_quarantine/total*100:.1f}%)",
        f"Prédits                 : {n_ok} ({n_ok/total*100:.1f}%)",
        f"@ΔE≤10 sur prédits      : {acc10_predicted:.1f}%",
        f"Baseline M3 (all)       : 73.2% @ΔE≤10",
        f"Target                  : 85.0% @ΔE≤10",
    ]
    for line in lines:
        print(f"{'║'}  {line:<62}{'║'}")
    print(f"{'╚' + '═'*64 + '╝'}\n")


# ═══════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ShadeNail")
    parser.add_argument("--legacy", action="store_true",
                        help="Utiliser XGBoost Classifier au lieu de ShadeNail Ranking")
    args = parser.parse_args()
    run_full_benchmark(legacy=args.legacy)
