import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..")
FEATURES_FILE = os.path.join(BASE_DIR, "data", "processed", "data_eye_features.parquet")
MODEL_FILE = os.path.join(BASE_DIR, "outputs", "models", "smart_selector_eye.pkl")
FINAL_FILE = os.path.join(BASE_DIR, "data", "processed", "data_eye_gold.parquet")
MULTICOLOR_FILE = os.path.join(BASE_DIR, "data", "processed", "data_eye_multicolor.parquet")

def main():
    print("⚙️ Inférence Eye - Production")
    print("=" * 70)
    
    # Vérification des fichiers requis
    if not os.path.exists(FEATURES_FILE):
        print(f"❌ Fichier features introuvable : {FEATURES_FILE}")
        print("👉 Lancez d'abord l'extraction de features (step1_eye_features_extractor.py)")
        return
    
    if not os.path.exists(MODEL_FILE):
        print(f"❌ Modèle introuvable : {MODEL_FILE}")
        print("👉 Lancez d'abord l'entraînement du modèle (step3_eye_train_model.py)")
        return

    # Chargement des données
    print(f"📂 Chargement des features...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"   ✅ {len(df)} produits chargés")
    
    print(f"\n🧠 Chargement du modèle...")
    clf = joblib.load(MODEL_FILE)
    print(f"   ✅ Modèle chargé : {MODEL_FILE}")
    
    # Filtrer uniquement les lignes avec features valides
    mask = df['features_spatial'].notna()
    df_valid = df[mask].copy()

    if 'is_multicolor' not in df_valid.columns:
        df_valid['is_multicolor'] = False
    df_valid['is_multicolor'] = df_valid['is_multicolor'].fillna(False).astype(bool)
    
    if len(df_valid) == 0:
        print("\n❌ Aucune feature valide trouvée")
        return
    
    print(f"\n📊 Produits valides : {len(df_valid)}/{len(df)} ({len(df_valid)/len(df)*100:.1f}%)")

    df_multi = df_valid[df_valid['is_multicolor']].copy()
    df_single = df_valid[~df_valid['is_multicolor']].copy()

    print(f"   - Single-color : {len(df_single)}")
    print(f"   - Multi-couleurs: {len(df_multi)}")
    
    # Distribution par catégorie
    if 'category_level_3_name' in df_valid.columns:
        print(f"\n📊 Distribution par sous-catégorie:")
        for cat, count in df_valid['category_level_3_name'].value_counts().items():
            print(f"   - {cat}: {count} produits")
    
    # 1. Prédiction de masse
    if len(df_single) > 0:
        print(f"\n🎯 Prédiction sur {len(df_single)} produits single-color...")
        X = np.array(df_single['features_spatial'].tolist())
        predicted_indices = clf.predict(X)
        print(f"   ✅ Prédictions terminées")
    else:
        predicted_indices = np.array([], dtype=int)
        print(f"\n⚠️ Aucun produit single-color à prédire")
    
    # Distribution des choix
    print(f"\n📊 Distribution des choix du modèle:")
    unique, counts = np.unique(predicted_indices, return_counts=True)
    for choice, count in zip(unique, counts):
        pct = count / len(predicted_indices) * 100 if len(predicted_indices) > 0 else 0
        print(f"   - Candidat {choice+1}: {count:5d} produits ({pct:5.1f}%)")
    
    # 2. Récupération des couleurs RGB correspondantes
    print(f"\n🎨 Extraction des couleurs finales...")
    final_colors = []
    centers_col = df_single['kmeans_centers'].values
    
    for i, pred_idx in enumerate(predicted_indices):
        candidates = centers_col[i]
        # Sécurité : si l'index prédit est hors limites, prendre le candidat 0
        idx = pred_idx if pred_idx < len(candidates) else 0
        final_colors.append(candidates[idx])
    
    df_single['smart_color'] = final_colors
    df_single['predicted_index'] = predicted_indices

    if len(df_multi) > 0:
        df_multi['smart_color'] = None
        df_multi['predicted_index'] = -1
    
    print(f"   ✅ Couleurs extraites")
    
    # 3. Statistiques des couleurs sélectionnées
    print(f"\n🔍 Analyse des couleurs sélectionnées:")
    if len(final_colors) > 0:
        smart_colors_array = np.array(final_colors)

        # Moyennes RGB
        mean_rgb = smart_colors_array.mean(axis=0)
        print(f"   - Moyenne RGB : ({mean_rgb[0]:.1f}, {mean_rgb[1]:.1f}, {mean_rgb[2]:.1f})")

        # Min/Max
        min_rgb = smart_colors_array.min(axis=0)
        max_rgb = smart_colors_array.max(axis=0)
        print(f"   - Min RGB     : ({min_rgb[0]:.0f}, {min_rgb[1]:.0f}, {min_rgb[2]:.0f})")
        print(f"   - Max RGB     : ({max_rgb[0]:.0f}, {max_rgb[1]:.0f}, {max_rgb[2]:.0f})")
    else:
        print("   - Aucun RGB calculé (pas de produits single-color)")
    
    # 4. Sauvegarde finale
    print(f"\n💾 Sauvegarde du dataset final...")
    os.makedirs(os.path.dirname(FINAL_FILE), exist_ok=True)
    df_single.to_parquet(FINAL_FILE)
    print(f"   ✅ Fichier sauvegardé : {FINAL_FILE}")
    df_multi.to_parquet(MULTICOLOR_FILE)
    print(f"   ✅ Fichier multi-couleurs : {MULTICOLOR_FILE}")
    
    # Résumé final
    print("\n" + "=" * 70)
    print("✅ INFÉRENCE TERMINÉE AVEC SUCCÈS")
    print("=" * 70)
    print(f"📊 Produits traités    : {len(df_single)} single-color")
    print(f"🎨 Multi-couleurs      : {len(df_multi)} (mis à part)")
    print(f"🎨 Couleurs extraites  : {len(final_colors)}")
    print(f"💾 Fichier final       : {FINAL_FILE}")
    print(f"\n📋 Colonnes disponibles dans le fichier final:")
    print(f"   - smart_color       : Couleur RGB sélectionnée par le modèle")
    print(f"   - predicted_index   : Index du candidat choisi (0-4)")
    print(f"   - kmeans_centers    : Liste des 5 candidats")
    print(f"   - features_spatial  : Vecteur de 37 features")
    print(f"\n👉 Vous pouvez maintenant utiliser ce fichier pour :")
    print(f"   - Clustering de couleurs")
    print(f"   - Moteur de recherche visuelle")
    print(f"   - Recommandations de produits")
    print("=" * 70)

if __name__ == "__main__":
    main()
