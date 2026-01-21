import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIG ---
FEATURES_FILE = "data_lip_features.parquet"  # Le store complet
MODEL_FILE = "smart_selector_v3.pkl"         # Le cerveau
FINAL_FILE = "data_lip_gold.parquet"         # Le résultat final

def main():
    print("⚙️ Lancement de l'Inférence (Production)...")
    
    if not os.path.exists(FEATURES_FILE) or not os.path.exists(MODEL_FILE):
        print("❌ Fichiers manquants.")
        return

    df = pd.read_parquet(FEATURES_FILE)
    clf = joblib.load(MODEL_FILE)
    
    # On ne garde que les lignes valides (où l'extraction a fonctionné)
    mask = df['features_spatial'].notna()
    df_valid = df[mask].copy()
    
    print(f"🧠 Prédiction sur {len(df_valid)} produits...")
    
    # 1. Prédiction de masse (Ultra rapide)
    X = np.array(df_valid['features_spatial'].tolist())
    predicted_indices = clf.predict(X) # [0, 2, 0, 1, ...]
    
    # 2. Récupération de la couleur RGB correspondante
    # On utilise l'index prédit pour piocher dans la liste 'kmeans_centers'
    final_colors = []
    centers_col = df_valid['kmeans_centers'].values
    
    for i, pred_idx in enumerate(predicted_indices):
        candidates = centers_col[i]
        # Sécurité : si l'index prédit est hors limites (rare), on prend le 0
        idx = pred_idx if pred_idx < len(candidates) else 0
        final_colors.append(candidates[idx])
        
    df_valid['smart_color'] = final_colors
    
    # 3. Sauvegarde finale
    df_valid.to_parquet(FINAL_FILE)
    print(f"✅ Terminé ! Résultat sauvegardé dans : {FINAL_FILE}")
    print("👉 Vous pouvez maintenant lancer le moteur de recherche sur ce fichier.")

if __name__ == "__main__":
    main()