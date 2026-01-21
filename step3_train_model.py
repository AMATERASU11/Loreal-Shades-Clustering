import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIG ---
LABELED_FILE = "data_lip_labeled.parquet"
MODEL_FILE = "smart_selector_v3.pkl"
CATEGORY_COL = 'category_level_3_name' 

def main():
    print("🚀 Entraînement Stratifié par PRODUIT...")
    
    if not os.path.exists(LABELED_FILE):
        print("❌ Fichier labellisé introuvable.")
        return

    df = pd.read_parquet(LABELED_FILE)
    df = df[df['label_status'] == 'done']
    
    # 1. Préparation des données
    X = np.array(df['features_spatial'].tolist())
    y = df['manual_label_index'].values.astype(int)
    
    # On remplit les trous pour éviter le crash du stratify
    categories = df[CATEGORY_COL].fillna('Unknown').values
    
    print(f"📚 Données : {len(df)} images.")
    print(f"📊 Stratification sur : {np.unique(categories)}")

    # 3. SPLIT UNIQUE (Plus robuste)
    # On passe X, y ET categories dans la même fonction.
    # Ainsi, ils sont mélangés et coupés exactement de la même façon.
    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        X, y, categories,
        test_size=0.2, 
        random_state=42, 
        stratify=categories # On équilibre les Gloss/Lipsticks
    )
    
    print(f"   Train : {len(X_train)} | Test : {len(X_test)}")

    # 4. Entraînement
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Validation Globale
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"🏆 SCORE GLOBAL TEST SET : {acc*100:.2f}%")
    print("="*40)
    
    # 6. Analyse détaillée par Catégorie
    print("\n🔍 Performance par Type de Produit :")
    print(f"{'Catégorie':<20} | {'Score':<10} | {'Volume':<10}")
    print("-" * 45)
    
    # On crée un petit DataFrame temporaire pour analyser les résultats
    results = pd.DataFrame({
        'Category': cat_test,
        'True': y_test,
        'Pred': y_pred
    })
    
    for cat in np.unique(cat_test):
        sub = results[results['Category'] == cat]
        if len(sub) > 0:
            sub_acc = accuracy_score(sub['True'], sub['Pred'])
            print(f"{cat:<20} | {sub_acc*100:.1f}%     | {len(sub)} img")

    # 7. Matrice de Confusion (Pour voir les erreurs 0 vs 1 vs 2)
    print("\n?? Matrice de Confusion (Où se trompe-t-il ?) :")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Sauvegarde
    joblib.dump(clf, MODEL_FILE)
    print(f"\n💾 Modèle sauvegardé : {MODEL_FILE}")

if __name__ == "__main__":
    main()