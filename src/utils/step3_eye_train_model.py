import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..")
LABELED_FILE = os.path.join(BASE_DIR, "data", "processed", "data_eye_labeled.parquet")
MODEL_FILE = os.path.join(BASE_DIR, "outputs", "models", "smart_selector_eye.pkl")
CATEGORY_COL = 'category_level_3_name'

def main():
    print("🚀 Entraînement du Modèle Eye - Démarrage")
    print("=" * 70)
    
    if not os.path.exists(LABELED_FILE):
        print(f"❌ Fichier introuvable : {LABELED_FILE}")
        print("👉 Lancez d'abord la labellisation (step2_eye_label_features.py)")
        return

    # Chargement des données labellisées
    df = pd.read_parquet(LABELED_FILE)
    if 'is_multicolor' not in df.columns:
        df['is_multicolor'] = False
    df['is_multicolor'] = df['is_multicolor'].fillna(False).astype(bool)

    df_done = df[df['label_status'] == 'done'].copy()
    excluded_multi = int(df_done['is_multicolor'].sum())
    df_done = df_done[~df_done['is_multicolor']].copy()
    
    if len(df_done) == 0:
        print("❌ Aucune donnée labellisée trouvée")
        print("👉 Lancez la labellisation pour créer des exemples d'entraînement")
        return
    
    print(f"📚 Données chargées : {len(df_done)} exemples labellisés")
    if excluded_multi > 0:
        print(f"🎨 Exemples multi-couleurs exclus de l'entraînement : {excluded_multi}")
    
    # Distribution par catégorie
    print(f"\n📊 Distribution par catégorie:")
    for cat, count in df_done[CATEGORY_COL].value_counts().items():
        print(f"   - {cat}: {count} exemples")
    
    # 1. Préparation des features et labels
    X = np.array(df_done['features_spatial'].tolist())
    y = df_done['manual_label_index'].values.astype(int)
    
    # Catégories pour stratification (conversion en numpy array)
    categories = np.array(df_done[CATEGORY_COL].fillna('Unknown').tolist())
    
    print(f"\n🔍 Vérification des données:")
    print(f"   - Shape des features (X) : {X.shape}")
    print(f"   - Shape des labels (y)   : {y.shape}")
    print(f"   - Labels uniques         : {np.unique(y)}")
    print(f"   - Catégories uniques     : {np.unique(categories)}")

    # 2. Split stratifié train/test
    print(f"\n🔀 Split train/test (stratifié par catégorie)...")
    
    try:
        X_train, X_test, y_train, y_test, _, cat_test = train_test_split(
            X, y, categories,
            test_size=0.2, 
            random_state=42, 
            stratify=categories
        )
        print(f"   ✅ Train : {len(X_train)} exemples | Test : {len(X_test)} exemples")
    except ValueError as e:
        print(f"   ⚠️ Split stratifié impossible : {e}")
        print(f"   📝 Split non-stratifié utilisé à la place")
        X_train, X_test, y_train, y_test, _, cat_test = train_test_split(
            X, y, categories,
            test_size=0.2, 
            random_state=42
        )
        print(f"   ✅ Train : {len(X_train)} exemples | Test : {len(X_test)} exemples")

    # 3. Entraînement du modèle
    print(f"\n🧠 Entraînement du Gradient Boosting Classifier...")
    clf = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"   ✅ Modèle entraîné")
    
    # 4. Évaluation globale
    print(f"\n📊 ÉVALUATION GLOBALE")
    print("=" * 70)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"🎯 Accuracy Train : {train_acc*100:.2f}%")
    print(f"🎯 Accuracy Test  : {test_acc*100:.2f}%")
    
    # 5. Performance par catégorie
    print(f"\n📊 PERFORMANCE PAR CATÉGORIE (Test Set)")
    print("-" * 70)
    print(f"{'Catégorie':<20} | {'Accuracy':<10} | {'Nb exemples':<12}")
    print("-" * 70)
    
    results = pd.DataFrame({
        'Category': cat_test,
        'True': y_test,
        'Pred': y_test_pred
    })
    
    for cat in np.unique(cat_test):
        sub = results[results['Category'] == cat]
        if len(sub) > 0:
            sub_acc = accuracy_score(sub['True'], sub['Pred'])
            print(f"{cat:<20} | {sub_acc*100:>6.1f}%    | {len(sub):>4d} exemples")

    # 6. Matrice de confusion
    print(f"\n🔍 MATRICE DE CONFUSION (Test Set)")
    print("-" * 70)
    cm = confusion_matrix(y_test, y_test_pred)
    print("Lignes = Vraie couleur | Colonnes = Prédiction")
    print(cm)
    
    # Interpréter la matrice
    print(f"\n💡 Interprétation :")
    diagonal_sum = np.trace(cm)
    total_sum = np.sum(cm)
    print(f"   - Prédictions correctes (diagonale) : {diagonal_sum}/{total_sum}")
    
    # 7. Rapport de classification détaillé
    print(f"\n📋 RAPPORT DE CLASSIFICATION")
    print("-" * 70)
    unique_labels = np.unique(np.concatenate([y_test, y_test_pred]))
    target_names = [f"Candidat {i}" for i in unique_labels]
    print(classification_report(y_test, y_test_pred, 
                                labels=unique_labels,
                                target_names=target_names))

    # 8. Importance des features (top 10)
    print(f"\n🔑 TOP 10 DES FEATURES LES PLUS IMPORTANTES")
    print("-" * 70)
    feature_importance = clf.feature_importances_
    feature_names = []
    
    # Noms des features
    for i in range(5):  # 5 clusters
        feature_names.extend([
            f"C{i}_H", f"C{i}_S", f"C{i}_V", f"C{i}_Area",
            f"C{i}_R", f"C{i}_G", f"C{i}_B"
        ])
    feature_names.extend(["Delta_S", "Delta_Area"])
    
    # Trier par importance
    indices = np.argsort(feature_importance)[::-1]
    
    for i in range(min(10, len(feature_importance))):
        idx = indices[i]
        print(f"   {i+1}. {feature_names[idx]:<15s} : {feature_importance[idx]:.4f}")

    # 9. Sauvegarde du modèle
    print(f"\n💾 Sauvegarde du modèle...")
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(clf, MODEL_FILE)
    print(f"✅ Modèle sauvegardé : {MODEL_FILE}")
    
    # Résumé final
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("=" * 70)
    print(f"📊 Accuracy finale : {test_acc*100:.2f}%")
    print(f"📚 Exemples utilisés : {len(df_done)}")
    print(f"💾 Modèle : {MODEL_FILE}")
    print("👉 Vous pouvez maintenant lancer l'inférence en production")
    print("=" * 70)

if __name__ == "__main__":
    main()