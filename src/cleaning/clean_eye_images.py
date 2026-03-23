import pandas as pd
import cv2
import os
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..")
INPUT_PARQUET = os.path.join(BASE_DIR, "data", "raw", "data.parquet")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "data", "processed", "data_eye_cleaned.parquet")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")
REPORT_FILE = os.path.join(BASE_DIR, "data", "processed", "corrupted_eye_images_report.csv")

def check_image_integrity(image_path):
    """
    Vérifie si une image peut être lue correctement.
    Retourne: (is_valid, error_message)
    """
    if not os.path.exists(image_path):
        return False, "Fichier introuvable"
    
    if os.path.getsize(image_path) == 0:
        return False, "Fichier vide (0 bytes)"
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, "Impossible de lire avec OpenCV"
        
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return False, f"Dimensions invalides ({w}x{h})"
        
        return True, "OK"
    
    except Exception as e:
        return False, f"Erreur: {str(e)}"

def main():
    print("🧹 Nettoyage des images Eye - Démarrage")
    print("=" * 70)
    
    if not os.path.exists(INPUT_PARQUET):
        print(f"❌ Fichier introuvable : {INPUT_PARQUET}")
        return
    
    # Chargement des données
    print(f"📂 Chargement des données depuis : {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"✅ {len(df)} produits chargés")
    
    # Filtrage catégorie Eye
    if 'category_level_2_name' in df.columns:
        df_eye = df[df['category_level_2_name'] == 'Eye'].copy()
        print(f"🎯 Filtrage 'Eye' : {len(df_eye)} produits")
    else:
        print("⚠️ Colonne 'category_level_2_name' introuvable, traitement de toutes les données")
        df_eye = df.copy()
    
    # Afficher la distribution des sous-catégories
    if 'category_level_3_name' in df_eye.columns:
        print(f"\n📊 Distribution par sous-catégorie:")
        for cat, count in df_eye['category_level_3_name'].value_counts().items():
            print(f"   - {cat}: {count} produits")
    
    # Vérification des images
    print(f"\n🔍 Vérification de l'intégrité des images...")
    print(f"📁 Répertoire images : {IMAGE_DIR}\n")
    
    corrupted_images = []
    valid_count = 0
    error_types = {}
    
    for idx, row in tqdm(df_eye.iterrows(), total=len(df_eye), desc="Vérification"):
        if pd.isna(row.get('image_filename')):
            corrupted_images.append({
                'index': idx,
                'image_filename': 'N/A',
                'product_id': row.get('product_id', 'N/A'),
                'category_level_2_name': row.get('category_level_2_name', 'N/A'),
                'category_level_3_name': row.get('category_level_3_name', 'N/A'),
                'error': 'Nom de fichier manquant'
            })
            error_types['Nom de fichier manquant'] = error_types.get('Nom de fichier manquant', 0) + 1
            continue
        
        image_path = os.path.join(IMAGE_DIR, str(row['image_filename']))
        is_valid, error_msg = check_image_integrity(image_path)
        
        if is_valid:
            valid_count += 1
        else:
            corrupted_images.append({
                'index': idx,
                'image_filename': row['image_filename'],
                'product_id': row.get('product_id', 'N/A'),
                'category_level_2_name': row.get('category_level_2_name', 'N/A'),
                'category_level_3_name': row.get('category_level_3_name', 'N/A'),
                'error': error_msg
            })
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
    
    # Résumé de la vérification
    print(f"\n" + "=" * 70)
    print("📊 RÉSULTAT DE LA VÉRIFICATION")
    print("=" * 70)
    print(f"✅ Images valides        : {valid_count}")
    print(f"❌ Images corrompues     : {len(corrupted_images)}")
    print(f"📈 Taux de réussite      : {valid_count / len(df_eye) * 100:.2f}%")
    
    if len(corrupted_images) > 0:
        print(f"\n🔍 Types d'erreurs rencontrées :")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {error_type}: {count}")
        
        # Sauvegarde du rapport d'erreurs
        df_corrupted = pd.DataFrame(corrupted_images)
        df_corrupted.to_csv(REPORT_FILE, index=False)
        print(f"\n📄 Rapport d'erreurs sauvegardé : {REPORT_FILE}")
    
    # Nettoyage : retirer les images corrompues
    if len(corrupted_images) > 0:
        corrupted_filenames = set([item['image_filename'] for item in corrupted_images])
        df_eye_clean = df_eye[~df_eye['image_filename'].isin(corrupted_filenames)].copy()
        print(f"\n🧹 Après nettoyage : {len(df_eye_clean)} produits conservés")
    else:
        df_eye_clean = df_eye.copy()
        print(f"\n✅ Aucune image corrompue détectée - tous les produits conservés")
    
    # Sauvegarde du dataset nettoyé
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df_eye_clean.to_parquet(OUTPUT_PARQUET, index=False)
    
    print(f"\n💾 Dataset nettoyé sauvegardé : {OUTPUT_PARQUET}")
    print("=" * 70)
    print("✅ Nettoyage terminé avec succès !")
    print("👉 Vous pouvez maintenant lancer l'extraction de features")

if __name__ == "__main__":
    main()
