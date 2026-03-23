import pandas as pd
import numpy as np
import os
import cv2
from rembg import remove
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..")
INPUT_PARQUET = os.path.join(BASE_DIR, "data", "processed", "data_eye_cleaned.parquet")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "data", "processed", "data_eye_features.parquet")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")

# Configuration KMeans
N_CLUSTERS = 5  # 5 clusters comme pour Lip

def get_hsv(rgb):
    """RGB [0-255] -> HSV [0-1]"""
    rgb_norm = np.array(rgb).reshape(1, 1, 3) / 255.0
    return color.rgb2hsv(rgb_norm)[0][0]

def get_roi_by_category(img_rgb, category_l3):
    """
    Découpe la zone d'intérêt selon la catégorie de produit Eye.
    
    Stratégies:
    - Eye Liner / Mascara : Zone centrale (30%) pour capturer la pointe/applicateur
    - Eyebrow : Zone centrale (50%) pour le crayon
    - Eye Shadow : Image complète pour les palettes
    """
    h, w = img_rgb.shape[:2]
    
    if category_l3 in ['Eye Liner', 'Mascara']:
        crop_ratio = 0.30
        y1 = int(h * (0.5 - crop_ratio/2))
        y2 = int(h * (0.5 + crop_ratio/2))
        x1 = int(w * (0.5 - crop_ratio/2))
        x2 = int(w * (0.5 + crop_ratio/2))
        return img_rgb[y1:y2, x1:x2]
        
    elif category_l3 == 'Eyebrow':
        crop_ratio = 0.50
        y1 = int(h * (0.5 - crop_ratio/2))
        y2 = int(h * (0.5 + crop_ratio/2))
        x1 = int(w * (0.5 - crop_ratio/2))
        x2 = int(w * (0.5 + crop_ratio/2))
        return img_rgb[y1:y2, x1:x2]
    
    else:  # Eye Shadow et autres
        return img_rgb

def get_hsv_params_by_category(category_l3):
    """
    Retourne les paramètres HSV optimaux selon la catégorie.
    
    Returns: (s_min, v_min, v_max)
    """
    if category_l3 == 'Eye Liner':
        return 30, 50, 220
    elif category_l3 == 'Mascara':
        return 25, 40, 200
    elif category_l3 == 'Eyebrow':
        return 20, 50, 180
    else:  # Eye Shadow
        return 30, 60, 250

def process_image_complete(img_path, category_l3):
    """
    Extrait les features d'une image de produit Eye.
    
    Retourne: (sorted_centers, sorted_areas, feature_vector)
    """
    if not os.path.exists(img_path): 
        return None, None, None

    try:
        # 1. RemBG & Nettoyage
        img_pil = Image.open(img_path).convert("RGB")
        img_no_bg = remove(img_pil)
        img_np = np.array(img_no_bg)
        
        rgb = img_np[:, :, :3]
        alpha = img_np[:, :, 3]
        
        mask = (alpha > 10)
        
        # 2. ROI adapté à la catégorie
        img_rgb = rgb.copy()
        roi = get_roi_by_category(img_rgb, category_l3)
        
        # Appliquer le masque alpha sur la ROI
        h_orig, w_orig = img_rgb.shape[:2]
        h_roi, w_roi = roi.shape[:2]
        
        # Redimensionner le masque pour correspondre à la ROI
        if roi.shape[:2] != img_rgb.shape[:2]:
            y_start = (h_orig - h_roi) // 2
            x_start = (w_orig - w_roi) // 2
            mask_roi = mask[y_start:y_start+h_roi, x_start:x_start+w_roi]
        else:
            mask_roi = mask
        
        valid_pixels = roi[mask_roi]
        
        if len(valid_pixels) < 50: 
            return None, None, None

        # 3. KMeans (5 Clusters)
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=1, random_state=42)
        kmeans.fit(valid_pixels)
        
        raw_centers = kmeans.cluster_centers_.astype(int)
        
        # 4. Calcul des Surfaces
        counts = np.bincount(kmeans.labels_, minlength=N_CLUSTERS)
        total_pixels = len(valid_pixels)
        raw_props = counts / total_pixels

        # 5. PRÉPARATION & TRI
        candidates = []
        for i in range(N_CLUSTERS):
            c = raw_centers[i]
            p = raw_props[i]
            h, s, v = get_hsv(c)
            candidates.append({
                'rgb': c.tolist(),
                'h': h, 's': s, 'v': v, 
                'area': p
            })

        # TRI : Toujours par saturation décroissante
        # Index 0 = Le plus vif, Index 4 = Le plus gris
        candidates.sort(key=lambda x: x['s'], reverse=True)

        # 6. Construction du Vecteur X (37 features)
        feature_vector = []
        for c in candidates:
            feature_vector.extend([c['h'], c['s'], c['v'], c['area'], 
                                   c['rgb'][0], c['rgb'][1], c['rgb'][2]])
        
        # Features comparatives (Delta entre le 1er et le 2ème)
        feature_vector.append(candidates[0]['s'] - candidates[1]['s'])
        feature_vector.append(candidates[0]['area'] - candidates[1]['area'])

        sorted_centers = [c['rgb'] for c in candidates]
        sorted_areas = [c['area'] for c in candidates]

        return sorted_centers, sorted_areas, feature_vector

    except Exception as e:
        return None, None, None

def main():
    print(f"🚀 Extraction Features Eye - K={N_CLUSTERS}")
    print("=" * 70)
    
    if not os.path.exists(INPUT_PARQUET):
        print(f"❌ Fichier introuvable : {INPUT_PARQUET}")
        print("👉 Lancez d'abord le nettoyage des images (clean_eye_images.py)")
        return

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"📂 Chargement : {len(df)} produits Eye")
    
    # Distribution par catégorie
    if 'category_level_3_name' in df.columns:
        print(f"\n📊 Distribution par sous-catégorie:")
        for cat, count in df['category_level_3_name'].value_counts().items():
            print(f"   - {cat}: {count} produits")
    
    all_centers = []
    all_areas = []
    all_features = []
    
    print(f"\n⚡ Traitement de {len(df)} images...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extraction"):
        filename = row.get('image_filename')
        category = row.get('category_level_3_name', 'Unknown')
        
        if pd.isna(filename):
            all_centers.append(None)
            all_areas.append(None)
            all_features.append(None)
            continue
        
        path = os.path.join(IMAGE_DIR, str(filename))
        centers, areas, feats = process_image_complete(path, category)
        
        all_centers.append(centers)
        all_areas.append(areas)
        all_features.append(feats)
        
    df['kmeans_centers'] = all_centers
    df['kmeans_areas'] = all_areas
    df['features_spatial'] = all_features
    
    # Filtrer les lignes invalides
    df_clean = df[df['features_spatial'].notna()].copy()
    
    print(f"\n📊 Résumé :")
    print(f"   - Images traitées avec succès : {len(df_clean)}")
    print(f"   - Échecs d'extraction         : {len(df) - len(df_clean)}")
    print(f"   - Taux de réussite            : {len(df_clean) / len(df) * 100:.2f}%")
    
    # Sauvegarde
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    df_clean.to_parquet(OUTPUT_PARQUET)
    
    print(f"\n💾 Sauvegardé sous : {OUTPUT_PARQUET}")
    print("=" * 70)
    print("✅ Extraction terminée avec succès !")
    print("👉 Vous pouvez maintenant lancer la labellisation")

if __name__ == "__main__":
    main()
