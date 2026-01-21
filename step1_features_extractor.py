import pandas as pd
import numpy as np
import os
from rembg import remove
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR 
INPUT_PARQUET = os.path.join(BASE_DIR, "data.parquet")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "data_lip_features.parquet")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# ON PASSE A 5 CLUSTERS POUR CAPTER LA COULEUR NOYÉE
N_CLUSTERS = 5 

def get_hsv(rgb):
    """RGB [0-255] -> HSV [0-1]"""
    rgb_norm = np.array(rgb).reshape(1, 1, 3) / 255.0
    return color.rgb2hsv(rgb_norm)[0][0]

def process_image_complete(img_path):
    if not os.path.exists(img_path): 
        return None, None, None

    try:
        # 1. RemBG & Nettoyage (Version PURE - SANS FILTRE HSV)
        img_pil = Image.open(img_path).convert("RGB")
        img_no_bg = remove(img_pil)
        img_np = np.array(img_no_bg)
        
        rgb = img_np[:, :, :3]
        alpha = img_np[:, :, 3]
        
        mask = (alpha > 10)
        valid_pixels = rgb[mask]
        
        if len(valid_pixels) < 50: 
            return None, None, None

        # 2. KMeans (5 Clusters)
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=1, random_state=42)
        kmeans.fit(valid_pixels)
        
        raw_centers = kmeans.cluster_centers_.astype(int)
        
        # 3. Calcul des Surfaces
        counts = np.bincount(kmeans.labels_, minlength=N_CLUSTERS)
        total_pixels = len(valid_pixels)
        raw_props = counts / total_pixels

        # 4. PRÉPARATION & TRI
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

        # 5. Construction du Vecteur X (Adapté pour 5 candidats)
        feature_vector = []
        for c in candidates:
            feature_vector.extend([c['h'], c['s'], c['v'], c['area'], 
                                   c['rgb'][0], c['rgb'][1], c['rgb'][2]])
        
        # Features comparatives (Delta entre le 1er et le 2ème reste le plus pertinent)
        feature_vector.append(candidates[0]['s'] - candidates[1]['s'])
        feature_vector.append(candidates[0]['area'] - candidates[1]['area'])

        sorted_centers = [c['rgb'] for c in candidates]
        sorted_areas = [c['area'] for c in candidates]

        return sorted_centers, sorted_areas, feature_vector

    except Exception:
        return None, None, None

def main():
    print(f"🚀 Extraction K={N_CLUSTERS} (Mode Robustesse)")
    print("⏳ Relancement nécessaire pour capter les couleurs difficiles...")
    
    if not os.path.exists(INPUT_PARQUET):
        print("❌ Fichier data.parquet introuvable.")
        return

    df = pd.read_parquet(INPUT_PARQUET)
    if 'category_level_2_name' in df.columns:
        df = df[df['category_level_2_name'] == 'Lip'].copy()
        
    all_centers = []
    all_areas = []
    all_features = []
    
    print(f"⚡ Traitement de {len(df)} images...")
    
    for filename in tqdm(df['image_filename']):
        path = os.path.join(IMAGE_DIR, str(filename))
        centers, areas, feats = process_image_complete(path)
        all_centers.append(centers)
        all_areas.append(areas)
        all_features.append(feats)
        
    df['kmeans_centers'] = all_centers
    df['kmeans_areas'] = all_areas
    df['features_spatial'] = all_features
    
    df_clean = df[df['features_spatial'].notna()].copy()
    
    df_clean.to_parquet(OUTPUT_PARQUET)
    print(f"💾 Sauvegardé sous : {OUTPUT_PARQUET}")
    print("👉 Vous pouvez maintenant labelliser avec beaucoup plus de succès.")

if __name__ == "__main__":
    main()