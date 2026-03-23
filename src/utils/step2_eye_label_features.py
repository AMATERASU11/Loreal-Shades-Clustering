import cv2
import pandas as pd
import numpy as np
import os

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "..")
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "data_eye_features.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "data_eye_labeled.parquet")
IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw", "images")

# Objectifs de labeling par sous-catégorie Eye
TARGET_PER_CLASS = 100  # 100 exemples par sous-catégorie
TARGET_CLASSES = ['Eye Shadow', 'Eye Liner', 'Mascara', 'Eyebrow']
CATEGORY_COL = 'category_level_3_name'

# Configuration affichage
SCREEN_W, SCREEN_H = 1920, 1080
IMG_DISPLAY_H = 600

def create_display(img_path, candidates, category):
    """
    Crée une image composite pour la labellisation.
    
    Args:
        img_path: Chemin de l'image produit
        candidates: Liste des 5 couleurs candidates RGB
        category: Catégorie du produit
    
    Returns:
        Canvas avec image + nuancier + instructions
    """
    canvas = np.ones((SCREEN_H, SCREEN_W, 3), dtype=np.uint8) * 255
    
    if not os.path.exists(img_path): 
        return None
    img = cv2.imread(img_path)
    if img is None: 
        return None

    # Panneau des candidats (nuancier)
    swatch_w = 200
    
    # Redimensionnement de l'image pour s'assurer qu'elle tient dans l'écran
    h, w = img.shape[:2]
    
    # Calculer la largeur max disponible pour l'image (écran - swatch - marges)
    max_img_w = SCREEN_W - swatch_w - 100  # 100px de marge
    
    # Choisir le ratio qui respecte les contraintes de hauteur ET largeur
    ratio_h = IMG_DISPLAY_H / h
    ratio_w = max_img_w / w
    ratio = min(ratio_h, ratio_w)  # Prendre le plus petit pour tout faire tenir
    
    img_resized = cv2.resize(img, (int(w * ratio), int(h * ratio)))
    final_img_h = img_resized.shape[0]
    
    # Ajuster la hauteur du nuancier pour correspondre à l'image
    candidates_img = np.zeros((final_img_h, swatch_w, 3), dtype=np.uint8) + 255
    
    if candidates is not None:
        step_h = final_img_h // len(candidates)
        
        for i, color in enumerate(candidates):
            # Couleur BGR pour OpenCV
            bgr = [int(color[2]), int(color[1]), int(color[0])]
            y_start, y_end = i * step_h, (i + 1) * step_h
            
            # Rectangle de couleur
            cv2.rectangle(candidates_img, (0, y_start), (swatch_w, y_end), bgr, -1)
            cv2.rectangle(candidates_img, (0, y_start), (swatch_w, y_end), (0,0,0), 1)
            
            # Texte avec contraste automatique
            lum = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            txt_col = (255,255,255) if lum < 128 else (0,0,0)
            
            # Numéro du choix
            cv2.putText(candidates_img, f"{i+1}", (20, y_start + step_h//2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_col, 2)

    # Assemblage image + nuancier
    content = np.hstack((img_resized, candidates_img))
    h_c, w_c = content.shape[:2]
    
    # S'assurer que le contenu ne dépasse pas l'écran
    if w_c > SCREEN_W:
        # Redimensionner encore si nécessaire
        scale = SCREEN_W / w_c
        content = cv2.resize(content, (SCREEN_W, int(h_c * scale)))
        h_c, w_c = content.shape[:2]
    
    y_off = max(0, (SCREEN_H - h_c) // 2)
    x_off = max(0, (SCREEN_W - w_c) // 2)
    
    # Ajuster les offsets si le contenu dépasse encore
    y_end = min(y_off + h_c, SCREEN_H)
    x_end = min(x_off + w_c, SCREEN_W)
    h_c = y_end - y_off
    w_c = x_end - x_off
    
    canvas[y_off:y_end, x_off:x_end] = content[:h_c, :w_c]
    
    # Informations en haut
    cv2.putText(canvas, f"Categorie: {category}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    
    # Instructions en bas
    cv2.putText(canvas, "1 a 5 : Choisir la meilleure couleur | X : Jeter | S : Skip | ESC : Sauver & Quitter", 
                (50, SCREEN_H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)
    
    return canvas

def main():
    print("🎨 Labellisation Eye - Démarrage")
    print("=" * 70)
    
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"🔄 Reprise du fichier existant : {OUTPUT_FILE}")
    else:
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Fichier introuvable : {INPUT_FILE}")
            print("👉 Lancez d'abord l'extraction de features (step1_eye_features_extractor.py)")
            return
        
        df = pd.read_parquet(INPUT_FILE)
        df['manual_label_index'] = -1
        df['label_status'] = 'todo'
        print(f"🆕 Nouveau fichier créé depuis : {INPUT_FILE}")

    # Sélection équilibrée par catégorie
    todo_indices = []
    
    print(f"\n📊 Objectif: {TARGET_PER_CLASS} exemples par catégorie")
    print("-" * 70)
    
    for cat in TARGET_CLASSES:
        if cat not in df[CATEGORY_COL].values:
            print(f"⚠️ {cat:20s}: Non trouvé dans le dataset")
            continue
        
        done = len(df[(df[CATEGORY_COL] == cat) & (df['label_status'] == 'done')])
        needed = TARGET_PER_CLASS - done
        
        if needed > 0:
            candidates_idx = df[(df[CATEGORY_COL] == cat) & (df['label_status'] == 'todo')].index.tolist()
            if len(candidates_idx) > 0:
                picked = np.random.choice(candidates_idx, min(len(candidates_idx), needed), replace=False)
                todo_indices.extend(picked)
                print(f"✅ {cat:20s}: {done:3d} fait | {needed:3d} restant | {len(picked):3d} sélectionné")
            else:
                print(f"⚠️ {cat:20s}: {done:3d} fait | Aucun candidat disponible")
        else:
            print(f"✅ {cat:20s}: {done:3d} fait | Objectif atteint !")
    
    if len(todo_indices) == 0:
        print("\n🎉 Tous les objectifs sont atteints !")
        return
    
    np.random.shuffle(todo_indices)
    print(f"\n📝 {len(todo_indices)} images à traiter dans cette session.")
    print("-" * 70)

    # Interface de labellisation
    cv2.namedWindow("Eye Labeler", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Eye Labeler", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    count = 0
    for idx in todo_indices:
        row = df.loc[idx]
        img_path = os.path.join(IMAGE_DIR, str(row['image_filename']))
        centers = row['kmeans_centers']

        if centers is None or len(centers) == 0:
            df.at[idx, 'label_status'] = 'ignore'
            continue

        vis = create_display(img_path, centers, row[CATEGORY_COL])
        if vis is None: 
            df.at[idx, 'label_status'] = 'ignore'
            continue
        
        cv2.imshow("Eye Labeler", vis)
        
        valid = False
        while not valid:
            key = cv2.waitKey(0)
            
            # Touches 1 à 5 pour sélectionner
            possible_keys = [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]
            
            if key in possible_keys:
                choice = key - ord('1')  # Convertit '1'->0, '5'->4
                
                if choice < len(centers):
                    df.at[idx, 'manual_label_index'] = choice
                    df.at[idx, 'label_status'] = 'done'
                    valid = True
                    count += 1
                    print(f"✅ [{count}/{len(todo_indices)}] Image {idx} : Choix {choice+1} | Catégorie: {row[CATEGORY_COL]}")
            
            elif key in [ord('x'), ord('X')]:
                df.at[idx, 'label_status'] = 'ignore'
                valid = True
                print(f"❌ Image {idx} ignorée")
            
            elif key in [ord('s'), ord('S')]:
                valid = True
                print(f"⏭️ Image {idx} sautée")
            
            elif key == 27:  # ESC
                print("\n💾 Sauvegarde en cours...")
                os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
                df.to_parquet(OUTPUT_FILE)
                print(f"✅ Sauvegardé : {OUTPUT_FILE}")
                cv2.destroyAllWindows()
                return

        # Sauvegarde automatique tous les 10 labels
        if count % 10 == 0 and count > 0: 
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            df.to_parquet(OUTPUT_FILE)
            print(f"💾 Auto-sauvegarde ({count} labels)")

    # Sauvegarde finale
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_parquet(OUTPUT_FILE)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("🎉 Session de labellisation terminée !")
    print(f"💾 Fichier sauvegardé : {OUTPUT_FILE}")
    print("👉 Vous pouvez maintenant entraîner le modèle")
    print("=" * 70)

if __name__ == "__main__":
    main()
