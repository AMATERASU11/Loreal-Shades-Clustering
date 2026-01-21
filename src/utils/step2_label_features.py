import cv2
import pandas as pd
import numpy as np
import os

# --- CONFIG ---
# On pointe vers le nouveau fichier K=5
INPUT_FILE = "data_lip_features.parquet" 
OUTPUT_FILE = "data_lip_labeled.parquet" # Nouveau fichier de sortie aussi
IMAGE_DIR = "images"

# On garde les mêmes objectifs
TARGET_PER_CLASS = 125 
TARGET_CLASSES = ['Lipstick', 'Lip Gloss', 'Lip Liner', 'Lip Tint']
CATEGORY_COL = 'category_level_3_name'

# Configuration affichage
SCREEN_W, SCREEN_H = 1920, 1080
IMG_DISPLAY_H = 600

def create_display(img_path, candidates, category):
    canvas = np.ones((SCREEN_H, SCREEN_W, 3), dtype=np.uint8) * 255
    
    if not os.path.exists(img_path): 
        return None
    img = cv2.imread(img_path)
    if img is None: 
        return None

    # Redim Image
    h, w = img.shape[:2]
    ratio = IMG_DISPLAY_H / h
    img_resized = cv2.resize(img, (int(w * ratio), IMG_DISPLAY_H))
    
    # Panneau Candidats
    swatch_w = 200
    candidates_img = np.zeros((IMG_DISPLAY_H, swatch_w, 3), dtype=np.uint8) + 255
    
    if candidates is not None:
        # C'est ici que la magie opère : step_h s'adapte au nombre de candidats (5)
        step_h = IMG_DISPLAY_H // len(candidates)
        
        for i, color in enumerate(candidates):
            # Couleur BGR pour OpenCV
            bgr = [int(color[2]), int(color[1]), int(color[0])]
            y_start, y_end = i * step_h, (i + 1) * step_h
            
            cv2.rectangle(candidates_img, (0, y_start), (swatch_w, y_end), bgr, -1)
            cv2.rectangle(candidates_img, (0, y_start), (swatch_w, y_end), (0,0,0), 1)
            
            # Texte contrasté
            lum = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            txt_col = (255,255,255) if lum < 128 else (0,0,0)
            
            # Affichage "Choix X"
            cv2.putText(candidates_img, f"{i+1}", (20, y_start + step_h//2 + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, txt_col, 2)
            
            # Affichage de la part (Surface) si dispo (Optionnel, juste pour info visuelle)
            # cv2.putText(candidates_img, f"{proportions[i]*100:.0f}%", (80, y_start + step_h//2 + 10), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_col, 1)

    # Assemblage
    content = np.hstack((img_resized, candidates_img))
    h_c, w_c = content.shape[:2]
    y_off = (SCREEN_H - h_c) // 2
    x_off = (SCREEN_W - w_c) // 2
    
    canvas[y_off:y_off+h_c, x_off:x_off+w_c] = content
    
    # Info
    cv2.putText(canvas, f"Cat: {category}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    # Mise à jour des instructions
    cv2.putText(canvas, "1 à 5 : Choisir | X : Jeter | S : Skip | ESC : Sauver", 
                (50, SCREEN_H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2)
    
    return canvas

def main():
    print("🎨 Démarrage Labellisation (Mode K=5)...")
    
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_parquet(OUTPUT_FILE)
        print(f"🔄 Reprise de {OUTPUT_FILE}")
    else:
        # On lit le fichier généré par step1 (Version K=5)
        if not os.path.exists(INPUT_FILE):
            print(f"❌ Erreur : {INPUT_FILE} introuvable. Avez-vous relancé l'extraction ?")
            return
        df = pd.read_parquet(INPUT_FILE)
        df['manual_label_index'] = -1
        df['label_status'] = 'todo'
        print(f"🆕 Nouveau depuis {INPUT_FILE}")

    # Sélection équilibrée
    todo_indices = []
    for cat in TARGET_CLASSES:
        done = len(df[(df[CATEGORY_COL] == cat) & (df['label_status'] == 'done')])
        needed = TARGET_PER_CLASS - done
        if needed > 0:
            candidates = df[(df[CATEGORY_COL] == cat) & (df['label_status'] == 'todo')].index.tolist()
            picked = np.random.choice(candidates, min(len(candidates), needed), replace=False)
            todo_indices.extend(picked)
    
    np.random.shuffle(todo_indices)
    print(f"📝 {len(todo_indices)} images à traiter.")

    cv2.namedWindow("Labeler", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Labeler", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
            continue
        
        cv2.imshow("Labeler", vis)
        
        valid = False
        while not valid:
            key = cv2.waitKey(0)
            
            # --- MODIFICATION CLÉ : GESTION DE 1 à 5 ---
            possible_keys = [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]
            
            if key in possible_keys:
                choice = key - ord('1') # Convertit '1'->0, '5'->4
                
                if choice < len(centers):
                    df.at[idx, 'manual_label_index'] = choice
                    df.at[idx, 'label_status'] = 'done'
                    valid = True
                    print(f"✅ Image {idx} : Choix {choice+1}")
            
            elif key in [ord('x'), ord('X')]:
                df.at[idx, 'label_status'] = 'ignore'
                valid = True
            elif key in [ord('s'), ord('S')]:
                valid = True
            elif key == 27: # ESC
                df.to_parquet(OUTPUT_FILE)
                return

        count += 1
        if count % 10 == 0: 
            df.to_parquet(OUTPUT_FILE)

    df.to_parquet(OUTPUT_FILE)
    print("🎉 Session terminée.")

if __name__ == "__main__":
    main()