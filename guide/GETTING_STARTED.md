# Getting Started — ShadeNail

Guide pas-a-pas pour lancer votre premiere inference ShadeNail.

> **Prerequis** : avoir termine l'[installation](INSTALLATION.md).

## Workflow rapide (TL;DR)

```bash
cd Loreal-Shades-Clustering           # 1. Se placer a la racine
# Activer le venv (.venv ou poetry)
source .venv/bin/activate              # Linux/Mac
# .venv\Scripts\Activate.ps1           # Windows PowerShell

# 2. Placer vos images dans data/raw/images/
# 3. Preparer votre CSV avec une colonne image_filename
# 4. Lancer l'inference
python -m src.main --infer --input data/raw/mon_fichier.csv
# 5. Resultat dans outputs/predictions/predictions.csv
```

---

## Etape 1 — Preparer les images

Placez toutes vos images de vernis a ongles dans :

```
data/raw/images/
```

Le dossier existe deja (vide). Copiez-y vos images :
```bash
cp /chemin/vers/mes_images/*.jpg data/raw/images/
# ou glisser-deposer sous Windows/Mac
```

Formats supportes : `.jpg`, `.jpeg`, `.png`

## Etape 2 — Preparer le CSV d'entree

Creez un fichier CSV avec au minimum une colonne `image_filename` :

```csv
image_filename,product_id,product_name
vernis_rouge_01.jpg,P001,Rouge Passion
vernis_bleu_02.jpg,P002,Bleu Ocean
vernis_nude_03.jpg,P003,Nude Rose
```

Colonnes :
- **`image_filename`** (obligatoire) : nom du fichier image tel qu'il apparait dans `data/raw/images/`
- **`product_id`** (optionnel) : identifiant produit, sera recopie dans la sortie
- **`product_name`** (optionnel) : nom du produit
- **`shade_name`** (optionnel) : nom de la teinte — utilise par le Tier 2 (NLP Prior) pour ameliorer la prediction

> **Note** : si `shade_name` contient un nom de couleur (ex: "Rouge Passion", "Blue Lagoon"), le pipeline l'utilise comme indice supplementaire via le NLP Prior.

Placez ce CSV quelque part dans le projet, par exemple :
```
data/raw/mon_fichier.csv
```

## Etape 3 — Lancer l'inference

```bash
python -m src.main --infer --input data/raw/mon_fichier.csv
```

Le pipeline va :
1. **Precomputer le cache rembg** : suppression du fond de chaque image (~1-2 sec/image). Cache stocke dans `outputs/cache/` pour ne pas refaire le travail.
2. **Predire** pour chaque ligne du CSV :
   - Tier 0 (Gatekeeper) : detecte les kits multi-vernis et les incolores
   - Tier 1 (Texture) : detecte glitter/shimmer/holographique
   - Tier 3 (Vision) : KMeans K=5 sur les pixels, extraction de 35 features
   - Tier 2 (NLP Prior) : si `shade_name` existe, calcule les features deltaE
   - Tier 4 (ShadeNail Ranking) : XGBoost predit le deltaE par cluster, choisit le meilleur

### Duree estimee

| Nombre d'images | Premiere fois (avec cache) | Suivantes (cache existant) |
|-----------------|---------------------------|---------------------------|
| 10 | ~30 sec | ~5 sec |
| 100 | ~3 min | ~30 sec |
| 1000 | ~30 min | ~5 min |

## Etape 4 — Lire les resultats

La sortie est dans :
```
outputs/predictions/predictions.csv
```

Colonnes de sortie :
| Colonne | Description |
|---------|-------------|
| `image_filename` | Nom du fichier image |
| `predicted_rgb` | Couleur predite en RGB `(R, G, B)` |
| `predicted_hex` | Couleur predite en hexadecimal `#RRGGBB` |
| `predicted_lab` | Couleur predite en Lab `(L, a, b)` |
| `chosen_cluster` | Index du cluster choisi (0-4) |
| `tier` | Quel tier a produit la prediction |
| `is_kit` | True si detecte comme kit multi-vernis |
| `is_clear` | True si detecte comme incolore/transparent |
| `has_glitter` | True si detecte comme glitter/shimmer |

## Autres commandes utiles

### Prediction pour un seul produit

```bash
python -m src.main --predict --product_id P001
```

### Entrainement (necessite le dataset labellise)

```bash
python -m src.main --train
```

> Necessite `data/labeled/nail_all_labeled_1261.parquet` — demander a l'equipe.

### Evaluation

```bash
python -m src.main --evaluate
```

### Pipeline complet (train + evaluate)

```bash
python -m src.main --full
```

## Structure des dossiers

Apres une inference, votre projet ressemble a :
```
Loreal-Shades-Clustering/
  data/
    raw/
      images/           <-- vos images ici
      mon_fichier.csv   <-- votre CSV d'entree
  outputs/
    cache/              <-- cache rembg (genere automatiquement)
    models/
      shadenail_xgb.pkl <-- modele XGBoost (inclus dans le repo)
      shadenail_meta.json
    predictions/
      predictions.csv   <-- resultats ici
  src/
    ...
```

---

**Voir aussi : [ARCHITECTURE.md](ARCHITECTURE.md) pour comprendre le fonctionnement interne du pipeline.**
