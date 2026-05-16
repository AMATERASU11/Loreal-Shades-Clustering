# Architecture — ShadeNail Pipeline

Vue d'ensemble du pipeline d'extraction de couleur de vernis a ongles.

## Vue d'ensemble

ShadeNail est un pipeline en **5 tiers** (cascade). Chaque tier traite un aspect specifique et peut court-circuiter les tiers suivants.

```
Image + metadata
      |
  Tier 0 : Gatekeeper         -- detecte kits et incolores
      |
  Tier 1 : Texture            -- detecte glitter/shimmer/holo
      |
  Tier 3 : Vision Core        -- KMeans K=5, extraction features
      |
  Tier 2 : NLP Prior          -- shade_name -> Lab -> deltaE features
      |
  Tier 4 : ShadeNail Ranking  -- XGBoost regression -> choix cluster
      |
  Couleur predite (RGB/Lab/Hex)
```

> **Note** : Tier 2 vient apres Tier 3 car il a besoin des clusters pour calculer les deltaE entre la couleur NLP et chaque centre de cluster.

## Detail des tiers

### Tier 0 — Gatekeeper (`src/feature_engineering/tier0_gatekeeper.py`)

- **Role** : Detecter les cas speciaux avant tout traitement image
- **Detecte** :
  - **Kits multi-vernis** : images avec plusieurs flacons (mots-cles : "set", "kit", "collection")
  - **Incolores/transparents** : top coat, base coat, gel transparent
- **Sortie** : `is_kit=True` ou `is_clear=True` -> le pipeline s'arrete et renvoie une prediction speciale
- **Impact** : ~5% des produits sont filtres ici

### Tier 1 — Texture (`src/feature_engineering/tier1_texture.py`)

- **Role** : Detecter les textures speciales via analyse de l'image
- **Detecte** : glitter, shimmer, holographique, matte
- **Methode** : Analyse de la variance locale des pixels, detection de points brillants
- **Sortie** : `has_glitter=True`, `texture_type="shimmer"`, etc.
- **Impact** : Information ajoutee aux features, n'arrete pas le pipeline

### Tier 3 — Vision Core (`src/feature_engineering/tier3_vision.py`)

- **Role** : Extraction de couleur par clustering de pixels
- **Methode** :
  1. Suppression du fond avec **rembg** (cache dans `outputs/cache/`)
  2. Conversion en espace couleur **CIE Lab**
  3. **KMeans K=5** sur les pixels Lab
  4. Extraction de **35 features** par cluster :
     - Centre Lab (L, a, b)
     - Proportion du cluster
     - Saturation, luminosite
     - Position verticale moyenne (haut/bas de l'image)
     - Ecart-type intra-cluster

### Tier 2 — NLP Prior (`src/feature_engineering/tier2_nlp_prior.py`)

- **Role** : Utiliser le nom de la teinte comme indice supplementaire
- **Methode** :
  1. Extraction des mots-couleur du `shade_name` (dictionnaire de ~200 couleurs)
  2. Mapping mot -> Lab centroid (ex: "rouge" -> L=53, a=80, b=67)
  3. Calcul du **deltaE** entre le centroid NLP et chaque centre de cluster
  4. Generation de features deltaE pour le Tier 4
- **Impact** : Ameliore la precision de ~3-5% quand le nom est informatif
- **Fallback** : Si pas de shade_name ou nom non reconnu, features NLP mises a zero

### Tier 4 — ShadeNail Ranking (`src/clustering/tier4_ranking.py`)

- **Role** : Choisir le meilleur cluster parmi les 5 candidats
- **Modele** : **XGBoost Regressor** entraine pour predire le deltaE (erreur couleur)
- **Methode** :
  1. Pour chaque cluster, construire un vecteur de **34 features**
  2. XGBoost predit le deltaE attendu pour chaque cluster
  3. Le cluster avec le **deltaE minimum** est choisi
  4. Le centre Lab de ce cluster est la couleur predite
- **34 features par cluster** :
  - L, a, b du centre
  - Proportion, saturation, luminosite
  - Position verticale normalisee
  - Ecart-type intra-cluster
  - Rang par proportion, luminosite, saturation
  - DeltaE vers NLP prior (si disponible)
  - has_nlp flag
- **Performance** : ~81% accuracy (deltaE < 10 = match correct)

## Structure des fichiers source

```
src/
  main.py                          # Point d'entree : python -m src.main
  __init__.py
  clustering/
    __init__.py
    pipeline_main.py               # Orchestrateur (--infer, --train, --evaluate, --full)
    tier4_ranking.py               # ShadeNail : XGBoost ranking inference
    evaluate.py                    # Evaluation sur le dataset labellise
  feature_engineering/
    __init__.py
    tier0_gatekeeper.py            # Detection kits + incolores
    tier1_texture.py               # Detection textures (glitter, shimmer)
    tier2_nlp_prior.py             # NLP : shade_name -> Lab -> deltaE
    tier3_vision.py                # Vision : rembg + KMeans K=5
  utils/
    __init__.py
    config.py                      # Tous les chemins et constantes
    precompute_cache.py            # Cache rembg (precompute + inference)
```

## Flux de donnees

```
CSV d'entree (image_filename, shade_name, ...)
  |
  v
precompute_cache.py  -->  outputs/cache/*.npz  (fond supprime, pixels Lab)
  |
  v
tier0_gatekeeper.py  -->  is_kit? is_clear?  (filtre ~5%)
  |
  v
tier1_texture.py     -->  has_glitter? texture_type?
  |
  v
tier3_vision.py      -->  5 centres Lab + proportions + 35 features
  |
  v
tier2_nlp_prior.py   -->  deltaE NLP features (si shade_name dispo)
  |
  v
tier4_ranking.py     -->  XGBoost predict deltaE x5 -> argmin -> couleur
  |
  v
outputs/predictions/predictions.csv  (RGB, Hex, Lab, cluster, tier, flags)
```

## Modele pre-entraine

Le modele est inclus dans le repo :
- `outputs/models/shadenail_xgb.pkl` — XGBoost Regressor (~2 MB)
- `outputs/models/shadenail_meta.json` — Metadonnees d'entrainement

Entraine sur 1261 images labellisees manuellement. Accuracy test : ~81%.

## Configuration

Tous les chemins sont dans `src/utils/config.py` et sont **relatifs a la racine du projet** (pas de paths absolus). Aucune configuration manuelle n'est necessaire.

Constantes cles :
| Constante | Valeur | Description |
|-----------|--------|-------------|
| `IMAGE_DIR` | `data/raw/images/` | Dossier des images |
| `CACHE_DIR` | `outputs/cache/` | Cache rembg |
| `MODEL_DIR` | `outputs/models/` | Modeles entraines |
| `N_RANKING_FEATURES` | 34 | Features par cluster pour XGBoost |
