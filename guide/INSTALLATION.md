# Installation — ShadeNail

Guide d'installation du pipeline ShadeNail pour l'extraction automatique de couleur de vernis a ongles.

## Prerequis

| Element | Version |
|---------|---------|
| **Python** | 3.11 (obligatoire) |
| **Git** | recent |
| **Espace disque** | ~500 MB (dependances + modele rembg) |

Verifier Python :
```bash
python --version   # ou python3 --version
# Doit afficher Python 3.11.x
```

## 1. Cloner le projet

```bash
git clone https://github.com/Telecom-Paris-Team-L-oreal-2025-2026/Loreal-Shades-Clustering.git
cd Loreal-Shades-Clustering
git checkout vernis-a-ongle
```

## 2. Installer les dependances

### Option A — pip (recommande, plus simple)

```bash
# Creer un environnement virtuel
python -m venv .venv

# Activer l'environnement
# Linux / Mac :
source .venv/bin/activate
# Windows PowerShell :
.venv\Scripts\Activate.ps1
# Windows CMD :
.venv\Scripts\activate.bat

# Installer
pip install -r requirements.txt
```

### Option B — Poetry

```bash
# Installer Poetry si necessaire
curl -sSL https://install.python-poetry.org | python3 - --version 2.2.0

# Installer les dependances
poetry install --without dev

# Activer l'environnement
eval $(poetry env activate)
# Windows : executer la commande affichee par poetry env activate
```

## 3. Verifier l'installation

```bash
python -m src.main --help
```

Resultat attendu :
```
Pipeline ShadeNail
options:
  --infer        Mode inference ShadeNail
  --input INPUT  CSV d'entree pour --infer
  ...
```

## 4. Premier lancement — telechargement rembg

Au premier lancement, `rembg` telecharge automatiquement le modele de suppression de fond U2-Net (~176 MB). C'est normal, ca ne se fait qu'une seule fois.

Test rapide :
```bash
python -c "from rembg import remove; print('rembg OK')"
```

## Dependances principales

| Package | Role |
|---------|------|
| `xgboost` | Modele ShadeNail (regression deltaE) |
| `scikit-learn` | KMeans clustering (K=5) |
| `scikit-image` | Conversion couleur RGB / Lab |
| `rembg` | Suppression automatique du fond image |
| `pandas` | Manipulation des donnees CSV |
| `pillow` | Chargement des images |
| `tqdm` | Barres de progression |

## Troubleshooting

### `rembg` ne s'installe pas

```bash
pip install onnxruntime   # installer onnxruntime d'abord
pip install rembg          # puis rembg
```

### Erreur `ModuleNotFoundError: No module named 'src'`

Toujours lancer les commandes **depuis la racine du projet** :
```bash
cd Loreal-Shades-Clustering
python -m src.main --infer --input data/raw/mon_fichier.csv
```

### Windows : erreur d'encodage UTF-8

```powershell
$env:PYTHONIOENCODING="utf-8"
python -m src.main --infer --input data/raw/mon_fichier.csv
```

---

**Suite : [GETTING_STARTED.md](GETTING_STARTED.md) pour lancer votre premiere inference.**
