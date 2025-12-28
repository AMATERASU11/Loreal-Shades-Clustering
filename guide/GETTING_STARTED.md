# Premiers pas - Guide d'installation et contribution

## 1️⃣ Cloner le repository

```bash
# Clonez le projet
git clone https://github.com/your-org/Loreal-Shades-Clustering.git
cd Loreal-Shades-Clustering
```

## 2️⃣ Setup de l'environnement

### Option A : Avec le script (recommandé)

```bash
bash setup.sh
```

Ce script :
- ✅ Vérifie Python 3.11
- ✅ Vérifie Poetry 
- ✅ Configure l'environnement virtuel Poetry

### Option B : Manuel

```bash
# Installer les dépendances avec Poetry
poetry install

# Activer l'environnement
poetry shell
```

## 3️⃣ Créer votre branche de travail

### Nomenclature des branches

```
feature/nom-feature         # Nouvelle fonctionnalité
bugfix/nom-bug              # Correction de bug (branche pour la correction des bugs)
docs/nom-documentation      # Documentation (branche propre pour la documentation)
```

### Créer la branche

```bash
# Créer ET basculer sur la nouvelle branche
git checkout -b feature/votre-nom

# Ou si elle existe déjà
git checkout feature/votre-nom
```

## 4️⃣ Vérifier l'installation

```bash
# Vérifier que tout fonctionne
python src/main.py

# Lancer les tests (optionnel)
pytest tests/
```

## 5️⃣ Travailler sur votre branche

### Faire des modifications

```bash
# Voir les changements
git status

# Ajouter les fichiers
git add .

# Commit avec message clair
git commit -m "feat: description claire de ce que vous faites"

# Envoyer sur la branche distante
git push origin feature/votre-nom
```

### Messages de commit (Convention)

```
feat:     nouvelle fonctionnalité
fix:      correction de bug
docs:     documentation
refactor: restructuration du code
test:     ajout de tests
```

**Exemple** :
```bash
git commit -m "feat: ajoute preprocessing des nuances de couleur"
git commit -m "fix: corrige bug dans le clustering KMeans"
git commit -m "docs: met à jour ARCHITECTURE.md"
```

## 6️⃣ Pull Request et Merge

### Avant de faire un PR

1. **Assurez-vous que tout fonctionne** :
   ```bash
   python src/main.py
   pytest tests/
   ```

2. **Récupérez les derniers changements** :
   ```bash
   git fetch origin
   git rebase origin/main
   ```

3. **Pushez votre branche** :
   ```bash
   git push origin feature/votre-nom
   ```

### Créer une Pull Request

- Allez sur GitHub
- Cliquez sur "Compare & pull request"
- Décrivez vos changements
- Demandez une review

---

## Commandes utiles

```bash
# Voir toutes les branches
git branch -a

# Voir l'historique des commits
git log --oneline

# Voir les changements en attente
git status

# Annuler un changement local
git restore fichier.py

# Annuler un commit (garde les changements)
git reset --soft HEAD~1

# Voir les différences
git diff
```

## 📁 Structure du projet

Voir [ARCHITECTURE.md](ARCHITECTURE.md) pour la structure complète.

---

##  Besoin d'aide ?

- Consultez [ARCHITECTURE.md](ARCHITECTURE.md)
- Ouvrez une issue sur GitHub

**Bonne contribution !** 
