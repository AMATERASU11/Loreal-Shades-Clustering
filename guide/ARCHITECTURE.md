# Architecture du Projet

## Vue d'ensemble

Ce projet suit une architecture modulaire basée sur le workflow de Data Science :

```
Loreal-Shades-Clustering/
├── notebooks/              # Exploration et expérimentation
├── src/                    # Code de production
│   ├── cleaning/          # Nettoyage et préprocessing
│   ├── feature_engineering/ # Création de features
│   ├── clustering/        # Modèles de clustering
│   ├── utils/             # Utilitaires génériques
│   └── main.py            # Point d'entrée principal
├── data/                  # Données (raw, processed, external)
├── outputs/               # Sorties du pipeline (models, reports, etc)
├── tests/                 # Tests unitaires et d'intégration
└── docs/                  # Documentation
```

## Flux de travail

1. **Cleaning** (`src/cleaning/`) - Nettoyage et normalisation
2. **Feature Engineering** (`src/feature_engineering/`) - Création de features
3. **Clustering** (`src/clustering/`) - Modèles et algorithmes
4. **Utils** (`src/utils/`) - Fonctions utilitaires (I/O, visualisations, etc)

## Exécution

```bash
python src/main.py
```

## Organisation des données

- `data/raw/` - Données brutes importées
- `data/processed/` - Données nettoyées et préparées
- `data/external/` - Données externes supplémentaires

## Sorties

- `outputs/models/` - Modèles entraînés (.pkl, .h5, .json)
- `outputs/reports/` - Résultats et métriques (.csv, .json)
- `outputs/visualizations/` - Graphiques générés (.png, .html)
- `outputs/predictions/` - Prédictions finales
- `outputs/logs/` - Logs d'exécution


## Exécution du pipeline

```bash
# Setup initial (une fois)
bash setup.sh

# Exécuter le pipeline complet
python src/main.py

# Lancer les tests
pytest tests/
```

## Utilisation en développement

- **Exploration** : Utilisez les notebooks dans `notebooks/` pour explorer et tester
- **Production** : Tous les codes réutilisables doivent être dans `src/`
- **Tests** : Écrivez des tests dans `tests/` pour les fonctions critiques

## Principes

✅ **Modularité** - Chaque étape est indépendante et testable  
✅ **Reproductibilité** - Tout est versionné (sauf data/outputs)  
✅ **Documentation** - Code clair et commenté  
✅ **Testabilité** - Tests unitaires pour les fonctions clés  
✅ **Production-ready** - Code prêt pour déploiement
