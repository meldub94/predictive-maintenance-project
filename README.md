# 🏭 Prédiction de Risque de Défaillance Industrielle

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/Status-En%20développement-yellow.svg)
![License](https://img.shields.io/badge/License-Academic-green.svg)

Projet de Machine Learning réalisé dans le cadre du **Master en Data Management** à AIVancity.

**Objectif** : Prédire les risques de panne d'équipement industriel pour optimiser la maintenance préventive et réduire les coûts.

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Objectifs Business](#-objectifs-business)
- [Architecture du Projet](#-architecture-du-projet)
- [Pipeline de Données](#-pipeline-de-données)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Technologies](#-technologies)
- [Équipe](#-équipe)
- [Livrables](#-livrables)

---

## 🎯 Vue d'ensemble

Ce projet implémente un système de **maintenance prédictive** utilisant des techniques de Machine Learning pour anticiper les défaillances d'équipements industriels.

### Données

- **259,205** enregistrements de capteurs (6 mois de données)
- **23** défaillances documentées
- **5** équipements surveillés (compresseurs)
- **4** types de capteurs : température, vibration, pression, courant

### Types de défaillances détectées

- `pressure_loss` : Perte de pression
- `bearing_failure` : Défaillance de roulement
- `overheating` : Surchauffe

---

## 💼 Objectifs Business

1. **⏱️ Réduire les temps d'arrêt** non planifiés de 30%
2. **🔧 Optimiser la maintenance préventive** en anticipant les pannes
3. **💰 Estimer les coûts potentiels** de défaillance (moyenne : 7,984€)
4. **📊 Fournir des insights actionnables** aux équipes de maintenance

---

## 🏗️ Architecture du Projet

```
ML project sprint/
├── data/
│   ├── raw/                              # Données brutes (non versionnées)
│   │   ├── predictive_maintenance_sensor_data.csv
│   │   └── predictive_maintenance_failure_logs.csv
│   ├── processed/                        # Données traitées
│   │   ├── extracted_data/              # Étape 1: Extraction
│   │   ├── cleaned_data/                # Étape 2: Nettoyage
│   │   └── augmented_data/              # Étape 3: Augmentation
│   └── validation/                       # Données de validation
│
├── src/
│   ├── data/                            # 📊 Pipeline de données
│   │   ├── __init__.py                  # Orchestration du pipeline
│   │   ├── __main__.py                  # Point d'entrée (python -m src.data)
│   │   ├── extract.py                   # Extraction des données
│   │   ├── clean.py                     # Nettoyage et détection d'outliers
│   │   └── augment.py                   # Feature engineering
│   │
│   ├── features/                        # 🔧 Feature engineering avancé
│   │   ├── __init__.py
│   │   └── build_features.py
│   │
│   ├── models/                          # 🤖 Modèles ML
│   │   ├── __init__.py
│   │   ├── train_model.py               # Entraînement
│   │   ├── predict_model.py             # Prédictions
│   │   └── evaluation.py                # Évaluation des performances
│   │
│   └── monitoring/                      # 📈 Suivi et monitoring
│       ├── __init__.py
│       ├── data_drift.py
│       ├── performance_tracking.py
│       └── wandb_tracking.py
│
├── notebooks/                           # 📓 Notebooks d'exploration
├── tests/                               # 🧪 Tests unitaires
├── models/                              # 💾 Modèles sauvegardés
├── docs/                                # 📚 Documentation
│
├── .gitignore                           # Fichiers à ignorer par Git
├── requirements.txt                     # Dépendances Python
└── README.md                            # Ce fichier
```

---

## 🔄 Pipeline de Données

Notre pipeline de traitement des données est divisé en **3 étapes principales** :

### 1️⃣ Extraction (`extract.py`)

**Objectif** : Charger et valider les données brutes

- ✅ Lecture des fichiers CSV (capteurs et défaillances)
- ✅ Conversion des timestamps en datetime
- ✅ Validation de la cohérence des données
- ✅ Sauvegarde en format Parquet (optimisé)
- ✅ Génération d'un rapport d'extraction

**Résultats** :
- 259,205 enregistrements capteurs extraits
- 23 défaillances identifiées
- 5 équipements uniques
- Période : 01/01/2023 → 30/06/2023

### 2️⃣ Nettoyage (`clean.py`)

**Objectif** : Nettoyer et préparer les données

**Opérations** :
- ✅ Suppression des valeurs infinies et NaN
- ✅ Détection des outliers (méthode IQR)
  - 33 outliers en température (0.01%)
  - 388 outliers en vibration (0.15%)
  - 0 outliers en pression
  - 49 outliers en courant (0.02%)
- ✅ Suppression des doublons
- ✅ Validation de la cohérence entre capteurs et défaillances
- ✅ Génération de visualisations (distributions)

**Résultats** :
- **100% de rétention** des données (excellente qualité)
- 470 outliers détectés et marqués
- 4 graphiques de distribution créés

### 3️⃣ Augmentation (`augment.py`)

**Objectif** : Créer des features pour le Machine Learning

**Features créées** (138 nouvelles colonnes) :

| Catégorie | Nombre | Description |
|-----------|--------|-------------|
| **Temporelles** | 12 | hour, day_of_week, month, is_night, is_weekend, sin/cos cycliques |
| **Rolling** | 48 | Moyennes, std, min, max sur fenêtres 5, 10, 30 |
| **Lag** | 48 | Valeurs passées (lags 1, 3, 5, 10) + variations |
| **Target** | 3 | `failure_soon`, `time_to_failure`, `next_failure_type` |
| **Health** | 3 | Jours depuis dernière panne, compteurs 30/90j |
| **Interactions** | 8 | Produits et ratios entre variables |
| **Statistiques** | 16 | Z-scores et déviations par équipement |

**Résultats** :
- **10 → 148 colonnes** (+138 features)
- **6,647 échantillons positifs (2.56%)** - Bon équilibre
- Toutes les features **standardisées** (mean=0, std=1)

---

## 📦 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de packages)
- Git

### Étapes d'installation

```bash
# 1. Cloner le repository
git clone https://github.com/meldub94/predictive-maintenance-project.git
cd predictive-maintenance-project

# 2. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Sur Mac/Linux
# ou
venv\Scripts\activate  # Sur Windows

# 3. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

### Dépendances principales

```txt
# Data manipulation
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0

# Machine Learning
scikit-learn>=1.3.0

# Visualisation
matplotlib>=3.7.0
seaborn>=0.12.0

# Experiment tracking
wandb>=0.15.0

# Testing
pytest>=7.4.0

# Utilities
python-dotenv>=1.0.0
```

---

## 🚀 Utilisation

### Option 1 : Pipeline complet

Exécuter tout le pipeline de traitement des données :

```bash
# Méthode 1 : Via module
python -m src.data

# Méthode 2 : Via script
python test_pipeline.py
```

### Option 2 : Étapes individuelles

Exécuter chaque étape séparément :

```bash
# Étape 1 : Extraction
python src/data/extract.py

# Étape 2 : Nettoyage
python src/data/clean.py

# Étape 3 : Augmentation
python src/data/augment.py
```

### Option 3 : Import dans votre code

```python
from src.data import process_data

# Exécuter le pipeline complet
processed_data = process_data(
    sensor_file_path="data/raw/predictive_maintenance_sensor_data.csv",
    failure_file_path="data/raw/predictive_maintenance_failure_logs.csv",
    output_dir="data/processed",
    skip_existing=True,  # Skip les étapes déjà complétées
    verbose=True
)

print(f"Dataset prêt : {len(processed_data):,} lignes × {len(processed_data.columns)} colonnes")
```

### Exécution rapide avec skip_existing

Si vous avez déjà exécuté le pipeline une fois :

```python
# Ne ré-exécute que les étapes nécessaires
df = process_data(
    sensor_file_path="...",
    failure_file_path="...",
    skip_existing=True  # ← Très rapide (< 1 seconde)
)
```

---

## 📊 Résultats

### Statistiques du Dataset Final

| Métrique | Valeur |
|----------|--------|
| **Lignes** | 259,205 |
| **Colonnes** | 148 |
| **Features originales** | 10 |
| **Features créées** | 138 |
| **Taille mémoire** | 319.52 MB |
| **Échantillons positifs** | 6,647 (2.56%) |
| **Taux de rétention** | 100% |

### Distribution des Défaillances

| Type de défaillance | Nombre | Durée réparation (moy.) | Coût (moy.) |
|---------------------|--------|-------------------------|-------------|
| `pressure_loss` | 8 | 17.5 jours | 8,420€ |
| `bearing_failure` | 9 | 18.6 jours | 9,156€ |
| `overheating` | 6 | 22.3 jours | 5,789€ |

### Qualité des Données

- ✅ **Aucune valeur manquante** après nettoyage
- ✅ **Aucun doublon** détecté
- ✅ **470 outliers** (0.18%) détectés et marqués
- ✅ **100% de cohérence** entre capteurs et défaillances

---

## 🛠️ Technologies

### Data Processing
- **pandas** : Manipulation de données tabulaires
- **numpy** : Calculs numériques
- **pyarrow** : Format Parquet (stockage optimisé)

### Machine Learning
- **scikit-learn** : Algorithmes ML, preprocessing
- **StandardScaler / MinMaxScaler** : Normalisation des features

### Visualisation
- **matplotlib** : Graphiques de base
- **seaborn** : Visualisations statistiques avancées

### Development
- **pathlib** : Gestion des chemins (cross-platform)
- **logging** : Traçabilité et débogage
- **pytest** : Tests unitaires

### Experiment Tracking
- **Weights & Biases (wandb)** : Suivi des expériences ML

### Containerization (à venir)
- **Docker** : Conteneurisation de l'application

---

## 👥 Équipe

- **Mariame Eldubuni** - Data Engineer / ML Engineer
- **[Coéquipier 2]** - [Rôle]
- **[Coéquipier 3]** - [Rôle]
- **[Coéquipier 4]** - [Rôle]

### Répartition des rôles (exemple)

| Rôle | Responsabilités |
|------|-----------------|
| **Data Engineer** | Pipeline de données, nettoyage, feature engineering |
| **ML Engineer** | Entraînement des modèles, optimisation des hyperparamètres |
| **DevOps** | Docker, CI/CD, déploiement |
| **Data Scientist** | Analyse exploratoire, sélection des features, évaluation |

---

## 📋 Livrables

### ✅ Complétés

- [x] Structure technique du projet
- [x] Configuration Git & GitHub
- [x] Environnement virtuel Python
- [x] Pipeline de données complet (extract, clean, augment)
- [x] Logs de traçabilité
- [x] Scripts modulaires et réutilisables
- [x] Visualisations (distributions, feature importances)
- [x] Rapports automatiques (extraction, nettoyage, augmentation)
- [x] Documentation (README, docstrings)

### 🔄 En cours

- [ ] Feature selection et dimensionality reduction
- [ ] Split train/validation/test
- [ ] Entraînement des modèles ML
- [ ] Évaluation des performances
- [ ] Tracking avec W&B
- [ ] Board Trello

### ⏳ À venir

- [ ] Containerisation Docker
- [ ] Interface de monitoring
- [ ] API de prédiction
- [ ] Rapport final de projet
- [ ] Présentation

---

## 🔒 Considérations Éthiques

Ce projet prend en compte les aspects éthiques suivants :

### Anonymisation des données
- ✅ Aucune donnée personnelle dans le dataset
- ✅ Identifiants d'équipements génériques (`EQ001`, `EQ002`, etc.)

### Transparence algorithmique
- ✅ Code source ouvert (pour l'équipe académique)
- ✅ Documentation complète des transformations
- ✅ Traçabilité via logs

### Impact sur l'emploi
- ⚠️ La maintenance prédictive **assiste** les techniciens, ne les remplace pas
- ✅ Objectif : **améliorer les conditions de travail** en réduisant les interventions d'urgence

### Biais et équité
- ✅ Validation de la représentativité des données
- ✅ Analyse des outliers (pas de suppression systématique)
- ⏳ Tests sur différents types d'équipements (à venir)

---

## 📈 Performance du Pipeline

### Temps d'exécution (MacBook Air M1, 8GB RAM)

| Étape | Première exécution | Avec `skip_existing` |
|-------|-------------------|----------------------|
| **Extraction** | ~1 seconde | < 0.1 seconde |
| **Nettoyage** | ~10 secondes | < 0.1 seconde |
| **Augmentation** | ~2 minutes | < 0.5 seconde |
| **Pipeline complet** | ~2 min 15s | < 1 seconde |

### Utilisation mémoire

- **Données brutes** : ~30 MB (CSV)
- **Données nettoyées** : ~10 MB (Parquet)
- **Données augmentées** : ~320 MB (148 colonnes)

---

## 🐛 Problèmes Connus & Solutions

### 1. Erreur booléenne NumPy (anticipée et corrigée)

**Erreur potentielle** :
```
numpy boolean subtract, the `-` operator, is not supported
```

**Solution appliquée** :
- Utilisation de `|` (OR) au lieu de `-` (soustraction) pour les masques booléens
- Séparation explicite des comparaisons avant combinaison

### 2. Performance sur gros datasets

**Problème** : `create_component_health_features()` peut être lent (itère sur chaque ligne)

**Solution** :
- ✅ Optimisations vectorielles appliquées où possible
- 🔄 Refactoring pour utiliser `merge_asof` (à venir)

---

## 🚦 Prochaines Étapes

1. **Feature Selection** : Sélectionner les meilleures features (importance, corrélation)
2. **Train/Test Split** : Séparer les données avec stratification temporelle
3. **Model Training** : Tester plusieurs algorithmes (Random Forest, XGBoost, etc.)
4. **Hyperparameter Tuning** : Optimiser les hyperparamètres
5. **Evaluation** : Métriques (Precision, Recall, F1, ROC-AUC)
6. **Monitoring** : Tracking avec W&B, détection de drift
7. **Deployment** : API REST avec FastAPI, containerisation Docker

---

## 📞 Contact & Support

- **GitHub Issues** : [https://github.com/meldub94/predictive-maintenance-project/issues](https://github.com/meldub94/predictive-maintenance-project/issues)
- **Email** : [votre-email@aivancity.ai]
- **Trello Board** : [Lien vers votre board Trello]

---

## 📄 Licence

Projet académique - Master Data Management @ AIVancity  
© 2026 - Tous droits réservés

---

## 🙏 Remerciements

- **AIVancity** pour le cadre académique
- **Professeurs** pour leurs conseils et expertise
- **Anthropic Claude** pour l'assistance au développement

---

<div align="center">

**⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile ! ⭐**

Made with ❤️ by the AIVancity Data Management Team

</div>
