# 🏭 Projet de Maintenance Prédictive Industrielle

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Projet de Master Data Management - AIVancity
Prédiction des défaillances industrielles par Machine Learning

---

## 📑 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture du projet](#architecture-du-projet)
- [Pipeline de données](#pipeline-de-données)
- [Feature Engineering](#feature-engineering)
- [Modélisation ML](#modélisation-ml)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Technologies](#technologies)
- [Contributeurs](#contributeurs)

---

## 🎯 Vue d'ensemble

Ce projet vise à **prédire les défaillances d'équipements industriels** en utilisant des techniques de Machine Learning avancées. L'objectif est de permettre une **maintenance prédictive** pour :

- ✅ Réduire les temps d'arrêt non planifiés
- ✅ Optimiser les coûts de maintenance
- ✅ Prolonger la durée de vie des équipements
- ✅ Améliorer la sécurité opérationnelle

### 📊 Données

- **259,205 enregistrements** de capteurs (température, vibration, pression, courant)
- **23 défaillances** documentées sur 5 équipements (EQ001-EQ005)
- **Période** : 01/01/2023 → 30/06/2023 (6 mois)
- **3 types de défaillances** : pressure_loss, bearing_failure, overheating

---

## 🏗️ Architecture du projet

```
ML project sprint/
├── data/
│   ├── raw/                          # Données brutes (non versionnées)
│   └── processed/
│       ├── extracted_data/           # Données extraites (parquet + rapports)
│       ├── cleaned_data/             # Données nettoyées + visualisations
│       ├── augmented_data/           # Features augmentées (138 colonnes)
│       └── features/                 # Features finales pour ML
│           ├── train.parquet         # Dataset d'entraînement (207K lignes)
│           ├── test.parquet          # Dataset de test (51K lignes)
│           ├── artifacts/            # Encodeurs et transformateurs
│           └── visualizations/       # Graphiques de distribution
│
├── src/
│   ├── data/                         # Pipeline de préparation des données
│   │   ├── extract.py               # Extraction et validation
│   │   ├── clean.py                 # Nettoyage et outliers
│   │   ├── augment.py               # Feature engineering avancé
│   │   ├── __init__.py              # Orchestration du pipeline
│   │   └── __main__.py              # Point d'entrée
│   │
│   ├── features/                     # Feature engineering final
│   │   ├── build_features.py        # Construction features ML
│   │   └── __init__.py              # Exports du module
│   │
│   ├── models/                       # Entraînement et prédictions
│   │   ├── train_model.py           # Entraînement multi-modèles
│   │   ├── evaluation.py            # Évaluation et métriques
│   │   └── predict_model.py         # Prédictions sur nouvelles données
│   │
│   └── monitoring/
│       └── wandb_tracking.py        # Suivi des expériences
│
├── models/                           # Modèles entraînés (.pkl)
├── reports/
│   └── evaluation/                   # Rapports et visualisations
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── pr_curve.png
│       └── feature_importance.png
│
├── notebooks/                        # Jupyter notebooks (exploration)
├── tests/                           # Tests unitaires
├── docs/                            # Documentation technique
├── requirements.txt                 # Dépendances Python
├── .gitignore
└── README.md
```

---

## 🔄 Pipeline de données

### 1️⃣ **extract.py** - Extraction

**Rôle** : Charger et valider les données brutes

**Fonctionnalités** :
- Lecture CSV avec parsing optimisé des dates
- Validation de l'intégrité des fichiers
- Conversion des timestamps en datetime
- Sauvegarde en format Parquet (compression snappy)
- Génération d'un rapport d'extraction

**Résultats** :
```
✅ 259,205 enregistrements capteurs extraits
✅ 23 défaillances extraites
✅ 5 équipements (EQ001-EQ005)
✅ Types : compressor, pump, motor
```

**Commande** :
```bash
python src/data/extract.py
```

---

### 2️⃣ **clean.py** - Nettoyage

**Rôle** : Nettoyer et préparer les données pour l'analyse

**Fonctionnalités** :
- Détection et traitement des valeurs infinies
- Gestion des valeurs manquantes (imputation KNN)
- Suppression des doublons
- Détection d'outliers (IQR method)
- Validation de la cohérence des équipements
- Génération de visualisations (distributions)

**Résultats** :
```
✅ 259,205 → 259,205 lignes (100% rétention)
✅ 470 outliers détectés (0.18%)
   - 33 température
   - 388 vibration
   - 49 courant
✅ 0 doublon
✅ 0 valeur manquante
✅ 4 graphiques de distribution créés
```

**Commande** :
```bash
python src/data/clean.py
```

---

### 3️⃣ **augment.py** - Augmentation

**Rôle** : Créer des features avancées pour le ML

**Fonctionnalités** :
- **Features temporelles** (12) : hour, day_of_week, month, quarter, encodage cyclique (sin/cos)
- **Features rolling** (48) : mean, std, min, max sur fenêtres 5/10/30
- **Features lag** (48) : lags 1/3/5/10 + changements absolus et pourcentages
- **Target engineering** (3) : failure_soon, time_to_failure, next_failure_type
- **Health indicators** (3) : days_since_last_failure, failures_count (30/90 jours)
- **Interactions** (8) : produits entre variables + ratios
- **Statistiques** (16) : z-scores et déviations par équipement

**Résultats** :
```
✅ 10 → 148 colonnes (+138 features)
✅ 6,647 échantillons positifs (2.56%)
✅ 142 colonnes scalées (StandardScaler)
✅ 319.52 MB mémoire utilisée
```

**Commande** :
```bash
python src/data/augment.py
```

---

### 4️⃣ **Pipeline complet**

**Exécution des 3 étapes** :
```bash
python -m src.data
```

**Temps d'exécution** :
- Première fois : ~2 min 15s
- Avec skip_existing : <1s

---

## 🎨 Feature Engineering

### **build_features.py** - Préparation ML

**Rôle** : Transformer les features augmentées en dataset prêt pour le ML

**Fonctionnalités** :

1. **Features polynomiales** (degré 2)
   - Capture les relations non-linéaires
   - Appliqué sur : temperature, vibration, pressure, current

2. **Encodage catégorielles** (Label Encoding)
   - equipment_type → entier
   - next_failure_type → entier
   - Sauvegarde des encodeurs pour réutilisation

3. **Features fréquentielles** (optionnel)
   - Analyse FFT sur les vibrations
   - Extraction de features spectrales

4. **Réduction de dimensionnalité** (PCA)
   - 30 composantes principales
   - Explique 92.5% de la variance
   - Sauvegarde du transformateur PCA

5. **Préparation finale**
   - Nettoyage des NaN et inf
   - Suppression des colonnes non-features (timestamp, equipment_id)
   - Split stratifié 80/20 (train/test)

**Résultats** :
```
✅ 180 features finales (148 → 180)
✅ Train: 207,364 lignes (2.56% positifs)
✅ Test: 51,841 lignes (2.56% positifs)
✅ PCA: 30 composantes (92.5% variance)
✅ Encodeurs sauvegardés (joblib)
```

**Commande** :
```bash
python src/features/build_features.py
```

**Fichiers générés** :
```
data/processed/features/
├── train.parquet              # Dataset d'entraînement
├── test.parquet               # Dataset de test
├── train.csv                  # Version CSV
├── test.csv                   # Version CSV
├── features_report.csv        # Rapport détaillé
├── artifacts/
│   ├── encoders.joblib       # LabelEncoders sauvegardés
│   └── pca.joblib            # Transformateur PCA
└── visualizations/
    └── class_distribution.png # Distribution train/test
```

---

### **__init__.py** - Module features

**Rôle** : Exposer les fonctions du module pour import

**Exports** :
```python
from src.features import (
    build_features,              # Pipeline complet
    create_polynomial_features,  # Features polynomiales
    encode_categorical_features, # Encodage catégorielles
    reduce_dimensionality,       # PCA
    prepare_for_ml              # Nettoyage final
)
```

**Usage** :
```python
from src.features import build_features

# Construire les features
df = build_features(
    input_dir="data/processed/augmented_data",
    output_dir="data/processed/features"
)
```

**Note importante** : Le module **n'exécute pas** automatiquement le pipeline à l'import (contrairement à une version précédente). Il faut appeler explicitement `build_features()` ou lancer le script directement.

---

## 🤖 Modélisation ML

### **train_model.py** - Entraînement

**Modèles testés** :
- Random Forest
- Gradient Boosting
- Logistic Regression

**Processus** :
1. Charge train.parquet et test.parquet
2. Entraîne les 3 modèles en parallèle
3. Évalue chaque modèle (Accuracy, Precision, Recall, F1, ROC-AUC)
4. Sélectionne le meilleur modèle (ROC-AUC)
5. Sauvegarde le meilleur avec joblib

**Commande** :
```bash
python src/models/train_model.py
```

---

### **evaluation.py** - Évaluation

**Métriques calculées** :
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Average Precision
- Matrice de confusion
- Rapport de classification détaillé

**Visualisations générées** :
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

**Commande** :
```bash
python src/models/evaluation.py
```

---

### **predict_model.py** - Prédictions

**Fonctionnalités** :
- Charge automatiquement le modèle le plus récent
- Prétraite les données d'entrée
- Génère prédictions + probabilités
- Sauvegarde en CSV

**Commande** :
```bash
python src/models/predict_model.py
```

---

## 📦 Installation

### Prérequis

- Python 3.9+
- pip

### Étapes

1. **Cloner le repository**
```bash
git clone https://github.com/meldub94/predictive-maintenance-project.git
cd predictive-maintenance-project
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Placer les données brutes**
```bash
# Copier vos fichiers CSV dans data/raw/
cp sensor_data.csv data/raw/
cp failure_data.csv data/raw/
```

---

## 🚀 Utilisation

### Pipeline complet

```bash
# 1. Pipeline de données (extract → clean → augment)
python -m src.data

# 2. Feature engineering
python src/features/build_features.py

# 3. Entraînement des modèles
python src/models/train_model.py

# 4. Évaluation
python src/models/evaluation.py

# 5. Prédictions
python src/models/predict_model.py
```

### Utilisation programmatique

```python
from src.data import process_data
from src.features import build_features
from src.models.train_model import train_pipeline

# Pipeline de données
process_data()

# Feature engineering
features_df = build_features()

# Entraînement
model, metrics = train_pipeline()

print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
```

---

## 📊 Résultats

### Performance du modèle

| Métrique | Score |
|----------|-------|
| **Accuracy** | 100% |
| **Precision** | 100% |
| **Recall** | 99.92% |
| **F1-Score** | 99.96% |
| **ROC-AUC** | 1.0000 |

### Distribution des défaillances

| Type | Nombre | Durée moyenne | Coût moyen |
|------|--------|---------------|------------|
| pressure_loss | 8 | 19.3 jours | 7,984€ |
| bearing_failure | 9 | 19.3 jours | 7,984€ |
| overheating | 6 | 19.3 jours | 7,984€ |

### Performance du pipeline

| Étape | Temps | Mémoire |
|-------|-------|---------|
| Extract | ~10s | 50 MB |
| Clean | ~15s | 100 MB |
| Augment | ~1m30s | 320 MB |
| Build Features | ~30s | 400 MB |
| Train | ~30s | 200 MB |
| **Total** | **~2m30s** | **400 MB** |

---

## 🛠️ Technologies utilisées

| Catégorie | Technologies |
|-----------|-------------|
| **Langage** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost (optionnel) |
| **Visualisation** | Matplotlib, Seaborn |
| **Storage** | Parquet (pyarrow), CSV |
| **Serialization** | Joblib, Pickle |
| **Version Control** | Git, GitHub |
| **Environment** | venv |

---

## 👥 Contributeurs

- **Mariame El Dub** - Master Data Management, AIVancity
- GitHub: [@meldub94](https://github.com/meldub94)

---

## 📝 Livrables

### ✅ Complétés

- [x] Pipeline de données (extract, clean, augment)
- [x] Feature engineering avancé (180 features)
- [x] Entraînement multi-modèles
- [x] Évaluation complète avec visualisations
- [x] Module de prédiction fonctionnel
- [x] Documentation (README, docstrings)
- [x] Rapports automatiques (CSV)
- [x] Visualisations (PNG)

### 🔄 En cours

- [ ] Tests unitaires (pytest)
- [ ] Tracking Weights & Biases
- [ ] Conteneurisation Docker

### 📅 À venir

- [ ] API REST (FastAPI)
- [ ] Dashboard interactif (Streamlit)
- [ ] Déploiement cloud (AWS/Azure)
- [ ] CI/CD pipeline

---

## ⚠️ Notes importantes

### Correction des erreurs booléennes NumPy

Le code a été optimisé pour éviter les erreurs futures avec wandb :

```python
# ❌ ÉVITER
mask = (df['col'] < lower) - (df['col'] > upper)  # Soustraction de booléens

# ✅ CORRECT
below = df['col'] < lower
above = df['col'] > upper
mask = below | above  # Combinaison explicite avec |
```

### Gestion des chemins

Utilisation de `pathlib` pour la compatibilité multiplateforme :

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
```

---

## 📞 Contact

Pour toute question ou suggestion :
- Email : mariame.eldub@edu.aivancity.ai
- LinkedIn : [Mariame El Dub](#)

---

## 📄 License

MIT License - Voir LICENSE pour plus de détails

---

**Projet réalisé dans le cadre du Master Data Management - AIVancity 2025-2026**