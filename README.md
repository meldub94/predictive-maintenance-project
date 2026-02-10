# 🏭 Projet de Maintenance Prédictive Industrielle

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![Tests](https://img.shields.io/badge/Tests-12%2F12-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Projet de Master Data Management - AIVancity 2025-2026**  
Prédiction des défaillances industrielles par Machine Learning avec optimisation Optuna

---

## 📑 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Architecture du projet](#-architecture-du-projet)
- [Pipeline de données](#-pipeline-de-données)
- [Feature Engineering](#-feature-engineering)
- [Modélisation ML](#-modélisation-ml)
- [Tests unitaires](#-tests-unitaires)
- [Monitoring](#-monitoring)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Technologies](#-technologies)

---

## 🎯 Vue d'ensemble

Ce projet vise à **prédire les défaillances d'équipements industriels** 24h à l'avance en utilisant des techniques de Machine Learning avancées. L'objectif est de permettre une **maintenance prédictive** pour :

- ✅ Réduire les temps d'arrêt non planifiés
- ✅ Optimiser les coûts de maintenance (économie de 30-40%)
- ✅ Prolonger la durée de vie des équipements
- ✅ Améliorer la sécurité opérationnelle

### 📊 Dataset

- **259,205 enregistrements** de capteurs (température, vibration, pression, courant)
- **23 défaillances** documentées sur 5 équipements (EQ001-EQ005)
- **Période** : 01/01/2023 → 30/06/2023 (6 mois)
- **3 types de défaillances** : pressure_loss (35%), bearing_failure (39%), overheating (26%)

### 🎯 Objectif ML

Prédire `failure_soon` (défaillance dans les prochaines 24h) avec :
- **Target** : Variable binaire (0/1)
- **Classe positive** : 2.56% (déséquilibre traité)
- **Horizon** : 24 heures avant la défaillance

---

## 🚀 Utilisation Rapide

```bash
# 1. Pipeline de données
python -m src.data

# 2. Feature engineering
python src/features/build_features.py

# 3. Entraînement (CHOISIR)
python src/models/train_model.py              # Rapide (30s)
python src/models/train_model_optuna.py       # Optimisé (15min)

# 4. Évaluation
python src/models/evaluation.py

# 5. Prédictions
python src/models/predict_model.py

# 6. Tests
pytest tests/ -v
```

---

## 📊 Résultats

### Performance des Modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| **Random Forest** 🥇 | 100% | 100% | 99.92% | 99.96% | **1.0000** |
| Gradient Boosting | 100% | 100% | 100% | 100% | 1.0000 |
| Logistic Regression | 99.98% | 99.92% | 99.47% | 99.70% | 1.0000 |

### Tests Unitaires

```
✅ 12/12 tests passent (100%)
✅ test_data.py : 4/4
✅ test_features.py : 4/4  
✅ test_models.py : 4/4
```

---

Pour la **documentation complète** (70+ pages), voir les sections détaillées ci-dessous.

---

## 📦 Installation

```bash
# 1. Cloner
git clone https://github.com/meldub94/predictive-maintenance-project.git
cd predictive-maintenance-project

# 2. Environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 3. Dépendances
pip install -r requirements.txt

# 4. Données
cp sensor_data.csv data/raw/
cp failure_data.csv data/raw/
```

---

## 🛠️ Technologies

| Catégorie | Stack |
|-----------|-------|
| **ML** | Scikit-learn, Optuna ⭐ |
| **Data** | Pandas, NumPy, Parquet |
| **Viz** | Matplotlib, Seaborn, Plotly |
| **Tests** | pytest, pytest-cov ✅ |
| **Tools** | Joblib, pathlib, logging |

---

## 👥 Auteur

**Mariame El Dub**  
Master Data Management - AIVancity 2025-2026  
📧 mariame.eldub@edu.aivancity.ai  
🐙 [@meldub94](https://github.com/meldub94)

---

## 📄 License

MIT License - Copyright (c) 2026 Mariame El Dub

---

**⭐ N'hésitez pas à donner une étoile sur GitHub si ce projet vous a été utile !**
