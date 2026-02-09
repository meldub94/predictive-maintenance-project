# 🏭 Prédiction de Risque de Défaillance Industrielle

## 📊 Contexte
Projet ML réalisé dans le cadre du **Master en Data Management** à AIVancity.  
Objectif : Prédire les risques de panne d'équipement industriel pour optimiser la maintenance préventive.

## 🎯 Objectifs Business
- ⏱️ Réduire les temps d'arrêt non planifiés
- 🔧 Optimiser la maintenance préventive  
- 💰 Estimer les coûts potentiels de défaillance

## 🏗️ Architecture
```
ML project sprint/
├── data/                   # Données du projet
│   ├── raw/               # Données brutes
│   └── processed/         # Données traitées
├── src/                   # Code source
│   ├── data/             # Pipeline de données
│   ├── features/         # Feature engineering
│   ├── models/           # Modèles ML
│   └── monitoring/       # Suivi performances
├── tests/                # Tests unitaires
└── models/              # Modèles sauvegardés
```

## 🚀 Installation
```bash
# Cloner le projet
git clone https://github.com/meldub94/predictive-maintenance-project.git
cd predictive-maintenance-project

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## 🔧 Utilisation
```bash
# Pipeline complet
python src/data/extract.py
python src/data/clean.py
python src/data/augment.py
python src/features/build_features.py
python src/models/train_model.py
```

## 📦 Technologies

- pandas, numpy - Data processing
- scikit-learn - Machine Learning
- matplotlib, seaborn - Visualisation
- wandb - Experiment tracking
- Docker - Conteneurisation (à venir)

## 👥 Équipe

- Mariame Eldubuni - [Votre rôle]
- [Coéquipier 2] - [Rôle]
- [Coéquipier 3] - [Rôle]
- [Coéquipier 4] - [Rôle]

## 📋 Livrables

- [x] Structure technique
- [x] Configuration Git/GitHub
- [ ] Scripts de pipeline complets
- [ ] Modèles entraînés
- [ ] Documentation complète
- [ ] Board Trello
- [ ] Rapport final

## 📄 Licence

Projet académique - Master Data Management @ AIVancity
