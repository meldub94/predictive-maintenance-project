"""
Package de prédiction de défaillances industrielles.

Ce module contient les outils pour entraîner, évaluer et utiliser
des modèles de prédiction de défaillances.

Modules:
- train_model: Entraînement multi-modèles
- evaluation: Évaluation et métriques
- predict_model: Prédictions sur nouvelles données
"""

from .train_model import ModelTrainer, train_pipeline
from .evaluation import evaluate_model
from .predict_model import load_model, predict

__all__ = [
    'ModelTrainer',
    'train_pipeline',
    'evaluate_model',
    'load_model',
    'predict'
]

__version__ = '1.0.0'
__author__ = 'Mariame El Dub'
__project__ = 'Maintenance Prédictive Industrielle'

# Ne PAS exécuter automatiquement à l'import !
# Pour utiliser:
# from src.models import train_pipeline, evaluate_model, predict
