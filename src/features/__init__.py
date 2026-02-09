"""
Module de construction de caractéristiques avancées.

Fonctions principales:
- build_features: Construit les features pour le ML
- create_polynomial_features: Features polynomiales
- encode_categorical_features: Encodage catégorielles
- create_frequency_features: Features fréquentielles
- reduce_dimensionality: Réduction PCA
"""

from .build_features import (
    build_features,
    create_polynomial_features,
    encode_categorical_features,
    create_frequency_features,
    reduce_dimensionality,
    prepare_for_ml
)

__all__ = [
    'build_features',
    'create_polynomial_features',
    'encode_categorical_features',
    'create_frequency_features',
    'reduce_dimensionality',
    'prepare_for_ml'
]

__version__ = '0.1.0'


# Ne PAS exécuter automatiquement !
# Pour lancer : python src/features/build_features.py
# Ou : python -m src.features (si vous créez __main__.py)
