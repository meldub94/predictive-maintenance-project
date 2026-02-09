"""
Module de construction de caractéristiques avancées.

Fonctions principales:
- build_features: Construit les features pour le ML
- create_polynomial_features: Features polynomiales
- encode_categorical_features: Encodage catégorielles
"""

from .build_features import (
    build_features,
    create_polynomial_features,
    encode_categorical_features,
    reduce_dimensionality,
    prepare_for_ml
)

__all__ = [
    'build_features',
    'create_polynomial_features',
    'encode_categorical_features',
    'reduce_dimensionality',
    'prepare_for_ml'
]

__version__ = '0.1.0'

# Ne PAS exécuter automatiquement à l'import !
# Pour lancer: python src/features/build_features.py
