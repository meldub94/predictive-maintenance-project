"""Tests pour le module de feature engineering."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import (
    build_features,
    create_polynomial_features,
    encode_categorical_features,
    reduce_dimensionality,
    prepare_for_ml
)


class TestFeatureFunctions(unittest.TestCase):
    
    def setUp(self):
        """Création de données de test."""
        # DataFrame avec features
        self.test_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='h'),
            'equipment_id': ['EQ001'] * 50 + ['EQ002'] * 50,
            'equipment_type': ['compressor'] * 50 + ['pump'] * 50,
            'temperature': np.random.normal(70, 5, 100),
            'vibration': np.random.normal(1.5, 0.3, 100),
            'pressure': np.random.normal(20, 2, 100),
            'current': np.random.normal(100, 10, 100),
            'failure_soon': [0] * 80 + [1] * 20,
            'next_failure_type': ['none'] * 80 + ['bearing_failure'] * 20
        })
    
    def test_create_polynomial_features(self):
        """Test de la création de features polynomiales."""
        result_df = create_polynomial_features(self.test_df, degree=2)
        
        # Vérifier que de nouvelles colonnes sont créées
        original_cols = len(self.test_df.columns)
        new_cols = len(result_df.columns)
        self.assertGreaterEqual(new_cols, original_cols, 
                               "Des colonnes polynomiales devraient être ajoutées ou égales")
        
        # Vérifier qu'il n'y a pas de NaN ou Inf
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertFalse(result_df[col].isna().any(), 
                           f"{col} ne devrait pas contenir de NaN")
            self.assertFalse(np.isinf(result_df[col]).any(), 
                           f"{col} ne devrait pas contenir d'Inf")
    
    def test_encode_categorical_features(self):
        """Test de l'encodage des features catégorielles."""
        result_df, encoders = encode_categorical_features(
            self.test_df,
            method='label'
        )
        
        # Vérifier que equipment_type est encodé
        if 'equipment_type' in result_df.columns:
            self.assertTrue(
                pd.api.types.is_numeric_dtype(result_df['equipment_type']),
                "equipment_type devrait être numérique après encodage"
            )
        
        # Vérifier que les encodeurs sont retournés
        self.assertIsInstance(encoders, dict, "Les encodeurs devraient être un dictionnaire")
    
    def test_reduce_dimensionality(self):
        """Test de la réduction de dimensionnalité avec PCA."""
        # Créer un DataFrame avec beaucoup de features numériques
        n_features = 50
        large_df = pd.DataFrame(
            np.random.randn(100, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        n_components = 10
        exclude_cols = []
        result_df, pca = reduce_dimensionality(large_df, exclude_cols, n_components=n_components)
        
        # Vérifier que des composantes PCA sont créées
        pca_cols = [col for col in result_df.columns if col.startswith('pca_')]
        self.assertEqual(len(pca_cols), n_components,
                        f"Il devrait y avoir {n_components} composantes PCA")
        
        # Vérifier que le transformateur PCA est retourné
        self.assertIsNotNone(pca, "Le transformateur PCA devrait être retourné")
    
    def test_prepare_for_ml(self):
        """Test de la préparation finale des données pour ML."""
        result_df = prepare_for_ml(self.test_df)
        
        # Vérifier que timestamp et equipment_id sont supprimés
        self.assertNotIn('timestamp', result_df.columns,
                        "timestamp devrait être supprimé")
        self.assertNotIn('equipment_id', result_df.columns,
                        "equipment_id devrait être supprimé")
        
        # Vérifier qu'il n'y a pas de NaN
        self.assertEqual(result_df.isnull().sum().sum(), 0,
                        "Il ne devrait pas y avoir de valeurs manquantes")
        
        # Vérifier qu'il n'y a pas de Inf
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertFalse(np.isinf(result_df[col]).any(),
                           f"La colonne {col} ne devrait pas contenir d'infinis")


if __name__ == '__main__':
    unittest.main()