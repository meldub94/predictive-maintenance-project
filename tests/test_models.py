"""Tests pour les modèles de prédiction."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock
import sys

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train_model import ModelTrainer
from src.models.predict_model import load_model


class TestModelFunctions(unittest.TestCase):
    
    def setUp(self):
        """Création de données de test."""
        np.random.seed(42)
        n_samples = 100
        
        # Features
        self.X_train = pd.DataFrame({
            'temperature': np.random.normal(70, 5, n_samples),
            'vibration': np.random.normal(1.5, 0.3, n_samples),
            'pressure': np.random.normal(20, 2, n_samples),
            'current': np.random.normal(100, 10, n_samples),
            'pca_0': np.random.randn(n_samples),
            'pca_1': np.random.randn(n_samples)
        })
        
        # Target
        self.y_train = pd.Series(np.random.randint(0, 2, n_samples))
        
        # Test
        self.X_test = pd.DataFrame({
            'temperature': np.random.normal(70, 5, 30),
            'vibration': np.random.normal(1.5, 0.3, 30),
            'pressure': np.random.normal(20, 2, 30),
            'current': np.random.normal(100, 10, 30),
            'pca_0': np.random.randn(30),
            'pca_1': np.random.randn(30)
        })
        
        self.y_test = pd.Series(np.random.randint(0, 2, 30))
        
        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.fit.return_value = self.mock_model
        self.mock_model.predict.return_value = np.random.randint(0, 2, 30)
        self.mock_model.predict_proba.return_value = np.random.rand(30, 2)
    
    def test_model_trainer_init(self):
        """Test de l'initialisation du ModelTrainer."""
        trainer = ModelTrainer(models_dir="test_models", random_state=42)
        
        # Vérifications
        self.assertIsNotNone(trainer, "ModelTrainer devrait être créé")
        self.assertEqual(trainer.random_state, 42)
        self.assertIsInstance(trainer.models_dir, Path)
    
    def test_train_models(self):
        """Test de l'entraînement des modèles."""
        # Simuler un entraînement
        trained = {}
        self.mock_model.fit(self.X_train, self.y_train)
        trained['random_forest'] = self.mock_model
        
        # Vérifications
        self.mock_model.fit.assert_called_once()
        self.assertIn('random_forest', trained)
    
    def test_evaluate_model(self):
        """Test de l'évaluation."""
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        y_pred = self.mock_model.predict(self.X_test)
        y_prob = self.mock_model.predict_proba(self.X_test)[:, 1]
        
        # Calculer métriques
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Vérifications
        self.assertIsInstance(accuracy, (int, float))
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
    
    def test_predict(self):
        """Test des prédictions."""
        predictions = self.mock_model.predict(self.X_test)
        probabilities = self.mock_model.predict_proba(self.X_test)
        
        # Vérifications
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertEqual(len(probabilities), len(self.X_test))
        self.assertEqual(probabilities.shape[1], 2, "Devrait avoir 2 colonnes de probabilités")
        
        # Vérifier appels
        self.mock_model.predict.assert_called()
        self.mock_model.predict_proba.assert_called()


if __name__ == '__main__':
    unittest.main()