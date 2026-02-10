"""Tests pour le pipeline de données."""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.clean import clean_data, detect_outliers
from src.data.augment import augment_data, create_time_features, create_rolling_features, create_lag_features


class TestDataFunctions(unittest.TestCase):
    
    def setUp(self):
        """Création de données de test."""
        self.test_sensor_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='h'),
            'equipment_id': ['EQ001'] * 100,
            'temperature': np.random.normal(70, 5, 100),
            'vibration': np.random.normal(1.5, 0.3, 100),
            'pressure': np.random.normal(20, 2, 100),
            'current': np.random.normal(100, 10, 100),
            'equipment_type': ['compressor'] * 100
        })
    
    def test_detect_outliers(self):
        """Test de la détection d'outliers."""
        data_with_outliers = self.test_sensor_df.copy()
        data_with_outliers.loc[5, 'temperature'] = 150
        
        # Signature réelle: column (singulier)
        outliers = detect_outliers(data_with_outliers, column='temperature')
        
        self.assertIsInstance(outliers, pd.Series)
        self.assertGreater(len(outliers), 0)
    
    def test_create_time_features(self):
        """Test de la création de features temporelles."""
        result_df = create_time_features(self.test_sensor_df)
        
        time_cols = ['hour', 'day_of_week', 'month', 'quarter']
        created_cols = [col for col in time_cols if col in result_df.columns]
        
        self.assertGreater(len(created_cols), 0)
        self.assertIn('temperature', result_df.columns)
    
    def test_create_rolling_features(self):
        """Test des statistiques sur fenêtre glissante."""
        # Signature réelle: window_sizes
        result_df = create_rolling_features(
            self.test_sensor_df,
            window_sizes=[5, 10]
        )
        
        rolling_cols = [col for col in result_df.columns if 'rolling' in col.lower()]
        self.assertGreater(len(rolling_cols), 0)
    
    def test_create_lag_features(self):
        """Test de la création de features avec décalage temporel."""
        # Signature réelle: lag_periods
        result_df = create_lag_features(
            self.test_sensor_df,
            lag_periods=[1, 3, 5]
        )
        
        lag_cols = [col for col in result_df.columns if 'lag' in col.lower()]
        self.assertGreater(len(lag_cols), 0)
        self.assertEqual(len(result_df), len(self.test_sensor_df))


if __name__ == '__main__':
    unittest.main()
