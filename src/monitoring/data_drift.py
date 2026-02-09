"""Détection de data drift."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional


class DataDriftMonitor:
    """Moniteur de data drift."""

    def __init__(self, reference_data: pd.DataFrame,
                 drift_threshold: float = 0.05,
                 output_dir: str = "drift_reports"):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reference_stats = self._calculate_statistics(reference_data)

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calcule les statistiques."""
        stats = {}
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats[col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max())
                }
        return stats

    def detect_drift(self, new_data: pd.DataFrame) -> Dict:
        """Détecte le drift."""
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'sample_size': len(new_data),
            'features': {},
            'overall_drift_detected': False
        }

        features_with_drift = 0
        total_features = 0

        for col in self.reference_data.columns:
            if col not in new_data.columns:
                continue

            total_features += 1
            
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                ref_vals = self.reference_data[col].dropna()
                new_vals = new_data[col].dropna()

                if len(ref_vals) > 0 and len(new_vals) > 0:
                    ks_stat, p_val = ks_2samp(ref_vals, new_vals)
                    
                    drift_detected = p_val < self.drift_threshold
                    
                    report['features'][col] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_val),
                        'drift_detected': bool(drift_detected)
                    }
                    
                    if drift_detected:
                        features_with_drift += 1

        if total_features > 0:
            report['drift_percentage'] = features_with_drift / total_features
            report['overall_drift_detected'] = report['drift_percentage'] > 0.1

        self._save_report(report)
        return report

    def _save_report(self, report: Dict):
        """Sauvegarde le rapport."""
        ts = report['timestamp'].replace(' ', '_').replace(':', '-')
        filename = self.output_dir / f"drift_report_{ts}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=4)


def compare_datasets(reference_data: pd.DataFrame,
                     current_data: pd.DataFrame,
                     threshold: float = 0.05,
                     output_dir: str = "drift_reports") -> Dict:
    """Compare deux datasets."""
    monitor = DataDriftMonitor(reference_data, threshold, output_dir)
    return monitor.detect_drift(current_data)


__all__ = ['DataDriftMonitor', 'compare_datasets']
