"""
Suivi des performances des modèles de prédiction.

Permet de monitorer l'évolution des métriques au fil du temps
et de détecter les dégradations de performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional


class ModelPerformanceTracker:
    """Suivi des performances des modèles."""

    def __init__(
        self,
        model_name: str,
        model_version: str,
        output_dir: str = "performance_reports",
        baseline_metrics: Optional[Dict] = None,
        alert_threshold: float = 0.1
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.output_dir = Path(output_dir)
        self.baseline_metrics = baseline_metrics
        self.alert_threshold = alert_threshold
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self._load_history()

    def _load_history(self):
        """Charge l'historique."""
        history_file = self.output_dir / f"{self.model_name}_{self.model_version}_history.json"
        
        if history_file.exists():
            with open(history_file, "r") as f:
                self.metrics_history = json.load(f)

    def _save_history(self):
        """Sauvegarde l'historique."""
        history_file = self.output_dir / f"{self.model_name}_{self.model_version}_history.json"
        
        with open(history_file, "w") as f:
            json.dump(self.metrics_history, f, indent=4)

    def calculate_metrics(self, y_true, y_pred, y_prob=None) -> Dict:
        """Calcule les métriques."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_prob is not None and len(np.unique(y_true)) == 2:
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

        return metrics

    def track_performance(self, y_true, y_pred, y_prob=None, dataset_name="validation") -> Dict:
        """Enregistre les performances."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        report = {
            "timestamp": timestamp,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dataset": dataset_name,
            "sample_size": len(y_true),
            "metrics": metrics
        }
        
        if self.baseline_metrics:
            report["degradation"] = self._check_degradation(metrics)
        
        self.metrics_history.append(report)
        self._save_history()
        
        return report

    def _check_degradation(self, current_metrics: Dict) -> Dict:
        """Vérifie la dégradation."""
        report = {"has_degradation": False, "metrics_diff": {}}

        for metric, baseline_value in self.baseline_metrics.items():
            if metric not in current_metrics:
                continue

            current_value = current_metrics[metric]
            diff = current_value - baseline_value
            rel_diff = diff / baseline_value if baseline_value != 0 else 0
            degraded = diff < 0 and abs(rel_diff) > self.alert_threshold

            report["metrics_diff"][metric] = {
                "baseline": baseline_value,
                "current": current_value,
                "difference": diff,
                "relative_difference": rel_diff,
                "degraded": degraded
            }

            if degraded:
                report["has_degradation"] = True

        return report

    def visualize_trend(self, metric_name: str = "f1"):
        """Visualise l'évolution d'une métrique."""
        if not self.metrics_history:
            print("Aucune donnée.")
            return

        timestamps, values = [], []

        for entry in self.metrics_history:
            if metric_name in entry["metrics"]:
                timestamps.append(datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S"))
                values.append(entry["metrics"][metric_name])

        if not timestamps:
            return

        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, marker="o")
        plt.title(f"{metric_name} - {self.model_name}")
        plt.xlabel("Date")
        plt.ylabel(metric_name)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        fig_path = self.output_dir / f"{self.model_name}_{metric_name}_trend.png"
        plt.savefig(fig_path)
        plt.close()

    def set_baseline(self, y_true, y_pred, y_prob=None):
        """Définit la baseline."""
        self.baseline_metrics = self.calculate_metrics(y_true, y_pred, y_prob)


__all__ = ['ModelPerformanceTracker']
