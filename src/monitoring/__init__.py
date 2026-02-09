"""
Module de monitoring des modèles.

Suivi des performances et détection de drift.
"""

from .performance_tracking import ModelPerformanceTracker

__all__ = ['ModelPerformanceTracker']

__version__ = '1.0.0'
