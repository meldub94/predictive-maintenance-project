"""
Module de tracking d'expériences avec Weights & Biases (WandB).

Ce module permet de :
- Logger les métriques d'entraînement
- Sauvegarder les visualisations (confusion matrix, ROC, etc.)
- Tracker les hyperparamètres
- Comparer les modèles
- Monitorer le drift des données

Installation:
    pip install wandb

Usage:
    python src/monitoring/wandb_tracking.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import WandB (optionnel)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB non installé. Installez avec: pip install wandb")


class WandbExperimentTracker:
    """
    Classe pour suivre les expériences ML avec Weights & Biases.
    
    Exemple:
        tracker = WandbExperimentTracker(project_name="maintenance-predictive")
        tracker.start_run(run_name="random_forest_v1")
        tracker.log_metrics({"accuracy": 0.95, "f1": 0.93})
        tracker.end_run()
    """
    
    def __init__(
        self, 
        project_name: str = "industrial-failure-prediction",
        entity: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None
    ):
        """
        Initialise le tracker WandB.
        
        Args:
            project_name: Nom du projet WandB
            entity: Nom de l'entité (utilisateur/organisation)
            config: Configuration de l'expérience
            tags: Tags pour filtrer les expériences
            group: Groupe d'expériences
            job_type: Type de job ('training', 'evaluation', etc.)
        """
        if not WANDB_AVAILABLE:
            raise ImportError("WandB n'est pas installé. Installez avec: pip install wandb")
        
        self.project_name = project_name
        self.entity = entity
        self.run = None
        self.config = config or {}
        self.tags = tags or []
        self.group = group
        self.job_type = job_type
        self.artifacts = {}
        
    def start_run(self, run_name: Optional[str] = None):
        """
        Démarre une nouvelle run WandB.
        
        Args:
            run_name: Nom spécifique pour cette run
            
        Returns:
            self pour chaînage
        """
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            config=self.config,
            tags=self.tags,
            group=self.group,
            job_type=self.job_type,
            name=run_name,
            reinit=True
        )
        print(f"✅ WandB run démarrée: {self.run.name}")
        print(f"🔗 Dashboard: {self.run.url}")
        return self
    
    def end_run(self):
        """Termine la run actuelle."""
        if self.run:
            self.run.finish()
            print(f"✅ WandB run terminée: {self.run.name}")
            self.run = None
    
    def log_config(self, config_dict: Dict):
        """
        Enregistre la configuration.
        
        Args:
            config_dict: Dictionnaire de configuration
        """
        if self.run:
            for key, value in config_dict.items():
                self.run.config[key] = value
    
    def log_metrics(self, metrics_dict: Dict, step: Optional[int] = None):
        """
        Enregistre des métriques.
        
        Args:
            metrics_dict: Dictionnaire de métriques (ex: {"accuracy": 0.95})
            step: Étape optionnelle (epoch, iteration)
        """
        if self.run:
            self.run.log(metrics_dict, step=step)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "model"):
        """
        Enregistre une matrice de confusion.
        
        Args:
            y_true: Valeurs réelles
            y_pred: Prédictions
            model_name: Nom du modèle
        """
        if not self.run:
            return
        
        try:
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(y_true, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                       xticklabels=["No Failure", "Failure"],
                       yticklabels=["No Failure", "Failure"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - {model_name}")
            
            self.run.log({f"{model_name}_confusion_matrix": wandb.Image(plt)})
            plt.close()
            
        except Exception as e:
            print(f"❌ Erreur log confusion matrix: {e}")
    
    def log_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str = "model"):
        """
        Enregistre la courbe ROC.
        
        Args:
            y_true: Valeurs réelles
            y_prob: Probabilités prédites (classe positive)
            model_name: Nom du modèle
        """
        if not self.run:
            return
        
        try:
            from sklearn.metrics import roc_curve, auc
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            plt.grid(alpha=0.3)
            
            self.run.log({f"{model_name}_roc_curve": wandb.Image(plt)})
            plt.close()
            
            # Log AUC
            self.run.log({f"{model_name}_roc_auc": roc_auc})
            
        except Exception as e:
            print(f"❌ Erreur log ROC curve: {e}")
    
    def log_feature_importance(self, model: Any, feature_names: List[str], model_name: str = "model"):
        """
        Enregistre l'importance des features.
        
        Args:
            model: Modèle ML avec feature_importances_ ou coef_
            feature_names: Liste des noms de features
            model_name: Nom du modèle
        """
        if not self.run:
            return
        
        try:
            # Extraire importances
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            else:
                print("⚠️ Modèle ne supporte pas feature importances")
                return
            
            # Créer DataFrame
            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False).head(20)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
            plt.title(f"Top 20 Feature Importances - {model_name}")
            plt.xlabel("Importance")
            plt.tight_layout()
            
            self.run.log({f"{model_name}_feature_importance": wandb.Image(plt)})
            plt.close()
            
            # Log table
            self.run.log({f"{model_name}_importance_table": wandb.Table(dataframe=importance_df)})
            
        except Exception as e:
            print(f"❌ Erreur log feature importance: {e}")
    
    def log_model(self, model_path: str, model_name: str, metadata: Optional[Dict] = None):
        """
        Enregistre un modèle comme artifact.
        
        Args:
            model_path: Chemin du fichier modèle (.pkl, .joblib)
            model_name: Nom de l'artifact
            metadata: Métadonnées (métriques, hyperparams, etc.)
        """
        if not self.run:
            return
        
        try:
            artifact = wandb.Artifact(
                name=model_name,
                type="model",
                description=f"Modèle entraîné - {model_name}",
                metadata=metadata or {}
            )
            
            artifact.add_file(model_path)
            self.run.log_artifact(artifact)
            
            print(f"✅ Modèle sauvegardé: {model_name}")
            
        except Exception as e:
            print(f"❌ Erreur log model: {e}")
    
    def log_dataset(self, data_path: str, dataset_name: str, dataset_type: str = "train"):
        """
        Enregistre un dataset comme artifact.
        
        Args:
            data_path: Chemin du fichier dataset (.parquet, .csv)
            dataset_name: Nom de l'artifact
            dataset_type: Type ('train', 'test', 'val')
        """
        if not self.run:
            return
        
        try:
            artifact = wandb.Artifact(
                name=dataset_name,
                type="dataset",
                description=f"Dataset {dataset_type}",
                metadata={"type": dataset_type}
            )
            
            artifact.add_file(data_path)
            self.run.log_artifact(artifact)
            
            print(f"✅ Dataset sauvegardé: {dataset_name}")
            
        except Exception as e:
            print(f"❌ Erreur log dataset: {e}")


# ============================================================================
# EXEMPLE D'UTILISATION AVEC LE PROJET MAINTENANCE PRÉDICTIVE
# ============================================================================

if __name__ == "__main__":
    """
    Exemple d'utilisation du tracker WandB avec le projet.
    
    Note: Vous devez d'abord configurer WandB:
        1. Installer: pip install wandb
        2. Login: wandb login
        3. Entrer votre API key depuis https://wandb.ai/authorize
    """
    
    print("=" * 70)
    print("EXEMPLE WANDB TRACKING - MAINTENANCE PRÉDICTIVE")
    print("=" * 70)
    
    if not WANDB_AVAILABLE:
        print("\n❌ WandB n'est pas installé.")
        print("Installez avec: pip install wandb")
        sys.exit(1)
    
    # Vérifier que WandB est configuré
    try:
        wandb.login()
    except Exception:
        print("\n⚠️ WandB n'est pas configuré.")
        print("Lancez: wandb login")
        sys.exit(1)
    
    # =======================================================================
    # CONFIGURATION DE L'EXPÉRIENCE
    # =======================================================================
    
    config = {
        "model_type": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42,
        "test_size": 0.2,
        "smote": True,
        "features": {
            "temporal": True,
            "rolling_windows": [5, 10, 30],
            "lag_periods": [1, 3, 5, 10],
            "interactions": True
        }
    }

    # =======================================================================
    # INITIALISATION DU TRACKER
    # =======================================================================

    tracker = WandbExperimentTracker(
        project_name="industrial-failure-prediction",
        config=config,
        tags=["random_forest", "smote", "v2"],
        group="smote_experiments",
        job_type="training"
    )

    # Démarrer la run
    tracker.start_run(run_name=f"RF_SMOTE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    try:
        # ===================================================================
        # SIMULATION CHARGEMENT DONNÉES
        # ===================================================================
        
        print("\n📊 Chargement des données...")
        
        # Chemin du dataset (adapter selon votre structure)
        train_path = Path("data/processed/features/train.parquet")
        test_path = Path("data/processed/features/test.parquet")
        
        if not train_path.exists():
            print(f"\n⚠️ Fichier {train_path} introuvable.")
            print("Exécutez d'abord le pipeline:")
            print("  python -m src.data")
            print("  python src/features/build_features.py")
            sys.exit(1)
        
        # Charger données
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # Split X/y
        X_train = train_df.drop('failure_soon', axis=1)
        y_train = train_df['failure_soon']
        X_test = test_df.drop('failure_soon', axis=1)
        y_test = test_df['failure_soon']
        
        # Log statistiques dataset
        tracker.log_metrics({
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "positive_rate_train": y_train.mean(),
            "positive_rate_test": y_test.mean()
        })
        
        print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")

        # ===================================================================
        # SMOTE
        # ===================================================================

        print("\n⚖️ Application SMOTE...")
        from imblearn.over_sampling import SMOTE

        tracker.log_metrics({
            "before_smote_positive": int(y_train.sum()),
            "before_smote_negative": int((y_train == 0).sum()),
        })

        smote = SMOTE(random_state=config["random_state"])
        X_train, y_train = smote.fit_resample(X_train, y_train)

        tracker.log_metrics({
            "after_smote_samples": len(X_train),
            "after_smote_positive": int(y_train.sum()),
        })

        print(f"✅ SMOTE appliqué : {len(X_train):,} lignes (50/50)")

        # ===================================================================
        # ENTRAÎNEMENT MODÈLE
        # ===================================================================

        print("\n🔧 Entraînement du modèle...")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        # Créer et entraîner modèle
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=config["random_state"],
            n_jobs=-1,
            verbose=1
        )

        model.fit(X_train, y_train)
        print("✅ Modèle entraîné")
        
        # ===================================================================
        # ÉVALUATION
        # ===================================================================
        
        print("\n📈 Évaluation du modèle...")
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Métriques
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        
        # Log métriques
        tracker.log_metrics(metrics)
        
        print("\n📊 Métriques:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # ===================================================================
        # VISUALISATIONS
        # ===================================================================
        
        print("\n📊 Génération visualisations...")
        
        # Confusion matrix
        tracker.log_confusion_matrix(y_test, y_pred, model_name="RandomForest")
        
        # ROC curve
        tracker.log_roc_curve(y_test, y_prob, model_name="RandomForest")
        
        # Feature importance
        tracker.log_feature_importance(model, X_train.columns.tolist(), model_name="RandomForest")
        
        print("✅ Visualisations loggées")
        
        # ===================================================================
        # SAUVEGARDE MODÈLE
        # ===================================================================
        
        print("\n💾 Sauvegarde du modèle...")
        
        import joblib
        
        # Créer répertoire models si nécessaire
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Sauvegarder
        model_filename = f"random_forest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path = models_dir / model_filename
        
        joblib.dump({
            'model': model,
            'features': X_train.columns.tolist(),
            'metrics': metrics,
            'config': config
        }, model_path)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        
        # Log modèle dans WandB
        tracker.log_model(
            model_path=str(model_path),
            model_name="random_forest_baseline",
            metadata={
                **metrics,
                **config,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # ===================================================================
        # VALIDATION CROISÉE (OPTIONNEL)
        # ===================================================================
        
        print("\n🔄 Validation croisée...")
        
        from sklearn.model_selection import cross_val_score
        
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, scoring='f1', n_jobs=-1
        )
        
        tracker.log_metrics({
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "cv_f1_min": cv_scores.min(),
            "cv_f1_max": cv_scores.max()
        })
        
        print(f"  CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # ===================================================================
        # RÉSUMÉ
        # ===================================================================
        
        print("\n" + "=" * 70)
        print("✅ EXPÉRIENCE TERMINÉE AVEC SUCCÈS")
        print("=" * 70)
        print(f"\n🔗 Dashboard WandB: {tracker.run.url}")
        print("\nMétriques principales:")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - F1-Score: {metrics['f1_score']:.4f}")
        print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"\n💾 Modèle sauvegardé: {model_path}")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        
        # Log erreur dans WandB
        if tracker.run:
            tracker.log_metrics({"error": True, "error_message": str(e)})
    
    finally:
        # Terminer la run
        tracker.end_run()
        print("\n👋 Run WandB terminée")