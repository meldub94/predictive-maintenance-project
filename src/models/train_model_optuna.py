"""Entraînement de modèles avec optimisation Optuna."""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from joblib import dump
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Désactiver les logs verbeux d'Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

class OptunaModelTrainer:
    """Entraîneur de modèles avec optimisation Optuna."""
    
    def __init__(self, models_dir=MODELS_DIR, random_state=42):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def load_data(self, train_path, test_path):
        """Charge les données."""
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        
        self.X_train = train.drop('failure_soon', axis=1)
        self.y_train = train['failure_soon']
        self.X_test = test.drop('failure_soon', axis=1)
        self.y_test = test['failure_soon']
        
        logger.info(f"✓ Train: {len(self.X_train):,} × {len(self.X_train.columns)}")
        logger.info(f"✓ Test: {len(self.X_test):,} × {len(self.X_test.columns)}")
    
    def optimize_random_forest(self, n_trials=30):
        """Optimise Random Forest avec Optuna."""
        logger.info(f"\n🔍 Optimisation Random Forest ({n_trials} essais)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'n_jobs': -1,
                'random_state': self.random_state
            }
            
            model = RandomForestClassifier(**params)
            
            # Validation croisée 3-fold
            score = cross_val_score(
                model, self.X_train, self.y_train,
                cv=3, scoring='roc_auc', n_jobs=-1
            ).mean()
            
            return score
        
        study = optuna.create_study(direction='maximize', study_name='RandomForest')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"🏆 Meilleur score CV: {study.best_value:.4f}")
        logger.info(f"📊 Meilleurs paramètres: {study.best_params}")
        
        return study.best_params, study
    
    def optimize_gradient_boosting(self, n_trials=20):
        """Optimise Gradient Boosting avec Optuna."""
        logger.info(f"\n🔍 Optimisation Gradient Boosting ({n_trials} essais)...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'random_state': self.random_state
            }
            
            model = GradientBoostingClassifier(**params)
            
            score = cross_val_score(
                model, self.X_train, self.y_train,
                cv=3, scoring='roc_auc', n_jobs=-1
            ).mean()
            
            return score
        
        study = optuna.create_study(direction='maximize', study_name='GradientBoosting')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"🏆 Meilleur score CV: {study.best_value:.4f}")
        logger.info(f"📊 Meilleurs paramètres: {study.best_params}")
        
        return study.best_params, study
    
    def optimize_logistic_regression(self, n_trials=15):
        """Optimise Logistic Regression avec Optuna."""
        logger.info(f"\n🔍 Optimisation Logistic Regression ({n_trials} essais)...")
        
        def objective(trial):
            params = {
                'C': trial.suggest_float('C', 0.001, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'liblinear',
                'max_iter': 1000,
                'random_state': self.random_state
            }
            
            model = LogisticRegression(**params)
            
            score = cross_val_score(
                model, self.X_train, self.y_train,
                cv=3, scoring='roc_auc', n_jobs=-1
            ).mean()
            
            return score
        
        study = optuna.create_study(direction='maximize', study_name='LogisticRegression')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"🏆 Meilleur score CV: {study.best_value:.4f}")
        logger.info(f"📊 Meilleurs paramètres: {study.best_params}")
        
        return study.best_params, study
    
    def train_with_best_params(self, model_name, best_params):
        """Entraîne avec les meilleurs paramètres."""
        logger.info(f"\n🚀 Entraînement final: {model_name}")
        
        if model_name == 'random_forest':
            model = RandomForestClassifier(**best_params)
        elif model_name == 'gradient_boosting':
            model = GradientBoostingClassifier(**best_params)
        elif model_name == 'logistic_regression':
            # Ajouter le solver compatible avec la pénalité
            if best_params.get('penalty') == 'l1':
                best_params['solver'] = 'liblinear'
            model = LogisticRegression(**best_params)
        
        model.fit(self.X_train, self.y_train)
        return model
    
    def evaluate_model(self, model, model_name):
        """Évalue le modèle."""
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_prob)
        }
        
        print(f"\n📊 {model_name.upper()}")
        print(f"   Accuracy  : {metrics['accuracy']:.4f}")
        print(f"   Precision : {metrics['precision']:.4f}")
        print(f"   Recall    : {metrics['recall']:.4f}")
        print(f"   F1-Score  : {metrics['f1']:.4f}")
        print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def save_model(self, model, model_name, params, metrics):
        """Sauvegarde le modèle."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.models_dir / f"{model_name}_{ts}.pkl"
        
        dump({
            'model': model,
            'parameters': params,
            'features': self.X_train.columns.tolist(),
            'metrics': metrics,
            'timestamp': ts,
            'optimizer': 'optuna'
        }, path, compress=0)
        
        logger.info(f"✓ Modèle sauvegardé: {path}")
        return path
    
    def save_study_visualizations(self, study, model_name):
        """Sauvegarde les visualisations Optuna."""
        viz_dir = self.models_dir / "optuna_viz"
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # Historique d'optimisation
            fig = plot_optimization_history(study)
            fig.write_html(str(viz_dir / f"{model_name}_optimization_history.html"))
            
            # Importance des paramètres
            fig = plot_param_importances(study)
            fig.write_html(str(viz_dir / f"{model_name}_param_importances.html"))
            
            logger.info(f"✓ Visualisations sauvegardées dans {viz_dir}")
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder les visualisations: {e}")


def train_pipeline_optuna(n_trials_rf=30, n_trials_gb=20, n_trials_lr=15):
    """Pipeline complet avec Optuna."""
    try:
        logger.info("=== ENTRAÎNEMENT AVEC OPTUNA ===\n")
        
        trainer = OptunaModelTrainer()
        
        # 1. Charger données
        trainer.load_data(
            FEATURES_DIR / "train.parquet",
            FEATURES_DIR / "test.parquet"
        )
        
        # 2. Optimiser les 3 modèles
        models_results = {}
        
        # Random Forest
        rf_params, rf_study = trainer.optimize_random_forest(n_trials_rf)
        rf_model = trainer.train_with_best_params('random_forest', rf_params)
        rf_metrics = trainer.evaluate_model(rf_model, 'random_forest')
        trainer.save_study_visualizations(rf_study, 'random_forest')
        models_results['random_forest'] = {
            'model': rf_model,
            'params': rf_params,
            'metrics': rf_metrics,
            'cv_score': rf_study.best_value
        }
        
        # Gradient Boosting
        gb_params, gb_study = trainer.optimize_gradient_boosting(n_trials_gb)
        gb_model = trainer.train_with_best_params('gradient_boosting', gb_params)
        gb_metrics = trainer.evaluate_model(gb_model, 'gradient_boosting')
        trainer.save_study_visualizations(gb_study, 'gradient_boosting')
        models_results['gradient_boosting'] = {
            'model': gb_model,
            'params': gb_params,
            'metrics': gb_metrics,
            'cv_score': gb_study.best_value
        }
        
        # Logistic Regression
        lr_params, lr_study = trainer.optimize_logistic_regression(n_trials_lr)
        lr_model = trainer.train_with_best_params('logistic_regression', lr_params)
        lr_metrics = trainer.evaluate_model(lr_model, 'logistic_regression')
        trainer.save_study_visualizations(lr_study, 'logistic_regression')
        models_results['logistic_regression'] = {
            'model': lr_model,
            'params': lr_params,
            'metrics': lr_metrics,
            'cv_score': lr_study.best_value
        }
        
        # 3. Trouver le meilleur
        print("\n" + "="*70)
        print("COMPARAISON DES MODÈLES (OPTUNA)")
        print("="*70)
        
        best_name = max(models_results.keys(), 
                       key=lambda k: models_results[k]['metrics']['roc_auc'])
        best_result = models_results[best_name]
        
        print(f"\n🏆 MEILLEUR MODÈLE: {best_name.upper()}")
        print(f"   ROC-AUC Test : {best_result['metrics']['roc_auc']:.4f}")
        print(f"   ROC-AUC CV   : {best_result['cv_score']:.4f}")
        print("="*70)
        
        # 4. Sauvegarder le meilleur
        trainer.save_model(
            best_result['model'],
            best_name,
            best_result['params'],
            best_result['metrics']
        )
        
        logger.info("\n✅ Optimisation Optuna terminée !")
        
        return models_results, best_name
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Entraînement avec Optuna")
    parser.add_argument("--rf-trials", type=int, default=30, help="Essais Random Forest")
    parser.add_argument("--gb-trials", type=int, default=20, help="Essais Gradient Boosting")
    parser.add_argument("--lr-trials", type=int, default=15, help="Essais Logistic Regression")
    
    args = parser.parse_args()
    
    train_pipeline_optuna(
        n_trials_rf=args.rf_trials,
        n_trials_gb=args.gb_trials,
        n_trials_lr=args.lr_trials
    )
