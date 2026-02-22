"""Entraînement de modèles de prédiction de défaillance."""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from joblib import dump

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

class ModelTrainer:
    def __init__(self, models_dir=MODELS_DIR, random_state=42):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        
        # Définir les modèles à tester
        # Note: GB et LR sont trop lents sur le dataset SMOTE (404K lignes), RF suffit
        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_state
            ),
        }
    
    def load_data(self, train_path, test_path):
        """Charge train et test."""
        train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)
        
        X_train = train.drop('failure_soon', axis=1)
        y_train = train['failure_soon']
        X_test = test.drop('failure_soon', axis=1)
        y_test = test['failure_soon']
        
        logger.info(f"✓ Train: {len(X_train):,} × {len(X_train.columns)}")
        logger.info(f"✓ Test: {len(X_test):,} × {len(X_test.columns)}")
        return X_train, X_test, y_train, y_test
    
    def apply_smote(self, X_train, y_train):
        """Applique SMOTE pour rééquilibrer les classes."""
        logger.info(f"Distribution avant SMOTE : {dict(y_train.value_counts())}")
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(f"Distribution après SMOTE : {dict(pd.Series(y_resampled).value_counts())}")
        logger.info(f"✓ SMOTE appliqué : {len(X_train):,} → {len(X_resampled):,} lignes")
        return X_resampled, y_resampled

    def train_models(self, X_train, y_train):
        """Entraîne tous les modèles."""
        trained = {}
        
        for name, model in self.models.items():
            logger.info(f"\n🔄 Entraînement: {name}...")
            model.fit(X_train, y_train)
            trained[name] = model
            logger.info(f"✓ {name} entraîné")
        
        return trained
    
    def evaluate_models(self, trained, X_test, y_test):
        """Évalue tous les modèles."""
        results = {}
        
        print("\n" + "="*70)
        print("COMPARAISON DES MODÈLES")
        print("="*70)
        
        for name, model in trained.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            if y_prob is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
            
            results[name] = metrics
            
            # Afficher
            print(f"\n📊 {name.upper()}")
            print(f"   Accuracy  : {metrics['accuracy']:.4f}")
            print(f"   Precision : {metrics['precision']:.4f}")
            print(f"   Recall    : {metrics['recall']:.4f}")
            print(f"   F1-Score  : {metrics['f1']:.4f}")
            if 'roc_auc' in metrics:
                print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")
        
        print("="*70)
        
        return results
    
    def find_best_model(self, results):
        """Trouve le meilleur modèle selon ROC-AUC."""
        best_name = max(results.keys(), key=lambda k: results[k].get('roc_auc', 0))
        best_score = results[best_name].get('roc_auc', 0)
        
        print(f"\n🏆 MEILLEUR MODÈLE: {best_name.upper()} (ROC-AUC: {best_score:.4f})\n")
        logger.info(f"Meilleur modèle: {best_name}")
        
        return best_name
    
    def save_model(self, name, model, metrics, features):
        """Sauvegarde le meilleur modèle."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.models_dir / f"{name}_{ts}.pkl"
        
        dump({
            'model': model,
            'features': features,
            'metrics': metrics,
            'timestamp': ts
        }, path, compress=0)
        
        logger.info(f"✓ Modèle sauvegardé: {path}")
        return path

def train_pipeline():
    """Pipeline complet."""
    try:
        logger.info("=== ENTRAÎNEMENT DES MODÈLES ===\n")
        
        trainer = ModelTrainer()
        
        # 1. Charger données
        X_train, X_test, y_train, y_test = trainer.load_data(
            FEATURES_DIR / "train.parquet",
            FEATURES_DIR / "test.parquet"
        )
        
        # 2. Appliquer SMOTE pour rééquilibrer
        X_train, y_train = trainer.apply_smote(X_train, y_train)

        # 3. Entraîner tous les modèles
        trained = trainer.train_models(X_train, y_train)
        
        # 4. Évaluer tous les modèles
        results = trainer.evaluate_models(trained, X_test, y_test)

        # 5. Trouver le meilleur
        best_name = trainer.find_best_model(results)

        # 6. Sauvegarder le meilleur
        trainer.save_model(
            best_name,
            trained[best_name],
            results[best_name],
            X_train.columns.tolist()
        )
        
        logger.info("✅ Entraînement terminé !")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    train_pipeline()