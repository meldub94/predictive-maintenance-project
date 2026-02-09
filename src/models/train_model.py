"""Entraînement de modèles de prédiction de défaillance."""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from joblib import dump

# Chemins
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = BASE_DIR.parent / "data"

# Logging
def setup_logging(log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "train_log.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('train')

logger = setup_logging(LOG_DIR)


def load_data(train_path: Path, test_path: Path):
    """Charge train et test."""
    train = pd.read_parquet(train_path) if train_path.suffix == '.parquet' else pd.read_csv(train_path)
    test = pd.read_parquet(test_path) if test_path.suffix == '.parquet' else pd.read_csv(test_path)
    
    if 'failure_soon' not in train.columns:
        raise ValueError("Colonne 'failure_soon' manquante !")
    
    X_train = train.drop(columns=['failure_soon'])
    y_train = train['failure_soon']
    X_test = test.drop(columns=['failure_soon'])
    y_test = test['failure_soon']
    
    logger.info(f"✓ Train: {len(X_train):,} × {len(X_train.columns)}")
    logger.info(f"✓ Test: {len(X_test):,} × {len(X_test.columns)}")
    logger.info(f"✓ Positifs: {y_train.sum()} train, {y_test.sum()} test")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """Entraîne un Random Forest."""
    logger.info("Entraînement du modèle Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    logger.info("✓ Modèle entraîné")
    
    return model


def evaluate_model(model, X_test, y_test):
    """Évalue le modèle."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }
    
    print("\n" + "="*60)
    print("RÉSULTATS")
    print("="*60)
    for name, value in metrics.items():
        print(f"{name:12s}: {value:.4f}")
    print("="*60 + "\n")
    
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
    
    return metrics


def save_model(model, metrics, feature_names, output_dir: Path):
    """Sauvegarde le modèle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"random_forest_{timestamp}.pkl"
    
    model_data = {
        'model': model,
        'metrics': metrics,
        'features': feature_names,
        'timestamp': timestamp
    }
    
    dump(model_data, model_path)
    logger.info(f"✓ Modèle sauvegardé : {model_path}")
    
    return model_path


def train_pipeline(
    train_path: Path = FEATURES_DIR / "train.parquet",
    test_path: Path = FEATURES_DIR / "test.parquet",
    output_dir: Path = MODELS_DIR
):
    """Pipeline complet d'entraînement."""
    try:
        logger.info("=== ENTRAÎNEMENT DU MODÈLE ===")
        
        # Charger
        X_train, X_test, y_train, y_test = load_data(train_path, test_path)
        
        # Entraîner
        model = train_model(X_train, y_train)
        
        # Évaluer
        metrics = evaluate_model(model, X_test, y_test)
        
        # Sauvegarder
        model_path = save_model(model, metrics, X_train.columns.tolist(), output_dir)
        
        logger.info(f"✅ Entraînement terminé ! ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return model, metrics
    
    except Exception as e:
        logger.error(f"❌ Erreur : {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        model, metrics = train_pipeline()
        print(f"\n✅ Modèle prêt ! ROC-AUC = {metrics['roc_auc']:.4f}")
    except Exception as e:
        logger.error(f"❌ Échec : {e}")
        exit(1)