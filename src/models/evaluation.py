import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    classification_report, average_precision_score, roc_auc_score
)
from joblib import load

# Chemins
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "evaluation"
LOG_DIR = BASE_DIR.parent / "data"

# Logging
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "evaluation_log.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('evaluation')

logger = setup_logging(LOG_DIR)


def load_test_data(test_path: Path):
    """Charge les données de test."""
    if not test_path.exists():
        raise FileNotFoundError(f"Fichier test introuvable: {test_path}")
    
    df = pd.read_parquet(test_path) if test_path.suffix == '.parquet' else pd.read_csv(test_path)
    
    if 'failure_soon' not in df.columns:
        raise ValueError("Colonne 'failure_soon' manquante dans les données de test")
    
    y = df['failure_soon']
    X = df.drop(columns=['failure_soon'])
    
    logger.info(f"✓ Test chargé : {len(X):,} lignes × {len(X.columns)} features")
    return X, y


def load_model(model_path: Path):
    """Charge un modèle entraîné."""
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    
    obj = load(model_path)
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    
    logger.info(f"✓ Modèle chargé : {model_path.name}")
    return model


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calcule toutes les métriques."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        metrics['roc_auc'] = roc_auc_score(y_true, prob_pos)
        metrics['avg_precision'] = average_precision_score(y_true, prob_pos)
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    """Matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Matrice de confusion : {output_path.name}")


def plot_roc_curve(y_true, y_prob, output_path: Path):
    """Courbe ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Courbe ROC : {output_path.name}")


def plot_pr_curve(y_true, y_prob, output_path: Path):
    """Courbe Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'b-', lw=2, label=f'PR (AP = {avg_prec:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Courbe PR : {output_path.name}")


def plot_feature_importance(model, feature_names, output_path: Path, top_n=20):
    """Importance des features."""
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Modèle sans feature_importances_")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Feature importance : {output_path.name}")


def evaluate_model(
    model_path: Path = None,
    test_path: Path = FEATURES_DIR / "test.parquet",
    output_dir: Path = REPORTS_DIR
):
    """Évalue un modèle sur le test set."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("=== ÉVALUATION DU MODÈLE ===")
        
        # Charger
        X_test, y_test = load_test_data(test_path)
        
        # Trouver le modèle le plus récent si non spécifié
        if model_path is None:
            model_files = sorted(MODELS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not model_files:
                raise FileNotFoundError("Aucun modèle trouvé dans models/")
            model_path = model_files[0]
            logger.info(f"Utilisation du modèle le plus récent: {model_path.name}")
        
        model = load_model(model_path)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Métriques
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        
        print("\n" + "="*60)
        print("RÉSULTATS DE L'ÉVALUATION")
        print("="*60)
        for name, value in metrics.items():
            print(f"{name:15s}: {value:.4f}")
        print("="*60 + "\n")
        
        # Rapport détaillé
        print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
        
        # Visualisations
        plot_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")
        
        if y_prob is not None:
            prob_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            plot_roc_curve(y_test, prob_pos, output_dir / "roc_curve.png")
            plot_pr_curve(y_test, prob_pos, output_dir / "pr_curve.png")
        
        plot_feature_importance(model, X_test.columns, output_dir / "feature_importance.png")
        
        # Sauvegarder rapport
        report = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': model_path.name,
            'test_size': len(y_test),
            'positive_samples': int(y_test.sum()),
            **metrics
        }
        
        pd.DataFrame([report]).to_csv(output_dir / "evaluation_report.csv", index=False)
        
        logger.info(f"✅ Évaluation terminée : {output_dir}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"❌ Erreur : {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        metrics = evaluate_model()
        print(f"\n✅ Évaluation terminée ! ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
    except Exception as e:
        logger.error(f"❌ Échec : {e}")
        exit(1)