"""Prédictions de défaillances."""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path):
    """Charge le modèle."""
    logger.info(f"Chargement: {model_path}")
    data = joblib.load(model_path)
    return data.get('model'), data.get('features', [])

def predict(model_path, data_path, output_path):
    """Fait des prédictions."""
    # Charger modèle
    model, features = load_model(model_path)
    
    # Charger données
    df = pd.read_parquet(data_path) if data_path.endswith('.parquet') else pd.read_csv(data_path)
    logger.info(f"Données: {df.shape}")
    
    # Prédire
    X = df.drop('failure_soon', axis=1, errors='ignore')
    if features:
        X = X[features]
    
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Résultats
    result = pd.DataFrame({
        'predicted_failure': y_pred,
        'failure_probability': y_prob,
        'timestamp': datetime.now()
    })
    
    result.to_csv(output_path, index=False)
    logger.info(f"✅ Sauvegardé: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Défaillances prédites: {y_pred.sum()} / {len(y_pred)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    # Trouver le modèle le plus récent
    models_dir = Path("models")
    model_files = sorted(models_dir.glob("*.pkl"), reverse=True)
    
    if not model_files:
        print("❌ Aucun modèle trouvé !")
        exit(1)
    
    model = str(model_files[0])
    data = "data/processed/features/test.parquet"
    output = "predictions.csv"
    
    predict(model, data, output)
