import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy import stats
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Utilisation de pathlib
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUGMENTED_DIR = PROCESSED_DIR / "augmented_data"
FEATURES_DIR = PROCESSED_DIR / "features"
LOG_DIR = BASE_DIR.parent / "data"

# Configuration du logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure le système de logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "build_features_log.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('build_features')

logger = setup_logging(LOG_DIR)


def validate_data_file(input_dir: Path) -> Path:
    """Valide l'existence du fichier de données augmentées."""
    parquet_file = input_dir / "augmented_sensor_data.parquet"
    csv_file = input_dir / "augmented_sensor_data.csv"
    
    if parquet_file.exists():
        logger.info(f"✓ Utilisation du fichier parquet")
        return parquet_file
    elif csv_file.exists():
        logger.info(f"✓ Utilisation du fichier CSV")
        return csv_file
    else:
        raise FileNotFoundError(
            f"Fichier augmented_sensor_data introuvable dans {input_dir}"
        )

def create_polynomial_features(df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
    """Crée des caractéristiques polynomiales."""
    df = df.copy()
    base_cols = [c for c in ['temperature', 'vibration', 'pressure', 'current'] if c in df.columns]
    
    for col in base_cols:
        for d in range(2, degree + 1):
            df[f"{col}_power_{d}"] = df[col] ** d
    
    logger.info(f"✓ Caractéristiques polynomiales créées (degré {degree})")
    return df

def encode_categorical_features(df: pd.DataFrame, method: str = 'label'):
    """Encode les variables catégorielles."""
    df = df.copy()
    cat_columns = []
    
    if 'equipment_type' in df.columns:
        cat_columns.append('equipment_type')
    if 'next_failure_type' in df.columns:
        cat_columns.append('next_failure_type')
    
    encoders = {}
    
    if method == 'label':
        for col in cat_columns:
            le = LabelEncoder()
            # Gérer les NaN
            df[col] = df[col].fillna('missing')
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            df = df.drop(columns=[col])
    
    logger.info(f"✓ {len(cat_columns)} variables catégorielles encodées")
    return df, encoders


def create_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des caractéristiques fréquentielles."""
    df = df.copy()
    
    if 'vibration' not in df.columns or 'equipment_id' not in df.columns:
        logger.warning("Colonnes manquantes pour features fréquentielles")
        return df
    
    for equip_id in df['equipment_id'].unique():
        equip_data = df[df['equipment_id'] == equip_id].sort_values('timestamp')
        
        if len(equip_data) < 10:
            continue
        
        signal = equip_data['vibration'].values
        fft_result = np.fft.rfft(signal)
        fft_magnitude = np.abs(fft_result)
        
        spectral_mean = np.mean(fft_magnitude)
        spectral_std = np.std(fft_magnitude)
        
        df.loc[equip_data.index, 'vibration_spectral_mean'] = spectral_mean
        df.loc[equip_data.index, 'vibration_spectral_std'] = spectral_std
    
    logger.info("✓ Features fréquentielles créées")
    return df


def reduce_dimensionality(df: pd.DataFrame, exclude_cols: list, n_components: int = 30):
    """Réduction de dimensionnalité avec PCA."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    if len(numeric_cols) < 5:
        logger.warning("Trop peu de colonnes pour PCA")
        return df, None
    
    X = df[numeric_cols].fillna(0)
    n_components = min(n_components, len(numeric_cols), len(X) // 10)
    
    pca = PCA(n_components=n_components, random_state=42)
    transformed = pca.fit_transform(X)
    
    for i in range(n_components):
        df[f'pca_{i+1}'] = transformed[:, i]
    
    explained = sum(pca.explained_variance_ratio_)
    logger.info(f"✓ PCA: {n_components} composantes ({explained:.1%} variance)")
    
    return df, pca


def prepare_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare le DataFrame pour le ML."""
    df = df.copy()
    
    # Supprimer colonnes non-features
    drop_cols = ['timestamp', 'equipment_id', 'equipment_type']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Remplir NaN
    df = df.fillna(0)
    
    # Remplacer inf
    df = df.replace([np.inf, -np.inf], 0)
    
    logger.info(f"✓ Préparation ML : {len(df.columns)} colonnes finales")
    return df


def plot_class_distribution(y_train, y_test, output_path: Path):
    """Visualise la distribution des classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Train
    train_counts = pd.Series(y_train).value_counts()
    ax1.bar(['No Failure', 'Failure'], train_counts.values)
    ax1.set_title('Train Set Distribution')
    ax1.set_ylabel('Count')
    
    # Test
    test_counts = pd.Series(y_test).value_counts()
    ax2.bar(['No Failure', 'Failure'], test_counts.values)
    ax2.set_title('Test Set Distribution')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Graphique sauvegardé : {output_path}")


def build_features(
    input_dir: Path = AUGMENTED_DIR,
    output_dir: Path = FEATURES_DIR
) -> pd.DataFrame:
    """Construit les features finales pour le ML."""
    try:
        # Setup
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir = output_dir / 'artifacts'
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Répertoire de sortie : {output_dir}")
        
        # Chargement
        data_file = validate_data_file(input_dir)
        df = pd.read_parquet(data_file) if data_file.suffix == '.parquet' else pd.read_csv(data_file)
        logger.info(f"✓ {len(df):,} lignes × {len(df.columns)} colonnes chargées")
        
        original_shape = df.shape
        
        # Feature engineering
        logger.info("\n--- Feature Engineering ---")
        
        # 1. Polynomiales
        df = create_polynomial_features(df, degree=2)
        
        # 2. Encodage catégorielles
        df, encoders = encode_categorical_features(df, method='label')
        dump(encoders, artifacts_dir / 'encoders.joblib')
        
        # 3. Fréquentielles (optionnel, peut être lent)
        if len(df) < 100000:  # Seulement si dataset pas trop gros
            df = create_frequency_features(df)
        
        # 4. PCA
        exclude_pca = ['failure_soon', 'time_to_failure', 'days_since_last_failure']
        df, pca = reduce_dimensionality(df, exclude_cols=exclude_pca, n_components=30)
        if pca:
            dump(pca, artifacts_dir / 'pca.joblib')
        
        # 5. Préparation finale
        df = prepare_for_ml(df)
        
        # Vérifier target
        if 'failure_soon' not in df.columns:
            raise ValueError("Colonne 'failure_soon' manquante !")
        
        # Split stratifié
        logger.info("\n--- Train/Test Split ---")
        
        X = df.drop(columns=['failure_soon'])
        y = df['failure_soon']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"✓ Train: {len(train_df):,} lignes")
        logger.info(f"✓ Test: {len(test_df):,} lignes")
        logger.info(f"✓ Positifs train: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)")
        logger.info(f"✓ Positifs test: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)")
        
        # Sauvegarde
        logger.info("\n--- Sauvegarde ---")
        
        train_df.to_parquet(output_dir / 'train.parquet', index=False)
        test_df.to_parquet(output_dir / 'test.parquet', index=False)
        train_df.to_csv(output_dir / 'train.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        logger.info(f"✓ Train/Test sauvegardés")
        
        # Visualisation
        plot_class_distribution(
            y_train, y_test,
            viz_dir / 'class_distribution.png'
        )
        
        # Rapport
        report = {
            'date_creation': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'lignes_originales': original_shape[0],
            'colonnes_originales': original_shape[1],
            'lignes_finales': len(df),
            'colonnes_finales': len(df.columns),
            'train_size': len(train_df),
            'test_size': len(test_df),
            'positifs_train': int(y_train.sum()),
            'positifs_test': int(y_test.sum()),
            'taux_positifs_train_%': round(y_train.sum()/len(y_train)*100, 2),
            'taux_positifs_test_%': round(y_test.sum()/len(y_test)*100, 2),
        }
        
        pd.DataFrame([report]).to_csv(output_dir / 'features_report.csv', index=False)
        
        print("\n" + "="*60)
        print("FEATURES PRÊTES POUR LE ML")
        print("="*60)
        print(f"📊 Original : {original_shape[0]:,} × {original_shape[1]}")
        print(f"🚀 Final : {len(df):,} × {len(df.columns)}")
        print(f"📈 Train : {len(train_df):,} ({report['taux_positifs_train_%']}% positifs)")
        print(f"📉 Test : {len(test_df):,} ({report['taux_positifs_test_%']}% positifs)")
        print("="*60 + "\n")
        
        logger.info("✅ Feature engineering terminé")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Erreur : {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        featured_df = build_features()
        print(f"\n✅ Dataset prêt pour le ML : {len(featured_df):,} lignes × {len(featured_df.columns)} colonnes")
    except Exception as e:
        logger.error(f"❌ Échec : {e}")
        exit(1)