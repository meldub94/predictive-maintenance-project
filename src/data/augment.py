import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Utilisation de pathlib
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DIR = PROCESSED_DIR / "cleaned_data"
AUGMENTED_DIR = PROCESSED_DIR / "augmented_data"
LOG_DIR = BASE_DIR

# Configuration du logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure le système de logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "augment_log.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('augment')

logger = setup_logging(LOG_DIR)


def validate_data_files(input_dir: Path) -> Tuple[Path, Path]:
    """Valide l'existence des fichiers de données nettoyées."""
    # Essayer d'abord les fichiers parquet (plus rapides)
    sensor_parquet = input_dir / "clean_sensor_data.parquet"
    failure_parquet = input_dir / "clean_failure_data.parquet"
    
    sensor_csv = input_dir / "clean_sensor_data.csv"
    failure_csv = input_dir / "clean_failure_data.csv"
    
    if sensor_parquet.exists() and failure_parquet.exists():
        logger.info("✓ Utilisation des fichiers parquet (plus rapide)")
        return sensor_parquet, failure_parquet
    elif sensor_csv.exists() and failure_csv.exists():
        logger.info("✓ Utilisation des fichiers CSV")
        return sensor_csv, failure_csv
    else:
        raise FileNotFoundError(
            f"Fichiers de données nettoyées introuvables dans {input_dir}. "
            f"Exécutez d'abord clean.py"
        )


def load_data(file_path: Path) -> pd.DataFrame:
    """Charge un fichier CSV ou Parquet avec parsing de dates."""
    if file_path.suffix == '.parquet':
        return pd.read_parquet(file_path)
    else:
        # Détecter automatiquement les colonnes de type datetime
        df = pd.read_csv(file_path)
        for col in df.columns:
            if 'timestamp' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des caractéristiques temporelles à partir de timestamp."""
    df = df.copy()
    
    if 'timestamp' not in df.columns:
        logger.warning("Colonne 'timestamp' introuvable, skip time features")
        return df
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Composantes temporelles basiques
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=lundi
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['year'] = df['timestamp'].dt.year
    
    # Indicateurs jour/nuit et weekend
    # CORRECTION IMPORTANTE : Utiliser isin() qui retourne un booléen puis .astype(int)
    night_hours = list(range(0, 6)) + list(range(22, 24))
    df['is_night'] = df['hour'].isin(night_hours).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cycliques (sin/cos pour capturer la nature circulaire du temps)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    logger.info(f"✓ {12} caractéristiques temporelles créées")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    window_sizes: List[int] = [5, 10, 30],
    group_by: str = 'equipment_id'
) -> pd.DataFrame:
    """Crée des caractéristiques de fenêtres glissantes."""
    df = df.copy()
    
    if group_by not in df.columns or 'timestamp' not in df.columns:
        logger.warning(f"Colonnes requises manquantes pour rolling features")
        return df
    
    df = df.sort_values(by=[group_by, 'timestamp'])
    
    numeric_cols = ['temperature', 'vibration', 'pressure', 'current']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    features_created = 0
    
    for window in window_sizes:
        for col in numeric_cols:
            # Grouper et calculer statistiques rolling
            grouped = df.groupby(group_by)[col]
            
            # OPTIMISATION : Utiliser transform pour garder l'index
            df[f'{col}_rolling_mean_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'{col}_rolling_std_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'{col}_rolling_min_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f'{col}_rolling_max_{window}'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
            features_created += 4
    
    logger.info(f"✓ {features_created} rolling features créées (windows: {window_sizes})")
    return df


def create_lag_features(
    df: pd.DataFrame,
    lag_periods: List[int] = [1, 3, 5, 10],
    group_by: str = 'equipment_id'
) -> pd.DataFrame:
    """Crée des caractéristiques de lag et de variations."""
    df = df.copy()
    
    if group_by not in df.columns or 'timestamp' not in df.columns:
        logger.warning("Colonnes requises manquantes pour lag features")
        return df
    
    df = df.sort_values(by=[group_by, 'timestamp'])
    
    numeric_cols = ['temperature', 'vibration', 'pressure', 'current']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    features_created = 0
    
    for lag in lag_periods:
        for col in numeric_cols:
            # Lag simple
            df[f'{col}_lag_{lag}'] = df.groupby(group_by)[col].shift(lag)
            
            # Changement absolu
            df[f'{col}_change_{lag}'] = df[col] - df[f'{col}_lag_{lag}']
            
            # Changement en pourcentage
            df[f'{col}_pct_change_{lag}'] = df.groupby(group_by)[col].pct_change(periods=lag)
            
            # Remplacer les inf par NaN
            df[f'{col}_pct_change_{lag}'].replace([np.inf, -np.inf], np.nan, inplace=True)
            
            features_created += 3
    
    logger.info(f"✓ {features_created} lag features créées (lags: {lag_periods})")
    return df


def add_failure_indicators(
    sensor_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    time_window: int = 24
) -> pd.DataFrame:
    """
    Ajoute des indicateurs de défaillance prochaine.
    
    ATTENTION : Cette fonction peut être lente sur de gros datasets.
    """
    sensor_df = sensor_df.copy()
    
    if 'equipment_id' not in sensor_df.columns or 'timestamp' not in sensor_df.columns:
        logger.warning("Colonnes requises manquantes pour failure indicators")
        return sensor_df
    
    # Initialisation
    sensor_df['failure_soon'] = 0
    # sensor_df['time_to_failure'] = pd.NA
    # sensor_df['next_failure_type'] = pd.NA
    
    logger.info(f"Ajout des indicateurs de défaillance (fenêtre: {time_window}h)...")
    
    failure_count = 0
    
    for idx, failure in failure_df.iterrows():
        equipment_id = failure['equipment_id']
        failure_time = pd.to_datetime(failure['failure_timestamp'])
        failure_type = failure.get('failure_type', 'unknown')
        
        # Fenêtre temporelle avant la défaillance
        window_start = failure_time - pd.Timedelta(hours=time_window)
        
        # CORRECTION : Utiliser des masques séparés puis combiner avec &
        mask_equipment = sensor_df['equipment_id'] == equipment_id
        mask_before_failure = sensor_df['timestamp'] <= failure_time
        mask_after_window = sensor_df['timestamp'] >= window_start
        
        # Combiner les masques avec & (AND logique)
        window_mask = mask_equipment & mask_before_failure & mask_after_window
        
        if window_mask.sum() > 0:
            sensor_df.loc[window_mask, 'failure_soon'] = 1
            
            # ❌ DATA LEAKAGE CORRIGÉ : Ces features "voient le futur"
            # # Temps jusqu'à la défaillance (en heures)
            # time_diff = (failure_time - sensor_df.loc[window_mask, 'timestamp']).dt.total_seconds() / 3600
            # sensor_df.loc[window_mask, 'time_to_failure'] = time_diff
            # 
            # sensor_df.loc[window_mask, 'next_failure_type'] = failure_type
            
            failure_count += 1
    
    positive_samples = (sensor_df['failure_soon'] == 1).sum()
    logger.info(
        f"✓ {failure_count} défaillances traitées, "
        f"{positive_samples:,} échantillons positifs ({positive_samples/len(sensor_df)*100:.2f}%)"
    )
    
    return sensor_df


def create_component_health_features(
    sensor_df: pd.DataFrame,
    failure_df: pd.DataFrame
) -> pd.DataFrame:
    """Crée des indicateurs de santé basés sur l'historique des défaillances."""
    sensor_df = sensor_df.copy()
    
    if 'equipment_id' not in sensor_df.columns or 'timestamp' not in sensor_df.columns:
        logger.warning("Colonnes requises manquantes pour health features")
        return sensor_df
    
    # Initialisation avec des valeurs par défaut appropriées
    sensor_df['days_since_last_failure'] = 999  # Grande valeur = pas de défaillance récente
    sensor_df['failures_count_last_30days'] = 0
    sensor_df['failures_count_last_90days'] = 0
    
    logger.info("Création des caractéristiques de santé des composants...")
    
    for equipment_id in sensor_df['equipment_id'].unique():
        equip_failures = failure_df[failure_df['equipment_id'] == equipment_id].sort_values('failure_timestamp')
        
        if len(equip_failures) == 0:
            continue
        
        equip_mask = sensor_df['equipment_id'] == equipment_id
        equip_sensors = sensor_df[equip_mask].copy()
        
        for idx in equip_sensors.index:
            current_time = sensor_df.at[idx, 'timestamp']
            
            # Défaillances passées
            prev_failures = equip_failures[equip_failures['failure_timestamp'] <= current_time]
            
            if len(prev_failures) > 0:
                # Jours depuis la dernière défaillance
                last_failure_time = prev_failures['failure_timestamp'].max()
                days_since = (current_time - last_failure_time).days
                sensor_df.at[idx, 'days_since_last_failure'] = days_since
                
                # Défaillances dans les 30 derniers jours
                window_30 = current_time - timedelta(days=30)
                failures_30 = prev_failures[prev_failures['failure_timestamp'] >= window_30]
                sensor_df.at[idx, 'failures_count_last_30days'] = len(failures_30)
                
                # Défaillances dans les 90 derniers jours
                window_90 = current_time - timedelta(days=90)
                failures_90 = prev_failures[prev_failures['failure_timestamp'] >= window_90]
                sensor_df.at[idx, 'failures_count_last_90days'] = len(failures_90)
    
    logger.info("✓ 3 health features créées")
    return sensor_df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des interactions entre variables de base."""
    df = df.copy()
    
    base_cols = ['temperature', 'vibration', 'pressure', 'current']
    base_cols = [c for c in base_cols if c in df.columns]
    
    features_created = 0
    
    # Interactions multiplicatives
    for i, col1 in enumerate(base_cols):
        for col2 in base_cols[i+1:]:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            features_created += 1
    
    # Ratios (avec protection division par zéro)
    if 'temperature' in base_cols and 'pressure' in base_cols:
        df['temp_pressure_ratio'] = np.where(
            df['pressure'] != 0,
            df['temperature'] / df['pressure'],
            0
        )
        features_created += 1
    
    if 'vibration' in base_cols and 'current' in base_cols:
        df['vibration_current_ratio'] = np.where(
            df['current'] != 0,
            df['vibration'] / df['current'],
            0
        )
        features_created += 1
    
    logger.info(f"✓ {features_created} interaction features créées")
    return df


def create_statistical_features(df: pd.DataFrame, group_by: str = 'equipment_id') -> pd.DataFrame:
    """Crée des features statistiques par équipement."""
    df = df.copy()
    
    numeric_cols = ['temperature', 'vibration', 'pressure', 'current']
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    
    features_created = 0
    
    for col in numeric_cols:
        # Statistiques globales par équipement
        stats = df.groupby(group_by)[col].agg(['mean', 'std']).add_prefix(f'{col}_equip_')
        df = df.merge(stats, left_on=group_by, right_index=True, how='left')
        
        # Déviation par rapport à la moyenne de l'équipement
        df[f'{col}_deviation_from_mean'] = df[col] - df[f'{col}_equip_mean']
        
        # Score z par équipement
        df[f'{col}_zscore'] = np.where(
            df[f'{col}_equip_std'] > 0,
            (df[col] - df[f'{col}_equip_mean']) / df[f'{col}_equip_std'],
            0
        )
        
        features_created += 4
    
    logger.info(f"✓ {features_created} statistical features créées")
    return df


def feature_scaling(
    df: pd.DataFrame,
    method: str = 'standard',
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Applique une mise à l'échelle aux features numériques.
    
    Returns:
        Tuple (DataFrame scalé, scaler fitted) pour pouvoir réutiliser le scaler
    """
    df = df.copy()
    
    # Colonnes à exclure de la mise à l'échelle
    default_exclude = ['timestamp', 'equipment_id', 'equipment_type', 'failure_soon']
    if exclude_cols:
        default_exclude.extend(exclude_cols)
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if col not in default_exclude]
    
    if len(cols_to_scale) == 0:
        logger.warning("Aucune colonne à scaler")
        return df, None
    
    # Remplacer inf et NaN par 0
    df[cols_to_scale] = df[cols_to_scale].replace([np.inf, -np.inf], np.nan)
    df[cols_to_scale] = df[cols_to_scale].fillna(0)
    
    # Appliquer le scaling
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'standard' ou 'minmax'.")
    
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    logger.info(f"✓ {len(cols_to_scale)} colonnes scalées (méthode: {method})")
    return df, scaler


def plot_feature_importances(
    df: pd.DataFrame,
    target_col: str = 'failure_soon',
    output_path: Optional[Path] = None,
    top_n: int = 20
) -> None:
    """Visualise les corrélations avec la cible."""
    if target_col not in df.columns:
        logger.warning(f"Colonne cible '{target_col}' introuvable")
        return
    
    try:
        # Calculer les corrélations
        correlations = df.corr(numeric_only=True)[target_col].drop(target_col, errors='ignore')
        correlations = correlations.abs().sort_values(ascending=False).head(top_n)
        
        # Créer le graphique
        plt.figure(figsize=(12, 8))
        sns.barplot(x=correlations.values, y=correlations.index, palette='viridis')
        plt.title(f'Top {top_n} features corrélées avec {target_col} (valeur absolue)')
        plt.xlabel('Corrélation absolue')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            logger.info(f"✓ Graphique sauvegardé : {output_path}")
            plt.close()
        else:
            plt.show()
    
    except Exception as e:
        logger.warning(f"Impossible de créer le graphique de corrélations : {e}")
        plt.close()


def generate_augmentation_report(
    df: pd.DataFrame,
    original_shape: Tuple[int, int],
    output_dir: Path
) -> None:
    """Génère un rapport d'augmentation."""
    report = {
        'date_augmentation': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'lignes': len(df),
        'colonnes_originales': original_shape[1],
        'colonnes_finales': len(df.columns),
        'colonnes_ajoutees': len(df.columns) - original_shape[1],
        'echantillons_positifs': int((df['failure_soon'] == 1).sum()) if 'failure_soon' in df.columns else 0,
        'taux_positifs_%': round((df['failure_soon'] == 1).sum() / len(df) * 100, 2) if 'failure_soon' in df.columns else 0,
        'valeurs_manquantes': int(df.isnull().sum().sum()),
        'taille_memoire_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }
    
    report_df = pd.DataFrame([report])
    report_path = output_dir / 'augmentation_report.csv'
    report_df.to_csv(report_path, index=False)
    
    logger.info(f"✓ Rapport d'augmentation créé : {report_path}")
    
    print("\n" + "="*60)
    print("RAPPORT D'AUGMENTATION")
    print("="*60)
    print(f"📊 Lignes : {report['lignes']:,}")
    print(f"📈 Colonnes : {report['colonnes_originales']} → {report['colonnes_finales']} (+{report['colonnes_ajoutees']})")
    print(f"⚠️  Échantillons positifs : {report['echantillons_positifs']:,} ({report['taux_positifs_%']}%)")
    print(f"💾 Taille mémoire : {report['taille_memoire_mb']} MB")
    print("="*60 + "\n")


def augment_data(
    input_dir: Path = CLEANED_DIR,
    output_dir: Path = AUGMENTED_DIR
) -> pd.DataFrame:
    """Pipeline complet d'augmentation des données."""
    try:
        # Création des répertoires
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Répertoire de sortie : {output_dir}")
        
        # Validation et chargement
        sensor_path, failure_path = validate_data_files(input_dir)
        
        logger.info(f"Chargement de {sensor_path.name}...")
        sensor_df = load_data(sensor_path)
        logger.info(f"✓ {len(sensor_df):,} enregistrements capteurs")
        
        logger.info(f"Chargement de {failure_path.name}...")
        failure_df = load_data(failure_path)
        logger.info(f"✓ {len(failure_df)} défaillances")
        
        original_shape = sensor_df.shape
        
        # === PIPELINE D'AUGMENTATION ===
        
        logger.info("\n--- Création des features ---")
        
        # 1. Features temporelles
        sensor_df = create_time_features(sensor_df)
        
        # 2. Rolling features
        sensor_df = create_rolling_features(sensor_df, window_sizes=[5, 10, 30])
        
        # 3. Lag features
        sensor_df = create_lag_features(sensor_df, lag_periods=[1, 3, 5, 10])
        
        # 4. Failure indicators (TARGET)
        sensor_df = add_failure_indicators(sensor_df, failure_df, time_window=24)
        
        # 5. Component health features
        sensor_df = create_component_health_features(sensor_df, failure_df)
        
        # 6. Interaction features
        sensor_df = create_interaction_features(sensor_df)
        
        # 7. Statistical features
        sensor_df = create_statistical_features(sensor_df)
        
        # 8. Feature scaling
        logger.info("\n--- Mise à l'échelle ---")
        sensor_df, scaler = feature_scaling(sensor_df, method='standard')
        
        # 9. Visualisation
        logger.info("\n--- Visualisation ---")
        plot_feature_importances(
            sensor_df,
            target_col='failure_soon',
            output_path=viz_dir / 'feature_importances.png'
        )
        
        # === SAUVEGARDE ===
        
        logger.info("\n--- Sauvegarde ---")
        
        # Parquet (plus rapide pour ML)
        parquet_path = output_dir / 'augmented_sensor_data.parquet'
        sensor_df.to_parquet(parquet_path, index=False)
        logger.info(f"✓ Données sauvegardées : {parquet_path}")
        
        # CSV (pour inspection)
        csv_path = output_dir / 'augmented_sensor_data.csv'
        sensor_df.to_csv(csv_path, index=False)
        logger.info(f"✓ CSV sauvegardé : {csv_path}")
        
        # Rapport
        generate_augmentation_report(sensor_df, original_shape, output_dir)
        
        logger.info("✅ Augmentation terminée avec succès")
        
        return sensor_df
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'augmentation : {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        augmented_data = augment_data()
        
        print("\n📊 Statistiques des données augmentées :")
        print(augmented_data.describe())
        
        print(f"\n📋 Colonnes créées : {len(augmented_data.columns)}")
        print(f"Liste des colonnes : {list(augmented_data.columns)}")
        
        print("\n✅ Augmentation terminée ! Fichiers disponibles dans data/processed/augmented_data/")
    
    except Exception as e:
        logger.error(f"❌ Échec de l'augmentation : {e}")
        exit(1)