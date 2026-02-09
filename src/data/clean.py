import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Tuple, List
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Utilisation de pathlib pour une meilleure gestion des chemins
BASE_DIR = Path(__file__).resolve().parent                    # src/data
PROJECT_ROOT = BASE_DIR.parent.parent                          # racine du projet
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTRACTED_DIR = PROCESSED_DIR / "extracted_data"
CLEANED_DIR = PROCESSED_DIR / "cleaned_data"
LOG_DIR = BASE_DIR

# Configuration du logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure le système de logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "clean_log.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('clean')

logger = setup_logging(LOG_DIR)


def plot_distribution(df: pd.DataFrame, column: str, output_path: Path) -> None:
    """
    Crée un graphique de distribution pour identifier les anomalies.
    
    Args:
        df: DataFrame contenant les données
        column: Nom de la colonne à visualiser
        output_path: Chemin de sauvegarde du graphique
    """
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True, bins=50)
        plt.title(f'Distribution de {column}')
        plt.xlabel(column)
        plt.ylabel('Fréquence')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Graphique sauvegardé : {output_path.name}")
    except Exception as e:
        logger.warning(f"Impossible de créer le graphique pour {column}: {e}")
        plt.close()


def detect_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    threshold: float = 3.0
) -> pd.Series:
    """
    Détecte les valeurs aberrantes dans une colonne.
    
    Args:
        df: DataFrame contenant les données
        column: Nom de la colonne à vérifier
        method: Méthode de détection ('zscore' ou 'iqr')
        threshold: Seuil de détection (pour z-score uniquement)
        
    Returns:
        Series booléenne indiquant les valeurs aberrantes (True = outlier)
    """
    if column not in df.columns:
        logger.warning(f"Colonne {column} introuvable")
        return pd.Series(False, index=df.index)
    
    # Vérifier qu'il y a des données numériques valides
    valid_data = df[column].dropna()
    if len(valid_data) == 0:
        return pd.Series(False, index=df.index)
    
    if method == "zscore":
        std = valid_data.std()
        if std == 0 or pd.isna(std):
            return pd.Series(False, index=df.index)
        
        mean = valid_data.mean()
        z_scores = np.abs((df[column] - mean) / std)
        return z_scores > threshold
    
    elif method == "iqr":
        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return pd.Series(False, index=df.index)
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # CORRECTION : Utiliser des comparaisons séparées au lieu de soustraction
        below_lower = df[column] < lower_bound
        above_upper = df[column] > upper_bound
        
        # Utiliser l'opérateur | (OR) au lieu de - (soustraction)
        return below_lower | above_upper
    
    else:
        raise ValueError(f"Méthode '{method}' non reconnue. Utilisez 'zscore' ou 'iqr'.")


def validate_data_files(input_dir: Path) -> Tuple[Path, Path]:
    """
    Valide l'existence des fichiers de données.
    
    Returns:
        Tuple des chemins (sensor_data, failure_data)
        
    Raises:
        FileNotFoundError si les fichiers n'existent pas
    """
    sensor_path = input_dir / "sensor_data.parquet"
    failure_path = input_dir / "failure_data.parquet"
    
    if not sensor_path.exists():
        raise FileNotFoundError(
            f"Fichier sensor_data.parquet introuvable dans {input_dir}. "
            f"Exécutez d'abord extract.py"
        )
    
    if not failure_path.exists():
        raise FileNotFoundError(
            f"Fichier failure_data.parquet introuvable dans {input_dir}. "
            f"Exécutez d'abord extract.py"
        )
    
    logger.info(f"✓ Fichiers de données validés")
    return sensor_path, failure_path


def remove_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace les valeurs infinies par NaN dans toutes les colonnes numériques.
    
    Args:
        df: DataFrame à nettoyer
        
    Returns:
        DataFrame avec valeurs infinies remplacées
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        logger.info(f"Valeurs infinies détectées : {inf_counts}")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "drop",
    fill_value=None
) -> pd.DataFrame:
    """
    Gère les valeurs manquantes selon la stratégie choisie.
    
    Args:
        df: DataFrame à traiter
        strategy: 'drop', 'mean', 'median', 'fill'
        fill_value: Valeur de remplacement si strategy='fill'
        
    Returns:
        DataFrame traité
    """
    missing_before = df.isnull().sum().sum()
    
    if missing_before == 0:
        logger.info("✓ Aucune valeur manquante")
        return df
    
    original_len = len(df)
    
    if strategy == "drop":
        df = df.dropna()
        logger.info(
            f"Stratégie 'drop': {original_len - len(df)} lignes supprimées "
            f"({missing_before} valeurs manquantes)"
        )
    
    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        logger.info(f"Stratégie 'mean': {missing_before} valeurs imputées")
    
    elif strategy == "median":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        logger.info(f"Stratégie 'median': {missing_before} valeurs imputées")
    
    elif strategy == "fill" and fill_value is not None:
        df = df.fillna(fill_value)
        logger.info(f"Stratégie 'fill': {missing_before} valeurs remplacées par {fill_value}")
    
    return df


def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """
    Supprime les doublons du DataFrame.
    
    Args:
        df: DataFrame à traiter
        subset: Liste de colonnes à considérer (None = toutes)
        
    Returns:
        DataFrame sans doublons
    """
    duplicates_count = df.duplicated(subset=subset).sum()
    
    if duplicates_count > 0:
        df = df.drop_duplicates(subset=subset)
        logger.info(f"✓ {duplicates_count} doublons supprimés")
    else:
        logger.info("✓ Aucun doublon détecté")
    
    return df


def process_outliers(
    df: pd.DataFrame,
    numeric_columns: List[str],
    viz_dir: Path,
    method: str = "iqr",
    action: str = "flag"
) -> pd.DataFrame:
    """
    Détecte et traite les valeurs aberrantes.
    
    Args:
        df: DataFrame à traiter
        numeric_columns: Liste des colonnes numériques à vérifier
        viz_dir: Répertoire pour les visualisations
        method: Méthode de détection ('zscore' ou 'iqr')
        action: 'flag' (ajouter colonne indicatrice) ou 'replace' (médiane)
        
    Returns:
        DataFrame traité
    """
    outlier_summary = {}
    
    for column in numeric_columns:
        if column not in df.columns:
            continue
        
        # Visualisation de la distribution
        plot_path = viz_dir / f"{column}_distribution.png"
        plot_distribution(df, column, plot_path)
        
        # Détection des outliers
        outliers_mask = detect_outliers(df, column, method=method)
        outliers_count = int(outliers_mask.sum())  # Conversion explicite en int
        outlier_summary[column] = outliers_count
        
        logger.info(f"Outliers dans {column}: {outliers_count} ({outliers_count/len(df)*100:.2f}%)")
        
        if outliers_count > 0:
            if action == "flag":
                # Créer une colonne indicatrice
                df[f"{column}_outlier"] = outliers_mask.astype(int)  # 0 ou 1
            
            elif action == "replace":
                # Remplacer par la médiane des valeurs non-aberrantes
                # CORRECTION : Utiliser ~ (NOT) au lieu de - (soustraction)
                non_outlier_mask = ~outliers_mask
                median_value = df.loc[non_outlier_mask, column].median()
                df.loc[outliers_mask, column] = median_value
                logger.info(f"  → Remplacés par médiane: {median_value:.2f}")
    
    return df


def impute_failure_data(failure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute les valeurs manquantes dans les données de défaillance.
    
    Args:
        failure_df: DataFrame des défaillances
        
    Returns:
        DataFrame avec valeurs imputées
    """
    for column in ['repair_duration', 'repair_cost']:
        if column not in failure_df.columns:
            continue
        
        missing_count = failure_df[column].isnull().sum()
        if missing_count == 0:
            continue
        
        logger.info(f"Imputation de {missing_count} valeurs pour {column}")
        
        # Médiane par type de défaillance
        if 'failure_type' in failure_df.columns:
            medians = failure_df.groupby('failure_type')[column].median()
            
            for failure_type in failure_df['failure_type'].unique():
                mask = (
                    (failure_df['failure_type'] == failure_type) &
                    (failure_df[column].isnull())
                )
                
                if failure_type in medians.index and not pd.isna(medians[failure_type]):
                    failure_df.loc[mask, column] = medians[failure_type]
        
        # Médiane globale pour les valeurs restantes
        global_median = failure_df[column].median()
        failure_df[column] = failure_df[column].fillna(global_median)
    
    return failure_df


def validate_equipment_consistency(
    sensor_df: pd.DataFrame,
    failure_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Valide la cohérence des équipements entre capteurs et défaillances.
    
    Args:
        sensor_df: DataFrame des capteurs
        failure_df: DataFrame des défaillances
        
    Returns:
        DataFrame des défaillances nettoyé
    """
    if 'equipment_id' not in failure_df.columns or 'equipment_id' not in sensor_df.columns:
        return failure_df
    
    valid_equipment_ids = set(sensor_df['equipment_id'].unique())
    failure_equipment_ids = set(failure_df['equipment_id'].unique())
    
    invalid_ids = failure_equipment_ids - valid_equipment_ids
    
    if invalid_ids:
        logger.warning(
            f"⚠️  {len(invalid_ids)} équipement(s) dans failures inexistant(s) "
            f"dans sensors: {invalid_ids}"
        )
        
        before_len = len(failure_df)
        failure_df = failure_df[failure_df['equipment_id'].isin(valid_equipment_ids)]
        removed = before_len - len(failure_df)
        
        logger.info(f"✓ {removed} défaillances supprimées pour cohérence")
    else:
        logger.info("✓ Tous les équipements sont cohérents")
    
    return failure_df


def generate_cleaning_report(
    sensor_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    original_len_sensor: int,
    original_len_failure: int,
    numeric_columns: List[str],
    output_dir: Path
) -> None:
    """
    Génère un rapport détaillé du nettoyage.
    
    Args:
        sensor_df: DataFrame capteurs nettoyé
        failure_df: DataFrame défaillances nettoyé
        original_len_sensor: Nombre initial de lignes capteurs
        original_len_failure: Nombre initial de lignes défaillances
        numeric_columns: Liste des colonnes numériques analysées
        output_dir: Répertoire de sortie
    """
    outlier_counts = {}
    for col in numeric_columns:
        if col in sensor_df.columns:
            outliers = detect_outliers(sensor_df, col, method="iqr")
            outlier_counts[col] = int(outliers.sum())
    
    report = {
        "date_nettoyage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "capteurs_initial": original_len_sensor,
        "capteurs_final": len(sensor_df),
        "capteurs_supprimes": original_len_sensor - len(sensor_df),
        "taux_retention_capteurs_%": round(len(sensor_df) / original_len_sensor * 100, 2),
        "defaillances_initial": original_len_failure,
        "defaillances_final": len(failure_df),
        "defaillances_supprimees": original_len_failure - len(failure_df),
        "outliers_temperature": outlier_counts.get('temperature', 0),
        "outliers_vibration": outlier_counts.get('vibration', 0),
        "outliers_pressure": outlier_counts.get('pressure', 0),
        "outliers_current": outlier_counts.get('current', 0),
    }
    
    report_df = pd.DataFrame([report])
    report_path = output_dir / "cleaning_report.csv"
    report_df.to_csv(report_path, index=False)
    
    logger.info(f"✓ Rapport de nettoyage créé : {report_path}")
    
    # Affichage console
    print("\n" + "="*60)
    print("RAPPORT DE NETTOYAGE")
    print("="*60)
    print(f"📊 Capteurs : {original_len_sensor:,} → {len(sensor_df):,} "
          f"(rétention: {report['taux_retention_capteurs_%']}%)")
    print(f"⚠️  Défaillances : {original_len_failure} → {len(failure_df)}")
    print(f"🔍 Outliers totaux : {sum(outlier_counts.values()):,}")
    print("="*60 + "\n")


def clean_data(
    input_dir: Path = EXTRACTED_DIR,
    output_dir: Path = CLEANED_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nettoie les données extraites et les sauvegarde.
    
    Args:
        input_dir: Répertoire contenant les données extraites (parquet)
        output_dir: Répertoire pour les données nettoyées
        
    Returns:
        Tuple (DataFrame capteurs nettoyé, DataFrame défaillances nettoyé)
        
    Raises:
        FileNotFoundError: Si les fichiers d'entrée n'existent pas
    """
    try:
        # Création des répertoires
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Répertoire de sortie : {output_dir}")
        logger.info(f"Répertoire visualisations : {viz_dir}")
        
        # Validation des fichiers d'entrée
        sensor_path, failure_path = validate_data_files(input_dir)
        
        # Chargement des données
        logger.info(f"Chargement de {sensor_path.name}...")
        sensor_df = pd.read_parquet(sensor_path)
        logger.info(f"✓ {len(sensor_df):,} enregistrements capteurs chargés")
        
        logger.info(f"Chargement de {failure_path.name}...")
        failure_df = pd.read_parquet(failure_path)
        logger.info(f"✓ {len(failure_df)} défaillances chargées")
        
        # Sauvegarde des longueurs initiales
        original_len_sensor = len(sensor_df)
        original_len_failure = len(failure_df)
        
        # === NETTOYAGE DES DONNÉES CAPTEURS ===
        
        logger.info("\n--- Nettoyage des données capteurs ---")
        
        # 1. Valeurs infinies
        sensor_df = remove_infinite_values(sensor_df)
        
        # 2. Valeurs manquantes
        missing_values = sensor_df.isnull().sum()
        if missing_values.sum() > 0:
            logger.info(f"Valeurs manquantes :\n{missing_values[missing_values > 0]}")
        sensor_df = handle_missing_values(sensor_df, strategy="drop")
        
        # 3. Doublons
        sensor_df = remove_duplicates(sensor_df)
        
        # 4. Valeurs aberrantes
        numeric_columns = ['temperature', 'vibration', 'pressure', 'current']
        numeric_columns = [c for c in numeric_columns if c in sensor_df.columns]
        
        sensor_df = process_outliers(
            sensor_df,
            numeric_columns,
            viz_dir,
            method="iqr",
            action="flag"  # ou "replace" pour remplacer par la médiane
        )
        
        # 5. Tri par équipement et timestamp
        if {'equipment_id', 'timestamp'}.issubset(sensor_df.columns):
            sensor_df = sensor_df.sort_values(by=['equipment_id', 'timestamp'])
            logger.info("✓ Données triées par équipement et timestamp")
        
        # === NETTOYAGE DES DONNÉES DE DÉFAILLANCE ===
        
        logger.info("\n--- Nettoyage des données de défaillance ---")
        
        # 1. Valeurs manquantes
        missing_failure = failure_df.isnull().sum()
        if missing_failure.sum() > 0:
            logger.info(f"Valeurs manquantes :\n{missing_failure[missing_failure > 0]}")
        
        # 2. Imputation
        failure_df = impute_failure_data(failure_df)
        
        # 3. Doublons
        failure_df = remove_duplicates(failure_df)
        
        # 4. Cohérence des équipements
        failure_df = validate_equipment_consistency(sensor_df, failure_df)
        
        # === SAUVEGARDE ===
        
        logger.info("\n--- Sauvegarde des données nettoyées ---")
        
        # CSV pour inspection manuelle
        sensor_csv_path = output_dir / "clean_sensor_data.csv"
        failure_csv_path = output_dir / "clean_failure_data.csv"
        
        sensor_df.to_csv(sensor_csv_path, index=False)
        failure_df.to_csv(failure_csv_path, index=False)
        
        logger.info(f"✓ Capteurs sauvegardés : {sensor_csv_path}")
        logger.info(f"✓ Défaillances sauvegardées : {failure_csv_path}")
        
        # Parquet pour le pipeline (plus performant)
        sensor_parquet_path = output_dir / "clean_sensor_data.parquet"
        failure_parquet_path = output_dir / "clean_failure_data.parquet"
        
        sensor_df.to_parquet(sensor_parquet_path, index=False)
        failure_df.to_parquet(failure_parquet_path, index=False)
        
        logger.info(f"✓ Parquet capteurs : {sensor_parquet_path}")
        logger.info(f"✓ Parquet défaillances : {failure_parquet_path}")
        
        # Rapport de nettoyage
        generate_cleaning_report(
            sensor_df,
            failure_df,
            original_len_sensor,
            original_len_failure,
            numeric_columns,
            output_dir
        )
        
        logger.info("✅ Nettoyage terminé avec succès")
        
        return sensor_df, failure_df
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage : {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    try:
        # Exécution du nettoyage
        clean_sensor_df, clean_failure_df = clean_data()
        
        # Affichage des statistiques descriptives
        print("\n📊 Statistiques des données capteurs nettoyées :")
        print(clean_sensor_df.describe())
        
        print("\n📊 Statistiques des données de défaillance nettoyées :")
        print(clean_failure_df.describe())
        
        print("\n✅ Nettoyage terminé ! Fichiers disponibles dans data/processed/cleaned_data/")
    
    except Exception as e:
        logger.error(f"❌ Échec du nettoyage : {e}")
        exit(1)