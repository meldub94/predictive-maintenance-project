import pandas as pd
import os
import logging
from datetime import datetime
from typing import Tuple, Optional
from pathlib import Path

# Utilisation de pathlib pour une meilleure gestion des chemins
BASE_DIR = Path(__file__).resolve().parent                    # src/data
PROJECT_ROOT = BASE_DIR.parent.parent                          # racine du projet
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTRACTED_DIR = PROCESSED_DIR / "extracted_data"
LOG_DIR = BASE_DIR

# Configuration du logging
def setup_logging(log_dir: Path) -> logging.Logger:
    """Configure le système de logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "extract_log.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('extract')

logger = setup_logging(LOG_DIR)


def validate_file_exists(file_path: Path) -> None:
    """Vérifie l'existence d'un fichier."""
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")
    logger.info(f"Fichier validé : {file_path}")


def read_csv_with_timestamp(
    file_path: Path, 
    timestamp_col: str,
    **read_csv_kwargs
) -> pd.DataFrame:
    """
    Lit un fichier CSV et convertit automatiquement la colonne timestamp.
    
    Args:
        file_path: Chemin vers le fichier CSV
        timestamp_col: Nom de la colonne à convertir en datetime
        **read_csv_kwargs: Arguments additionnels pour pd.read_csv
        
    Returns:
        DataFrame avec timestamp converti
    """
    logger.info(f"Lecture de {file_path.name}")
    
    # Optimisation : parse_dates directement lors de la lecture
    df = pd.read_csv(
        file_path,
        parse_dates=[timestamp_col],
        **read_csv_kwargs
    )
    
    logger.info(f"✓ {len(df):,} lignes chargées")
    return df


def generate_extraction_report(
    sensor_data: pd.DataFrame,
    failure_data: pd.DataFrame
) -> dict:
    """Génère un rapport détaillé de l'extraction."""
    return {
        "date_extraction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nombre_equipements": int(sensor_data["equipment_id"].nunique()),
        "types_equipement": sensor_data["equipment_type"].unique().tolist(),
        "periode_debut": str(sensor_data["timestamp"].min()),
        "periode_fin": str(sensor_data["timestamp"].max()),
        "nombre_enregistrements_capteurs": len(sensor_data),
        "nombre_defaillances": len(failure_data),
        "types_defaillance": failure_data["failure_type"].unique().tolist(),
        # Statistiques additionnelles
        "taille_memoire_capteurs_mb": round(sensor_data.memory_usage(deep=True).sum() / 1024**2, 2),
        "taille_memoire_defaillances_mb": round(failure_data.memory_usage(deep=True).sum() / 1024**2, 2),
        "colonnes_capteurs": list(sensor_data.columns),
        "colonnes_defaillances": list(failure_data.columns)
    }


def extract_data(
    sensor_file_path: Path,
    failure_file_path: Path,
    output_dir: Path = EXTRACTED_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extrait les données des fichiers CSV et les sauvegarde dans un format structuré.
    
    Args:
        sensor_file_path: Chemin vers le fichier de données capteurs
        failure_file_path: Chemin vers le fichier de journal des défaillances
        output_dir: Répertoire de sortie pour les données extraites
        
    Returns:
        Tuple contenant (DataFrame des capteurs, DataFrame des défaillances)
        
    Raises:
        FileNotFoundError: Si un fichier source est introuvable
        ValueError: Si les données sont invalides
    """
    try:
        # Création du répertoire de sortie
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire de sortie : {output_dir}")
        
        # Validation des fichiers source
        validate_file_exists(sensor_file_path)
        validate_file_exists(failure_file_path)
        
        # Extraction des données de capteurs avec parsing optimisé
        sensor_data = read_csv_with_timestamp(
            sensor_file_path,
            timestamp_col="timestamp",
            low_memory=False  # Évite les warnings de type mixte
        )
        
        # Extraction des données de défaillance
        failure_data = read_csv_with_timestamp(
            failure_file_path,
            timestamp_col="failure_timestamp",
            low_memory=False
        )
        
        # Validation basique des données
        if sensor_data.empty:
            raise ValueError("Le fichier de données capteurs est vide")
        if failure_data.empty:
            raise ValueError("Le fichier de défaillances est vide")
        
        # Sauvegarde en parquet (compression par défaut)
        sensor_output = output_dir / "sensor_data.parquet"
        failure_output = output_dir / "failure_data.parquet"
        
        sensor_data.to_parquet(
            sensor_output,
            index=False,
            compression='snappy'  # Bon compromis vitesse/compression
        )
        failure_data.to_parquet(
            failure_output,
            index=False,
            compression='snappy'
        )
        
        logger.info(f"✓ Données capteurs sauvegardées : {sensor_output}")
        logger.info(f"✓ Données défaillances sauvegardées : {failure_output}")
        logger.info(f"Forme données capteurs : {sensor_data.shape}")
        logger.info(f"Forme données défaillances : {failure_data.shape}")
        
        # Génération et sauvegarde du rapport
        report = generate_extraction_report(sensor_data, failure_data)
        report_df = pd.DataFrame([report])
        
        report_path = output_dir / "extraction_report.csv"
        report_df.to_csv(report_path, index=False)
        logger.info(f"✓ Rapport d'extraction créé : {report_path}")
        
        # Affichage du résumé
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'EXTRACTION")
        print("="*60)
        print(f"📊 Équipements uniques : {report['nombre_equipements']}")
        print(f"📈 Enregistrements capteurs : {report['nombre_enregistrements_capteurs']:,}")
        print(f"⚠️  Défaillances : {report['nombre_defaillances']:,}")
        print(f"📅 Période : {report['periode_debut']} → {report['periode_fin']}")
        print("="*60 + "\n")
        
        return sensor_data, failure_data
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'extraction : {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    # Chemins vers les fichiers sources
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    SENSOR_FILE = RAW_DIR / "predictive_maintenance_sensor_data.csv"
    FAILURE_FILE = RAW_DIR / "predictive_maintenance_failure_logs.csv"
    
    try:
        # Exécution de l'extraction
        sensor_df, failure_df = extract_data(SENSOR_FILE, FAILURE_FILE)
        
        # Aperçu des données
        print("\n📋 Aperçu des données capteurs :")
        print(sensor_df.head())
        print(f"\nTypes de données :\n{sensor_df.dtypes}")
        
        print("\n📋 Aperçu des données de défaillance :")
        print(failure_df.head())
        print(f"\nTypes de données :\n{failure_df.dtypes}")
        
        logger.info("✅ Extraction terminée avec succès")
        
    except Exception as e:
        logger.error(f"❌ Échec de l'extraction : {e}")
        exit(1)