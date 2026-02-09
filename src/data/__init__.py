"""
Module de traitement de données pour la prédiction de risque de défaillance industrielle.

Ce module contient des outils pour extraire, nettoyer et augmenter des données
de maintenance prédictive issues de capteurs industriels.

Fonctions principales:
- process_data: Exécute le pipeline complet (extraction, nettoyage, augmentation)
- extract_data: Extrait les données des fichiers CSV sources
- clean_data: Nettoie les données extraites
- augment_data: Enrichit les données avec des caractéristiques additionnelles

Utilisation typique:
```python
from src.data import process_data

# Exécuter le pipeline complet
processed_data = process_data(
    sensor_file_path="data/raw/predictive_maintenance_sensor_data.csv",
    failure_file_path="data/raw/predictive_maintenance_failure_logs.csv",
    output_dir="data/processed"
)
```
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

# Imports relatifs des modules du package
from .extract import extract_data
from .clean import clean_data
from .augment import augment_data

# Configuration du logging pour le module
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "predictive_maintenance.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('predictive_maintenance')

# Exports publics du module
__all__ = ['extract_data', 'clean_data', 'augment_data', 'process_data']
__version__ = '0.1.0'


def process_data(
    sensor_file_path: str,
    failure_file_path: str,
    output_dir: str = 'data/processed',
    skip_existing: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Exécute le pipeline complet de traitement des données.
    
    Pipeline: Extraction → Nettoyage → Augmentation
    
    Args:
        sensor_file_path: Chemin vers le fichier de données capteurs (CSV)
        failure_file_path: Chemin vers le fichier de journal des défaillances (CSV)
        output_dir: Répertoire principal pour les sorties
        skip_existing: Si True, ignore les étapes déjà complétées
        verbose: Si True, affiche des informations détaillées
        
    Returns:
        DataFrame des données finales augmentées, prêtes pour l'entraînement
        
    Raises:
        FileNotFoundError: Si les fichiers sources sont introuvables
        ValueError: Si les données sont invalides
        
    Example:
        >>> from src.data import process_data
        >>> df = process_data(
        ...     sensor_file_path="data/raw/sensors.csv",
        ...     failure_file_path="data/raw/failures.csv"
        ... )
    """
    try:
        start_time = datetime.now()
        
        # Conversion en Path pour meilleure gestion
        output_path = Path(output_dir)
        sensor_path = Path(sensor_file_path)
        failure_path = Path(failure_file_path)
        
        # Validation des fichiers sources
        if not sensor_path.exists():
            raise FileNotFoundError(f"Fichier capteurs introuvable : {sensor_path}")
        if not failure_path.exists():
            raise FileNotFoundError(f"Fichier défaillances introuvable : {failure_path}")
        
        # Structure de répertoires
        extracted_dir = output_path / 'extracted_data'
        cleaned_dir = output_path / 'cleaned_data'
        augmented_dir = output_path / 'augmented_data'
        
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire principal : {output_path}")
        
        # === ÉTAPE 1: EXTRACTION ===
        
        sensor_parquet = extracted_dir / 'sensor_data.parquet'
        
        if skip_existing and sensor_parquet.exists():
            logger.info("✓ Données extraites trouvées, skip extraction")
            # Charger pour le résumé
            sensor_df = pd.read_parquet(sensor_parquet)
            failure_df = pd.read_parquet(extracted_dir / 'failure_data.parquet')
        else:
            logger.info("\n=== ÉTAPE 1/3: EXTRACTION ===")
            sensor_df, failure_df = extract_data(
                str(sensor_path),
                str(failure_path),
                output_dir=extracted_dir
            )
            logger.info("✅ Extraction terminée")
        
        # === ÉTAPE 2: NETTOYAGE ===
        
        clean_parquet = cleaned_dir / 'clean_sensor_data.parquet'
        
        if skip_existing and clean_parquet.exists():
            logger.info("✓ Données nettoyées trouvées, skip nettoyage")
            sensor_clean_df = pd.read_parquet(clean_parquet)
            failure_clean_df = pd.read_parquet(cleaned_dir / 'clean_failure_data.parquet')
        else:
            logger.info("\n=== ÉTAPE 2/3: NETTOYAGE ===")
            sensor_clean_df, failure_clean_df = clean_data(
                input_dir=extracted_dir,
                output_dir=cleaned_dir
            )
            logger.info("✅ Nettoyage terminé")
        
        # === ÉTAPE 3: AUGMENTATION ===
        
        augmented_parquet = augmented_dir / 'augmented_sensor_data.parquet'
        
        if skip_existing and augmented_parquet.exists():
            logger.info("✓ Données augmentées trouvées, chargement")
            augmented_df = pd.read_parquet(augmented_parquet)
        else:
            logger.info("\n=== ÉTAPE 3/3: AUGMENTATION ===")
            augmented_df = augment_data(
                input_dir=cleaned_dir,
                output_dir=augmented_dir
            )
            logger.info("✅ Augmentation terminée")
        
        # === RÉSUMÉ ===
        
        elapsed_time = datetime.now() - start_time
        
        if verbose:
            print("\n" + "="*60)
            print("RÉSUMÉ DU PIPELINE DE TRAITEMENT")
            print("="*60)
            print(f"📊 Capteurs originaux      : {len(sensor_df):,}")
            print(f"⚠️  Défaillances originales : {len(failure_df)}")
            print(f"🧹 Capteurs après nettoyage: {len(sensor_clean_df):,}")
            print(f"⚠️  Défaillances nettoyées  : {len(failure_clean_df)}")
            print(f"🚀 Dataset final           : {len(augmented_df):,} lignes × {len(augmented_df.columns)} colonnes")
            print(f"⏱️  Temps total             : {elapsed_time}")
            print("="*60 + "\n")
            
            logger.info(f"Pipeline terminé en {elapsed_time}")
        
        return augmented_df
    
    except Exception as e:
        logger.error(f"❌ Erreur lors du traitement : {str(e)}", exc_info=True)
        raise


# Note importante : Ne PAS exécuter le pipeline automatiquement à l'import
# Cela causerait des problèmes lors de l'import du module

# Si vous voulez un script d'exécution, créez un fichier séparé comme :
# src/data/run_pipeline.py

if __name__ == "__main__":
    # Ce bloc ne s'exécute que si le fichier est lancé directement
    # ex: python -m src.data
    
    PROJECT_ROOT = BASE_DIR.parent.parent
    
    processed_data = process_data(
        sensor_file_path=str(PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_sensor_data.csv"),
        failure_file_path=str(PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_failure_logs.csv"),
        output_dir=str(PROJECT_ROOT / "data" / "processed")
    )
    
    print(f"\n✅ Pipeline terminé ! {len(processed_data):,} lignes prêtes pour le ML")