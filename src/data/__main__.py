"""Point d'entrée pour exécuter le pipeline avec python -m src.data"""
from pathlib import Path
from . import process_data

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    
    processed_data = process_data(
        sensor_file_path=str(PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_sensor_data.csv"),
        failure_file_path=str(PROJECT_ROOT / "data" / "raw" / "predictive_maintenance_failure_logs.csv"),
        output_dir=str(PROJECT_ROOT / "data" / "processed"),
        skip_existing=True
    )
    
    print(f"\n✅ Pipeline terminé ! {len(processed_data):,} lignes × {len(processed_data.columns)} colonnes")
