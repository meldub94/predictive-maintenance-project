#!/bin/bash

# Script pour corriger le data leakage dans augment.py

echo "🔧 Correction du data leakage dans augment.py..."

# Vérifier qu'on est dans le bon dossier
if [ ! -f "src/data/augment.py" ]; then
    echo "❌ Erreur: src/data/augment.py introuvable"
    echo "   Exécutez ce script depuis la racine du projet ML"
    exit 1
fi

# Backup de l'original
cp src/data/augment.py src/data/augment.py.backup
echo "✅ Backup créé: src/data/augment.py.backup"

# Utiliser Python pour faire la correction (plus fiable que sed sur Mac)
python3 << 'PYTHON_EOF'
import re

# Lire le fichier
with open('src/data/augment.py', 'r') as f:
    content = f.read()

# Pattern à remplacer
old_pattern = r'''            # Temps jusqu'à la défaillance \(en heures\)
            time_diff = \(failure_time - sensor_df\.loc\[window_mask, 'timestamp'\]\)\.dt\.total_seconds\(\) / 3600
            sensor_df\.loc\[window_mask, 'time_to_failure'\] = time_diff
            
            sensor_df\.loc\[window_mask, 'next_failure_type'\] = failure_type'''

new_pattern = '''            # ❌ DATA LEAKAGE CORRIGÉ : Ces features "voient le futur"
            # # Temps jusqu'à la défaillance (en heures)
            # time_diff = (failure_time - sensor_df.loc[window_mask, 'timestamp']).dt.total_seconds() / 3600
            # sensor_df.loc[window_mask, 'time_to_failure'] = time_diff
            # 
            # sensor_df.loc[window_mask, 'next_failure_type'] = failure_type'''

# Remplacer
content = re.sub(old_pattern, new_pattern, content, flags=re.MULTILINE)

# Écrire
with open('src/data/augment.py', 'w') as f:
    f.write(content)

print("✅ Corrections appliquées!")
PYTHON_EOF

echo ""
echo "📝 Changements effectués:"
echo "  - time_to_failure → COMMENTÉ (data leakage)"
echo "  - next_failure_type → COMMENTÉ (data leakage)"  
echo "  - failure_soon → CONSERVÉ (OK)"
echo ""
echo "🎯 Résultat attendu:"
echo "  - Features: 148 → 146 colonnes"
echo "  - ROC-AUC: 1.0 → 0.85-0.95 (plus réaliste)"
echo ""
echo "▶️  Prochaines étapes:"
echo "  1. Relancer le pipeline: python -m src.data"
echo "  2. Rebuilder features: python src/features/build_features.py"
echo "  3. Ré-entraîner: python src/models/train_model.py"
echo ""
