"""
ShadeNail — Point d'entrée principal
═══════════════════════════════════════════════════════════════════
Extraction automatique de couleur de vernis à ongles.

Usage inférence (mode principal) :
  python -m src.main --input data/raw/products.csv
  python -m src.main --input data/raw/products.csv --output results.csv

Usage complet (training + evaluation) :
  python -m src.main --full
  python -m src.main --train
  python -m src.main --evaluate
"""
from src.clustering.pipeline_main import main

if __name__ == "__main__":
    main()