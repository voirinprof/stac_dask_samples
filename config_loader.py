"""
config_loader.py
----------------
Utilitaire partagé par tous les scripts.
Charge config.yaml et expose load_config().
"""

from pathlib import Path
import yaml


def load_config() -> dict:
    """Charger le fichier config.yaml depuis la racine du repo."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
