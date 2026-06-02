"""
Script 03 : Inspecter un Item STAC
------------------------------------
Objectif : comprendre toutes les métadonnées contenues
           dans un Item (géométrie, propriétés, assets)
           et accéder aux URLs des fichiers de bandes.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pystac
from pystac_client import Client
from config_loader import load_config

cfg = load_config()

client = Client.open(cfg["stac"]["catalogue_url"])

# ── Charger la meilleure scène disponible ────────────────────────────────────
items = sorted(
    client.search(
        collections=[cfg["stac"]["collection"]],
        bbox=cfg["zone"]["bbox"],
        datetime=cfg["period"]["dates"],
        query={"eo:cloud_cover": {"lt": cfg["filters"]["max_cloud_cover"]}}
    ).items(),
    key=lambda i: i.properties.get("eo:cloud_cover", 100)
)

if not items:
    print("Aucune scène trouvée.")
    exit()

item = items[0]

# ── Afficher la structure complète d'un Item ─────────────────────────────────
print("=" * 60)
print(f"ITEM : {item.id}")
print("=" * 60)

# Informations de base (standard GeoJSON)
print(f"\nType GeoJSON : {item.geometry['type']}")
print(f"BBox         : {item.bbox}")
print(f"Datetime     : {item.datetime}")

# Toutes les propriétés (métadonnées)
print(f"\nPropriétés ({len(item.properties)}) :")
for key, val in sorted(item.properties.items()):
    val_str = str(val)[:70]
    print(f"  {key:45s} : {val_str}")

# Assets : les fichiers réels associés à la scène
print(f"\nAssets ({len(item.assets)}) :")
print(f"  {'Nom':<20} {'Rôle':<12} {'Type MIME'}")
print("  " + "-" * 55)
for name, asset in item.assets.items():
    role = (asset.roles[0] if asset.roles else "?")
    mime = asset.media_type or ""
    print(f"  {name:<20} {role:<12} {mime}")

# Accéder aux URLs des bandes spectrales utiles
print("\nURLs des bandes (tronquées) :")
for band in ["red", "green", "blue", "nir", "scl", "thumbnail"]:
    if band in item.assets:
        url = item.assets[band].href
        print(f"  {band:12s} → ...{url[-60:]}")

# ── Valider la conformité STAC ────────────────────────────────────────────────
print("\nValidation STAC :")
try:
    item.validate()
    print("  Item valide ✓")
except pystac.STACValidationError as e:
    print(f"  Erreur : {e}")
