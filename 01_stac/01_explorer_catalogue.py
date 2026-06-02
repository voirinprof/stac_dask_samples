"""
Script 01 : Explorer un catalogue STAC
---------------------------------------
Objectif : se connecter à un catalogue STAC public,
           lister les collections disponibles et comprendre
           la hiérarchie Catalog → Collection → Item → Asset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pystac_client import Client
from config_loader import load_config

cfg = load_config()

# ── Connexion au catalogue ────────────────────────────────────────────────────
catalogue_url = cfg["stac"]["catalogue_url"]
print(f"Connexion à : {catalogue_url}\n")

client = Client.open(catalogue_url)
print(f"Titre : {client.title}")
print(f"URL   : {client.self_href}")

# ── Lister les collections disponibles ───────────────────────────────────────
print("\nCollections disponibles :")
for col in client.get_collections():
    print(f"  {col.id:45s}  {col.title or ''}")

# ── Inspecter la collection Sentinel-2 ───────────────────────────────────────
collection_id = cfg["stac"]["collection"]
print(f"\nDétail : {collection_id}")
col = client.get_collection(collection_id)

print(f"  Titre       : {col.title}")
print(f"  Description : {col.description[:100]}...")
print(f"  Licence     : {col.license}")

# Emprise temporelle de la collection
if col.extent.temporal.intervals:
    start, end = col.extent.temporal.intervals[0]
    print(f"  Période     : {start} → {end or 'présent'}")

print(f"  BBox monde  : {col.extent.spatial.bboxes[0]}")
