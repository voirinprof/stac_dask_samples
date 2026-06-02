"""
Script 02 : Rechercher et filtrer des scènes
---------------------------------------------
Objectif : lancer une recherche par zone, période et
           qualité nuageuse, puis trier et inspecter
           les résultats pour choisir les meilleures scènes.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pystac_client import Client
from config_loader import load_config

cfg = load_config()

# ── Paramètres depuis la config ───────────────────────────────────────────────
catalogue_url    = cfg["stac"]["catalogue_url"]
collection_id    = cfg["stac"]["collection"]
bbox             = cfg["zone"]["bbox"]
dates            = cfg["period"]["dates"]
max_cloud        = cfg["filters"]["max_cloud_cover"]

client = Client.open(catalogue_url)

# ── Recherche avec filtre nuageux ─────────────────────────────────────────────
print(f"Recherche : {collection_id}")
print(f"Zone      : {bbox}")
print(f"Période   : {dates}")
print(f"Nuages    : < {max_cloud}%\n")

results = client.search(
    collections=[collection_id],
    bbox=bbox,
    datetime=dates,
    query={"eo:cloud_cover": {"lt": max_cloud}}
)

items = list(results.items())
print(f"{len(items)} scènes trouvées")

# ── Trier par couverture nuageuse croissante ──────────────────────────────────
items_sorted = sorted(
    items,
    key=lambda i: i.properties.get("eo:cloud_cover", 100)
)

print(f"\n{'ID':<42} {'Date':<12} {'Nuages':>8}")
print("-" * 66)
for item in items_sorted[:8]:
    date  = str(item.datetime.date()) if item.datetime else "?"
    cloud = item.properties.get("eo:cloud_cover", "?")
    print(f"  {item.id:<40} {date:<12} {cloud:>7.1f}%")

# ── Inspecter la meilleure scène ──────────────────────────────────────────────
best = items_sorted[0]
print(f"\nMeilleure scène : {best.id}")
print(f"  Date        : {best.datetime.date()}")
print(f"  Nuages      : {best.properties['eo:cloud_cover']:.1f}%")
print(f"  Plateforme  : {best.properties.get('platform', '?')}")

# Lister les assets (fichiers) disponibles
print(f"\n  Assets ({len(best.assets)}) :")
for name, asset in best.assets.items():
    role = (asset.roles[0] if asset.roles else "?")
    print(f"    {name:20s} [{role}]")

# ── Mettre en cache pour éviter de re-requêter ───────────────────────────────
cache_path = Path(__file__).parent.parent / "cache_items.json"
with open(cache_path, "w") as f:
    json.dump([i.to_dict() for i in items_sorted[:10]], f, default=str)
print(f"\nTop 10 items mis en cache : {cache_path.name}")
