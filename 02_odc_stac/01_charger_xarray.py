"""
Script 04 : Charger des scènes en xarray avec odc-stac
--------------------------------------------------------
Objectif : transformer une liste d'Items STAC en un
           xarray Dataset aligné, reprojeté et découpé
           sur la zone d'étude, sans télécharger manuellement
           chaque fichier.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import odc.stac
from pystac_client import Client
from config_loader import load_config

cfg = load_config()

# ── Paramètres depuis la config ───────────────────────────────────────────────
bbox       = cfg["zone"]["bbox"]
crs        = cfg["loading"]["crs"]
resolution = cfg["loading"]["resolution"]
bands      = cfg["loading"]["bands_rgb"]  # pour visualisation (simplifié)
chunks     = {"x": cfg["dask"]["chunk_x"],
              "y": cfg["dask"]["chunk_y"],
              "time": cfg["dask"]["chunk_time"]}

# ── Recherche STAC ────────────────────────────────────────────────────────────
client = Client.open(cfg["stac"]["catalogue_url"])
items  = sorted(
    client.search(
        collections=[cfg["stac"]["collection"]],
        bbox=bbox,
        datetime=cfg["period"]["dates"],
        query={"eo:cloud_cover": {"lt": cfg["filters"]["max_cloud_cover"]}}
    ).items(),
    key=lambda i: i.properties.get("eo:cloud_cover", 100)
)
print(f"{len(items)} scènes trouvées")

if not items:
    exit()

# ── Chargement avec odc-stac ──────────────────────────────────────────────────
# odc-stac gère automatiquement :
#   - la reprojection vers le CRS cible
#   - l'alignement spatial des tuiles qui se chevauchent
#   - le découpage à la bbox
#   - la création du backend Dask (lazy)
print(f"\nChargement odc-stac (lazy)...")
print(f"  Bandes      : {bands}")
print(f"  CRS cible   : {crs}")
print(f"  Résolution  : {resolution} m")
print(f"  Chunks      : {chunks}")

ds = odc.stac.load(
    items,
    bands=bands,
    crs=crs,
    resolution=resolution,
    bbox=bbox,
    chunks=chunks,
    groupby="solar_day"   # fusionner les tuiles du même jour
)

# Le Dataset est lazy : rien n'est téléchargé
print(f"\nDataset xarray (lazy) :")
print(ds)
print(f"\nDimensions  : {dict(ds.dims)}")
print(f"Variables   : {list(ds.data_vars)}")
print(f"Dates       : {[str(t)[:10] for t in ds.time.values]}")

# ── Charger une seule date et afficher une composition RGB ────────────────────
# .isel(time=0) sélectionne la première date
# .compute() déclenche le téléchargement réel
print(f"\nChargement de la première date...")
scene = ds.isel(time=0).compute()
date_str = str(ds.time.values[0])[:10]

# Les valeurs S2 L2A sont en réflectance × 10 000
red   = scene["red"].values.astype("float32")   / 10000
green = scene["green"].values.astype("float32") / 10000  # simplifié
blue  = scene["blue"].values.astype("float32")  / 10000

# Stretch 2–98e percentile pour améliorer le contraste visuel
def stretch(arr):
    """Normaliser une bande entre ses 2e et 98e percentiles."""
    valid = arr[arr > 0]
    if len(valid) == 0:
        return arr
    p2, p98 = np.percentile(valid, [2, 98])
    return np.clip((arr - p2) / (p98 - p2 + 1e-10), 0, 1)

fig, ax = plt.subplots(figsize=(8, 7))
# Composition RGB simple (pas de correction atmosphérique ni de fusion avancée)
ax.imshow(np.dstack([stretch(red), stretch(green), stretch(blue)]))
ax.set_title(f"Sentinel-2 — {cfg['zone']['name']}\n{date_str} | {crs}",
             fontsize=10)
ax.set_axis_off()

output_path = Path(__file__).parent.parent / "scene_chargee.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Image sauvegardée : {output_path.name}")
plt.show()
