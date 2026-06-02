"""
Script 10 : Pipeline complet — STAC → odc-stac → Dask → NDVI
--------------------------------------------------------------
Objectif : assembler toutes les notions du cours en un seul
           script reproductible et commenté étape par étape :

  1. Rechercher les scènes Sentinel-2 (STAC)
  2. Charger en xarray lazy (odc-stac + Dask)
  3. Masquer les nuages (SCL)
  4. Calculer le NDVI (lazy)
  5. Composite greenest pixel (.compute())
  6. Sauvegarder en Zarr + carte PNG
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import matplotlib.pyplot as plt
import dask
import xarray as xr
import zarr

from pystac_client import Client
import odc.stac

from config_loader import load_config

cfg        = load_config()
OUTPUT_DIR = Path(__file__).parent.parent
t_start    = time.perf_counter()

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — RECHERCHE STAC
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("ÉTAPE 1 — Recherche STAC")
print("=" * 60)

t0     = time.perf_counter()
client = Client.open(cfg["stac"]["catalogue_url"])

items = sorted(
    client.search(
        collections=[cfg["stac"]["collection"]],
        bbox=cfg["zone"]["bbox"],
        datetime=cfg["period"]["dates"],
        query={"eo:cloud_cover": {"lt": cfg["filters"]["max_cloud_cover"]}}
    ).items(),
    key=lambda i: i.properties.get("eo:cloud_cover", 100)
)

print(f"Scènes trouvées : {len(items)}")
print(f"Période         : {cfg['period']['dates']}")
print(f"Zone            : {cfg['zone']['name']}")
print(f"Filtre nuages   : < {cfg['filters']['max_cloud_cover']}%")
print(f"Temps           : {time.perf_counter() - t0:.2f}s")

if not items:
    print("\nAucune scène — ajuster les paramètres dans config.yaml")
    exit()

for item in items[:5]:
    print(f"  {str(item.datetime.date())}  "
          f"☁ {item.properties.get('eo:cloud_cover', '?'):.1f}%  "
          f"{item.id[:38]}")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — CHARGEMENT odc-stac (lazy)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ÉTAPE 2 — Chargement odc-stac (lazy)")
print("=" * 60)

chunks = {
    "x":    cfg["dask"]["chunk_x"],
    "y":    cfg["dask"]["chunk_y"],
    "time": cfg["dask"]["chunk_time"]
}

t0 = time.perf_counter()
ds = odc.stac.load(
    items,
    bands=cfg["loading"]["bands"],   # ["red", "nir", "scl"]
    crs=cfg["loading"]["crs"],
    resolution=cfg["loading"]["resolution"],
    bbox=cfg["zone"]["bbox"],
    chunks=chunks,
    groupby="solar_day"
)

print(f"Dataset créé en {time.perf_counter() - t0:.3f}s (rien téléchargé)")
print(f"Dimensions : {dict(ds.dims)}")
print(f"Taille estimée si compute : {ds.nbytes / 1e9:.2f} GB")
print(f"Dates : {[str(t)[:10] for t in ds.time.values]}")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — MASQUE NUAGES SCL (lazy)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ÉTAPE 3 — Masque nuages SCL (lazy)")
print("=" * 60)

# Classes SCL valides lues depuis la config
valid_scl  = cfg["filters"]["valid_scl"]
valid_mask = ds["scl"].isin(valid_scl)

# Normaliser les valeurs S2 L2A (× 10 000 → 0–1) et masquer
red = (ds["red"].astype("float32") / 10000).where(valid_mask)
nir = (ds["nir"].astype("float32") / 10000).where(valid_mask)

print(f"Classes SCL valides : {valid_scl}")
print(f"Masque appliqué (lazy) ✓")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — CALCUL NDVI (lazy)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ÉTAPE 4 — Calcul NDVI (lazy)")
print("=" * 60)

# Formule NDVI — opération lazy sur les DataArrays Dask
ndvi = (nir - red) / (nir + red + 1e-10)
ndvi.attrs.update({
    "long_name": "NDVI Sentinel-2 (masqué SCL)",
    "crs":       cfg["loading"]["crs"],
    "source":    f"{cfg['stac']['collection']} via Element84"
})

print(f"NDVI lazy construit ✓")
print(f"Shape : {ndvi.dims} → {dict(zip(ndvi.dims, ndvi.shape))}")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — COMPOSITE GREENEST PIXEL (.compute())
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ÉTAPE 5 — Composite Greenest Pixel (.compute())")
print("=" * 60)

# Premier et unique .compute() du pipeline — tout s'exécute ici
print("Téléchargement + calcul en cours...")
t0 = time.perf_counter()

ndvi_composite = ndvi.max(dim="time")       # valeur max par pixel sur la série
n_valid        = valid_mask.sum(dim="time") # nombre de scènes valides par pixel

composite_r, n_valid_r = dask.compute(ndvi_composite, n_valid)

t_compute = time.perf_counter() - t0
print(f"Compute terminé en {t_compute:.1f}s")

# Statistiques du composite
vals = composite_r.values
ndvi_thr = cfg["thresholds"]["ndvi_vegetation"]
print(f"\nNDVI composite :")
print(f"  Min        : {np.nanmin(vals):.4f}")
print(f"  Max        : {np.nanmax(vals):.4f}")
print(f"  Moyenne    : {np.nanmean(vals):.4f}")
print(f"  % végétation (> {ndvi_thr}) : "
      f"{np.nanmean(vals > ndvi_thr) * 100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — SAUVEGARDE ET CARTE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ÉTAPE 6 — Sauvegarde et carte")
print("=" * 60)

# Sauvegarder en Zarr
zarr_out = OUTPUT_DIR / "ndvi_composite.zarr"
ds_out   = xr.Dataset(
    {
        "ndvi_composite":  composite_r,
        "n_valid_scenes":  n_valid_r
    },
    attrs={
        "title":    f"NDVI Greenest Pixel — {cfg['zone']['name']}",
        "periode":  cfg["period"]["dates"],
        "source":   cfg["stac"]["collection"],
        "crs":      cfg["loading"]["crs"],
        "n_scenes": len(items)
    }
)
ds_out.to_zarr(str(zarr_out), mode="w")
print(f"Zarr sauvegardé : {zarr_out.name}")

# Carte finale
fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")

im1 = axes[0].imshow(composite_r.values, cmap="RdYlGn",
                      vmin=-0.1, vmax=0.8, interpolation="nearest")
plt.colorbar(im1, ax=axes[0], shrink=0.7, label="NDVI")
axes[0].set_title(f"NDVI Greenest Pixel Composite\n"
                   f"{cfg['stac']['collection']} | {cfg['period']['dates']}",
                   fontsize=9)
axes[0].set_axis_off()

im2 = axes[1].imshow(n_valid_r.values, cmap="Blues",
                      vmin=0, interpolation="nearest")
plt.colorbar(im2, ax=axes[1], shrink=0.7, label="Nb scènes valides")
axes[1].set_title(f"Scènes valides par pixel\n({len(items)} scènes au total)",
                   fontsize=9)
axes[1].set_axis_off()

fig.suptitle(
    f"Pipeline STAC → odc-stac → Dask → NDVI\n"
    f"{cfg['zone']['name']} | CRS: {cfg['loading']['crs']} | "
    f"{cfg['loading']['resolution']}m",
    fontsize=10
)

output_fig = OUTPUT_DIR / "pipeline_ndvi.png"
fig.savefig(output_fig, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Carte sauvegardée : {output_fig.name}")

print(f"\n{'='*60}")
print(f"Pipeline complet en {time.perf_counter() - t_start:.1f}s")
print(f"{'='*60}")

plt.show()
