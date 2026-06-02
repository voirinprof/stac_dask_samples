"""
Script 05 : Masquer les nuages avec la couche SCL
--------------------------------------------------
Objectif : utiliser la Scene Classification Layer (SCL)
           de Sentinel-2 pour identifier et exclure les
           pixels nuageux avant tout calcul d'indice.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import odc.stac
from pystac_client import Client
from config_loader import load_config

cfg = load_config()

# Correspondance des valeurs SCL avec leur signification
SCL_LABELS = {
    0:  ("No Data",             "#000000"),
    1:  ("Saturated/Defective", "#ff0000"),
    2:  ("Dark Area",           "#2f2f2f"),
    3:  ("Cloud Shadow",        "#643200"),
    4:  ("Vegetation",          "#00a000"),
    5:  ("Bare Soil",           "#ffe65a"),
    6:  ("Water",               "#0000ff"),
    7:  ("Unclassified",        "#808080"),
    8:  ("Cloud (med.)",        "#c0c0c0"),
    9:  ("Cloud (high)",        "#ffffff"),
    10: ("Thin Cirrus",         "#64c8ff"),
    11: ("Snow/Ice",            "#ff96ff"),
}

# Pixels considérés valides (sans nuages ni ombres)
valid_classes = cfg["filters"]["valid_scl"]

# ── Charger une scène avec la bande SCL ──────────────────────────────────────
client = Client.open(cfg["stac"]["catalogue_url"])
items  = sorted(
    client.search(
        collections=[cfg["stac"]["collection"]],
        bbox=cfg["zone"]["bbox"],
        datetime=cfg["period"]["dates"],
        query={"eo:cloud_cover": {"lt": 30}}   # accepter plus de nuages pour illustrer
    ).items(),
    key=lambda i: i.properties.get("eo:cloud_cover", 100)
)

# Prendre une scène modérément nuageuse pour mieux illustrer l'effet du masque
item  = items[min(1, len(items) - 1)]
print(f"Scène : {item.id}")
print(f"Nuages : {item.properties['eo:cloud_cover']:.1f}%")

chunks = {"x": cfg["dask"]["chunk_x"],
          "y": cfg["dask"]["chunk_y"],
          "time": cfg["dask"]["chunk_time"]}

ds = odc.stac.load(
    [item],
    bands=["red", "nir", "scl"],
    crs=cfg["loading"]["crs"],
    resolution=cfg["loading"]["resolution"],
    bbox=cfg["zone"]["bbox"],
    chunks=chunks
).isel(time=0).compute()

# ── Distribution des classes SCL ──────────────────────────────────────────────
scl = ds["scl"].values
print(f"\nDistribution SCL :")
for val, (label, _) in SCL_LABELS.items():
    n = (scl == val).sum()
    if n > 0:
        print(f"  {val:2d}  {label:22s} : {n:8,} px  ({n/scl.size*100:5.1f}%)")

# ── Construire le masque valide ───────────────────────────────────────────────
valid_mask = np.isin(scl, valid_classes)
print(f"\nPixels valides après masquage : "
      f"{valid_mask.sum():,} ({valid_mask.mean()*100:.1f}%)")

# ── NDVI brut vs NDVI masqué ─────────────────────────────────────────────────
# Normaliser les valeurs S2 L2A (× 10 000)
red = np.clip(ds["red"].values.astype("float32") / 10000, 0, None)
nir = np.clip(ds["nir"].values.astype("float32") / 10000, 0, None)

ndvi_raw    = (nir - red) / (nir + red + 1e-10)
ndvi_masked = np.where(valid_mask, ndvi_raw, np.nan)

print(f"\nNDVI brut (avec nuages)   : moy = {np.nanmean(ndvi_raw):.4f}")
print(f"NDVI masqué (sans nuages) : moy = {np.nanmean(ndvi_masked):.4f}")
print("→ Les nuages (valeurs proches de 0) biaisent la moyenne vers le bas")

# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

# Carte SCL colorée
scl_rgb = np.zeros((*scl.shape, 3))
for val, (_, hex_c) in SCL_LABELS.items():
    r, g, b = int(hex_c[1:3],16)/255, int(hex_c[3:5],16)/255, int(hex_c[5:7],16)/255
    scl_rgb[scl == val] = [r, g, b]

axes[0].imshow(scl_rgb)
patches = [mpatches.Patch(color=c, label=f"{v}: {l}")
           for v, (l, c) in SCL_LABELS.items() if (scl == v).sum() > 0]
axes[0].legend(handles=patches, fontsize=5, loc="lower right")
axes[0].set_title("SCL — Classification de scène", fontsize=9)
axes[0].set_axis_off()

axes[1].imshow(ndvi_raw, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
axes[1].set_title("NDVI brut (nuages inclus)", fontsize=9)
axes[1].set_axis_off()

im = axes[2].imshow(ndvi_masked, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
plt.colorbar(im, ax=axes[2], shrink=0.7)
axes[2].set_title("NDVI masqué (SCL valide seulement)", fontsize=9)
axes[2].set_axis_off()

fig.suptitle(f"Masquage SCL — {cfg['zone']['name']} | {str(item.datetime.date())}",
             fontsize=11)

output_path = Path(__file__).parent.parent / "masque_scl.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nFigure sauvegardée : {output_path.name}")
plt.show()
