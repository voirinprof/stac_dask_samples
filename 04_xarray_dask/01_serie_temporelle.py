"""
Script 08 : Construire et analyser une série temporelle xarray
---------------------------------------------------------------
Objectif : charger plusieurs scènes en un Dataset xarray
           avec Dask en backend, puis effectuer des opérations
           sur la dimension temporelle (moyenne, max, tendance).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import xarray as xr
import dask
import dask.array as da
import matplotlib.pyplot as plt
from config_loader import load_config

cfg  = load_config()
bbox = cfg["zone"]["bbox"]
ndvi_threshold = cfg["thresholds"]["ndvi_vegetation"]

# ── Simuler une série temporelle mensuelle ────────────────────────────────────
# En production : remplacer par odc.stac.load() (voir script 05_pipeline)
print("Construction d'une série temporelle NDVI simulée (12 mois)...\n")

N_TIMES = 12
HEIGHT  = 400
WIDTH   = 500

# Coordonnées spatiales issues de la bbox
lons  = np.linspace(bbox[0], bbox[2], WIDTH)
lats  = np.linspace(bbox[3], bbox[1], HEIGHT)   # nord → sud
times = pd.date_range("2024-01-01", periods=N_TIMES, freq="MS")

# Signal saisonnier réaliste : bas en hiver, haut en été
seasonal = np.sin(np.linspace(0, 2 * np.pi, N_TIMES)) * 0.25 + 0.38

# Créer les données avec Dask pour illustrer le backend lazy
# Variation spatiale fixe : simule zones urbaines, eau, forêts, cultures
spatial_base = np.random.normal(0, 0.15, (HEIGHT, WIDTH))
raw_data = np.stack([
    np.clip(seasonal[t] + spatial_base + np.random.normal(0, 0.04, (HEIGHT, WIDTH)), -0.2, 0.9)
    for t in range(N_TIMES)
], axis=0).astype("float32")

# Envelopper dans un Dask array chunké
data_dask = da.from_array(
    raw_data,
    chunks=(cfg["dask"]["chunk_time"],
            cfg["dask"]["chunk_y"],
            cfg["dask"]["chunk_x"])
)

# ── Construire le DataArray avec coordonnées nommées ─────────────────────────
da_ndvi = xr.DataArray(
    data_dask,
    dims=["time", "y", "x"],
    coords={"time": times, "y": lats, "x": lons},
    attrs={
        "long_name": "NDVI",
        "units":     "sans dimension",
        "crs":       cfg["loading"]["crs"]
    },
    name="ndvi"
)

print("DataArray xarray (lazy) :")
print(da_ndvi)
print(f"\nBackend : {type(da_ndvi.data).__module__} ← Dask actif")

# ── Opérations temporelles (toutes lazy jusqu'au compute) ────────────────────
print("\n=== Opérations temporelles (lazy) ===")

ndvi_mean    = da_ndvi.mean(dim="time")
ndvi_max     = da_ndvi.max(dim="time")     # greenest pixel composite
ndvi_std     = da_ndvi.std(dim="time")
ndvi_monthly = da_ndvi.groupby("time.month").mean(dim="time")

# Déclencher tout en un seul compute
print("Calcul en cours...")
mean_r, max_r, std_r, monthly_r = dask.compute(
    ndvi_mean, ndvi_max, ndvi_std, ndvi_monthly
)

print(f"NDVI moyen annuel : {float(mean_r.mean()):.4f}")
print(f"NDVI max (greenest pixel) : {float(max_r.mean()):.4f}")
print(f"Variabilité temporelle (std) : {float(std_r.mean()):.4f}")

# Profil temporel du pixel central
lon_c = (bbox[0] + bbox[2]) / 2
lat_c = (bbox[1] + bbox[3]) / 2
ts    = da_ndvi.sel(x=lon_c, y=lat_c, method="nearest").compute()

print(f"\nProfil mensuel — pixel central ({lon_c:.2f}°, {lat_c:.2f}°) :")
month_names = ["Jan","Fév","Mar","Avr","Mai","Jun",
               "Jul","Aoû","Sep","Oct","Nov","Déc"]
for name, val in zip(month_names, ts.values):
    bar = "█" * int(val * 25)
    print(f"  {name} : {val:.3f}  {bar}")

# ── Visualisation ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")

im1 = axes[0].imshow(mean_r.values, cmap="RdYlGn", vmin=0.1, vmax=0.6)
plt.colorbar(im1, ax=axes[0], shrink=0.7, label="NDVI")
axes[0].set_title("NDVI moyen annuel", fontsize=10)
axes[0].set_axis_off()

im2 = axes[1].imshow(std_r.values, cmap="hot_r", vmin=0, vmax=0.15)
plt.colorbar(im2, ax=axes[1], shrink=0.7, label="σ")
axes[1].set_title("Variabilité temporelle", fontsize=10)
axes[1].set_axis_off()

axes[2].fill_between(range(12), 0, ts.values,
                      where=ts.values > ndvi_threshold,
                      alpha=0.4, color="green", label="Végétation")
axes[2].fill_between(range(12), 0, ts.values,
                      where=ts.values <= ndvi_threshold,
                      alpha=0.3, color="brown")
axes[2].plot(ts.values, "o-", color="darkgreen", lw=2, ms=5)
axes[2].axhline(ndvi_threshold, color="green", ls="--", lw=0.8)
axes[2].set_xticks(range(12))
axes[2].set_xticklabels(month_names, rotation=45, fontsize=7)
axes[2].set_ylabel("NDVI")
axes[2].set_title("Profil temporel — pixel central", fontsize=10)
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"Série temporelle NDVI — {cfg['zone']['name']} (simulé)", fontsize=11)

output_path = Path(__file__).parent.parent / "serie_temporelle.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")

# Sauvegarder aussi le Dataset en NetCDF pour le script suivant
ds = xr.Dataset({"ndvi": da_ndvi})
ds.compute().to_netcdf(Path(__file__).parent.parent / "serie_ndvi.nc")
print(f"\nFigure sauvegardée : {output_path.name}")
print("Dataset sauvegardé : serie_ndvi.nc")
plt.show()
