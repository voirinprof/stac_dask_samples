"""
Script 09 : Sauvegarder et lire en format Zarr
------------------------------------------------
Objectif : comprendre Zarr comme format natif du big data
           raster — chunké, compressé, compatible cloud —
           et comparer ses performances avec NetCDF.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec
import time
from config_loader import load_config

cfg      = load_config()
ROOT     = Path(__file__).parent.parent
nc_path  = ROOT / "serie_ndvi.nc"
zarr_path = ROOT / "serie_ndvi.zarr"

# ── Charger le dataset existant ───────────────────────────────────────────────
if not nc_path.exists():
    print(f"Fichier manquant : {nc_path.name}")
    print("→ Exécuter d'abord 04_xarray_dask/01_serie_temporelle.py")
    exit()

ds = xr.open_dataset(nc_path)
print(f"Dataset chargé depuis {nc_path.name}")
print(ds)

# ── Écrire en Zarr ────────────────────────────────────────────────────────────
print("\n=== Écriture Zarr ===")

# Définir la compression et les chunks dans l'encodage
encoding = {
    "ndvi": {
        "chunks":     (cfg["dask"]["chunk_time"],
                       cfg["dask"]["chunk_y"],
                       cfg["dask"]["chunk_x"]),
        # Blosc + LZ4 : compression rapide et efficace pour les flottants
        "compressor": BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle"),
        "dtype":      "float32"
    }
}

t0 = time.perf_counter()
ds.to_zarr(str(zarr_path), mode="w", encoding=encoding)
t_write = time.perf_counter() - t0

# Comparer les tailles sur disque
zarr_size = sum(f.stat().st_size for f in zarr_path.rglob("*") if f.is_file())
nc_size   = nc_path.stat().st_size

print(f"NetCDF : {nc_size / 1e6:.2f} MB")
print(f"Zarr   : {zarr_size / 1e6:.2f} MB (compressé)")
print(f"Écriture en {t_write:.2f}s")

# ── Lire le Zarr avec Dask (lazy) ────────────────────────────────────────────
print("\n=== Lecture Zarr ===")

t0     = time.perf_counter()
ds_z   = xr.open_zarr(str(zarr_path))
t_open = time.perf_counter() - t0

print(f"Ouverture en {t_open:.4f}s (lazy — rien chargé)")
print(f"Backend : {type(ds_z['ndvi'].data).__module__}")

# ── Comparer les temps d'accès : tuile vs série temporelle ────────────────────
print("\n=== Benchmark accès partiel ===\n")

# Accès à une tuile spatiale (256×256) — typique pour la visualisation
tile = {"y": slice(0, 256), "x": slice(0, 256)}

if nc_path.exists():
    ds_nc = xr.open_dataset(nc_path, chunks={"time": 1})
    t0    = time.perf_counter()
    _     = ds_nc["ndvi"].isel(**tile).mean().compute()
    t_nc  = time.perf_counter() - t0
    print(f"NetCDF — tuile 256×256 (toutes dates) : {t_nc:.3f}s")

t0   = time.perf_counter()
_    = ds_z["ndvi"].isel(**tile).mean().compute()
t_zs = time.perf_counter() - t0
print(f"Zarr   — tuile 256×256 (toutes dates) : {t_zs:.3f}s")

# Accès à la série temporelle d'un seul pixel
pixel = {"y": 0, "x": 0}
t0    = time.perf_counter()
_     = ds_z["ndvi"].isel(**pixel).compute()
t_ts  = time.perf_counter() - t0
print(f"Zarr   — série temporelle 1 pixel    : {t_ts:.3f}s")

# ── Inspecter la structure interne du Zarr ───────────────────────────────────
print("\n=== Structure interne ===\n")
z     = zarr.open(str(zarr_path))
z_ndvi = z["ndvi"]
print(f"Arborescence :")
print(z.tree())
print(f"\nDétail de la variable ndvi :")
print(f"  Shape      : {z_ndvi.shape}")
print(f"  Chunks     : {z_ndvi.chunks}")
print(f"  Dtype      : {z_ndvi.dtype}")
print(f"  Codecs     : {z_ndvi.metadata.codecs}")
print(f"  Chunks initialisés : {z_ndvi.nchunks_initialized} / {z_ndvi.nchunks}")

print("\n→ Zarr sur S3/GCS/Azure : remplacer le chemin local par une URL")
print("  ds = xr.open_zarr('s3://mon-bucket/serie_ndvi.zarr')")
print("  Aucun autre changement de code nécessaire.")
