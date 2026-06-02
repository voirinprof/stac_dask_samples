"""
Script 06 : Lazy evaluation et .compute()
------------------------------------------
Objectif : comprendre le modèle mental fondamental de Dask —
           les opérations construisent un graphe de tâches
           sans rien calculer, jusqu'à l'appel de .compute().
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import dask.array as da
import dask
import time
from config_loader import load_config

cfg = load_config()
chunk_x = cfg["dask"]["chunk_x"]
chunk_y = cfg["dask"]["chunk_y"]

# ── 1. Créer un tableau Dask ──────────────────────────────────────────────────
print("=== 1. NumPy vs Dask ===\n")

# NumPy : toute la mémoire allouée immédiatement
arr_np = np.ones((3000, 3000), dtype="float32")
print(f"NumPy : {arr_np.nbytes / 1e6:.1f} MB en mémoire maintenant")

# Dask : description du futur calcul seulement
arr_da = da.ones((3000, 3000), dtype="float32",
                  chunks=(chunk_x, chunk_y))
print(f"Dask  : {arr_da}")
print(f"→ {arr_da.npartitions} chunks de {chunk_x}×{chunk_y} px — rien en mémoire")

# ── 2. Les opérations sont lazy ───────────────────────────────────────────────
print("\n=== 2. Construction du graphe (lazy) ===\n")

# Simuler un raster 5 bandes (bandes, y, x)
raster = da.random.random((5, 2000, 2000),
                           chunks=(5, chunk_y, chunk_x))

# Ces lignes construisent le graphe — aucun calcul effectué
red  = raster[0]
nir  = raster[3]
ndvi = (nir - red) / (nir + red + 1e-10)
mean = ndvi.mean()
pct  = (ndvi > 0.35).mean() * 100

print(f"Graphe construit : {len(ndvi.__dask_graph__())} tâches")
print(f"ndvi  → {ndvi}")
print(f"mean  → {mean}")
print("→ Toujours aucun calcul")


# ── 3. .compute() déclenche tout ─────────────────────────────────────────────
print("\n=== 3. .compute() — exécution réelle ===\n")

# Calculer mean et pct en un seul passage (optimal)
t0 = time.perf_counter()
mean_val, pct_val = da.compute(mean, pct)
t_dask = time.perf_counter() - t0

# Visualiser le graphe de calcul (optionnel, peut être très grand)
# il faut disposer de graphviz et pydot pour cela
# mean.visualize(filename="graphe_mean.png", rankdir="LR")

print(f"NDVI moyen       : {mean_val:.4f}")
print(f"% végétation     : {pct_val:.1f}%")
print(f"Temps (threads)  : {t_dask:.3f}s")

# ── 4. Impact de la taille des chunks ────────────────────────────────────────
print("\n=== 4. Impact de la taille des chunks ===\n")
print(f"{'Chunk (px)':<12} {'MB/chunk':>10} {'Nb chunks':>12} {'Temps (s)':>10}")
print("-" * 48)

test = da.random.random((3000, 3000))
for size in [64, 256, 512, 1024]:
    arr   = test.rechunk((size, size))
    t0    = time.perf_counter()
    arr.mean().compute()
    t     = time.perf_counter() - t0
    mb    = size * size * 8 / 1e6
    n     = len(arr.chunks[0]) * len(arr.chunks[1])
    print(f"  {size:<10} {mb:>10.2f} {n:>12} {t:>10.3f}")

print("\n→ Trop petits : overhead de scheduling")
print("→ Trop grands : pression mémoire")
print("→ Règle : 10–200 MB par chunk (ici float64 → ~2–50 MB)")
