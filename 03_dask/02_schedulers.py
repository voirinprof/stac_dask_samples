"""
Script 07 : Schedulers et dashboard Dask
------------------------------------------
Objectif : comprendre les différents modes d'exécution
           de Dask (synchrone, threads, processus, distribué)
           et utiliser le dashboard pour surveiller les calculs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import dask.array as da
import dask
import time
from config_loader import load_config

cfg    = load_config()
CHUNKS = (cfg["dask"]["chunk_y"], cfg["dask"]["chunk_x"])

if __name__ == "__main__":
    # Tableau de test
    arr = da.random.random((4000, 4000), chunks=CHUNKS)
    calcul = (arr * 2).mean()

    # ── 1. Comparer les schedulers ────────────────────────────────────────────────
    print("=== 1. Comparaison des schedulers ===\n")

    schedulers = {
        "synchronous": "1 tâche à la fois — idéal pour déboguer",
        "threads":     "multi-thread — défaut pour dask.array (NumPy libère le GIL)",
        "processes":   "multi-process — contourne le GIL, utile pour code Python pur",
    }

    for name, desc in schedulers.items():
        t0 = time.perf_counter()
        with dask.config.set(scheduler=name):
            result = calcul.compute()
        t = time.perf_counter() - t0
        print(f"  {name:15s} : {t:.3f}s — {desc}")

    # ── 2. Scheduler distribué avec dashboard ────────────────────────────────────
    print("\n=== 2. Scheduler distribué ===\n")

    try:
        from dask.distributed import Client as DaskClient

        # Créer un cluster local — le dashboard est à http://localhost:8787
        client = DaskClient(n_workers=4, threads_per_worker=2, processes=False, memory_limit="2GB")

        print("Cluster démarré :")
        print(f"  Workers   : {len(client.scheduler_info()['workers'])}")
        print(f"  Dashboard : {client.dashboard_link}")
        print(f"\n  → Ouvrir {client.dashboard_link} pour voir les tâches en temps réel")
        input("\n  Appuyer sur Entrée pour lancer le calcul...")

        # Simuler un calcul coûteux sur un grand raster
        big_raster = da.random.random((5, 4000, 4000), chunks=(1, 512, 512))
        red         = big_raster[0]
        nir         = big_raster[3]
        ndvi        = (nir - red) / (nir + red + 1e-10)

        # persist() : lancer en arrière-plan, garder en mémoire des workers
        # Utile quand on réutilise plusieurs fois le même résultat intermédiaire
        print("\n  Calcul NDVI (persist)...")
        t0          = time.perf_counter()
        ndvi_future = client.persist(ndvi)

        # Les trois stats réutilisent ndvi_future sans le recalculer
        mean_f = client.compute(ndvi_future.mean())
        std_f  = client.compute(ndvi_future.std())
        pct_f  = client.compute((ndvi_future > 0.35).mean())

        print(f"  NDVI moyen   : {mean_f.result():.4f}")
        print(f"  NDVI std     : {std_f.result():.4f}")
        print(f"  % végétation : {pct_f.result()*100:.1f}%")
        print(f"  Temps total  : {time.perf_counter()-t0:.2f}s")

        client.close()
        print("\n  Cluster fermé.")

    except ImportError:
        print("  dask.distributed non installé → pip install dask[distributed]")
    except Exception as e:
        print(f"  Cluster non disponible : {e}")
        print("  → Utiliser le scheduler 'threads' par défaut")

    # ── 3. Configurer Dask globalement ────────────────────────────────────────────
    print("\n=== 3. Configuration globale ===\n")

    # Modifier temporairement avec un context manager
    with dask.config.set({"array.chunk-size": "128MiB", "scheduler": "threads"}):
        print("Dans le contexte :")
        print(f"  chunk-size : {dask.config.get('array.chunk-size')}")
        print(f"  scheduler  : {dask.config.get('scheduler')}")

    print("\nHors du contexte :")
    print(f"  scheduler  : {dask.config.get('scheduler', 'synchronous (défaut)')}")
