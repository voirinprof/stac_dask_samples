# STAC, odc-stac, Dask et Big Data géospatial

## Présentation

Ce repo contient **10 scripts Python** couvrant les notions
essentielles de la recherche de données
satellite jusqu'au pipeline d'analyse complet.

Tous les exemples utilisent **Sentinel-2** sur **Sherbrooke**.
Les paramètres (zone, période, résolution, etc.) sont centralisés
dans `config.yaml` (modifier ce fichier suffit pour adapter
l'ensemble du repo à une autre zone ou période).

---

## Installation

```bash
conda env create -f environment.yml
conda activate gmq580-s05
# ou
pip install -r requirements.txt
```

> Les scripts `01_stac/`, `02_odc_stac/` et `05_pipeline/`
> nécessitent une connexion internet pour accéder aux catalogues
> STAC publics (Element84 / AWS).

> Il peut arriver sous Windows que la lib rasterio soit mal installée
> Vous allez avoir une erreur avec le DLL (comme *ImportError: DLL load failed while importing _rasterio: The specified module could not be found.*)
> Vous pouvez essayer ces 2 options :
```bash
conda install -c conda-forge rasterio
# ou
pip install --force-reinstall rasterio
```
---

## Structure

```
gmq580-seance05-simple/
│
├── config.yaml          ← paramètres partagés par tous les scripts
├── config_loader.py     ← utilitaire : charge config.yaml
│
├── 01_stac/
│   ├── 01_explorer_catalogue.py
│   ├── 02_rechercher_scenes.py
│   └── 03_inspecter_item.py
│
├── 02_odc_stac/
│   ├── 01_charger_xarray.py
│   └── 02_masque_nuages_scl.py
│
├── 03_dask/
│   ├── 01_lazy_et_compute.py
│   └── 02_schedulers.py
│
├── 04_xarray_dask/
│   ├── 01_serie_temporelle.py
│   └── 02_zarr_format.py
│
└── 05_pipeline/
    └── 01_pipeline_ndvi.py
```

---

## Description des scripts

### `config.yaml`
Fichier de configuration central. Contient la zone d'étude
(bbox Sherbrooke), la période, l'URL du catalogue STAC,
les bandes à charger, les paramètres Dask et les seuils d'analyse.
Tous les scripts le lisent via `config_loader.py` — il suffit
de modifier ce fichier pour changer de zone ou de période
sans toucher au code.

---

### 01 — Explorer un catalogue STAC
**`01_stac/01_explorer_catalogue.py`**

Point d'entrée dans l'écosystème STAC. Montre comment se
connecter à un catalogue public (Element84), lister toutes
les collections disponibles, et inspecter les métadonnées
d'une collection spécifique (Sentinel-2). À exécuter en
premier pour comprendre ce que le catalogue offre avant
de lancer une recherche.

---

### 02 — Rechercher et filtrer des scènes
**`01_stac/02_rechercher_scenes.py`**

Illustre la recherche par zone géographique (bbox), période
et filtre de couverture nuageuse. Montre comment trier les
résultats par qualité, afficher un tableau des scènes disponibles,
et mettre les Items en cache local (JSON) pour éviter de
re-requêter le catalogue à chaque exécution.

---

### 03 — Inspecter un Item STAC
**`01_stac/03_inspecter_item.py`**

Détaille toute la structure d'un Item STAC : géométrie GeoJSON,
propriétés (métadonnées de la scène), assets (fichiers associés).
Montre comment accéder directement aux URLs des bandes spectrales
et comment valider la conformité d'un Item au standard STAC.
Utile pour comprendre ce qu'on manipule avant de charger les données.

---

### 04 — Charger des scènes en xarray avec odc-stac
**`02_odc_stac/01_charger_xarray.py`**

Montre la valeur ajoutée d'odc-stac : passer d'une liste d'Items
STAC à un xarray Dataset aligné, reprojeté et découpé en une
seule ligne. Explique les paramètres clés (`crs`, `resolution`,
`chunks`, `groupby`), insiste sur le fait que le Dataset est
**lazy** (rien n'est téléchargé), et illustre comment charger
une seule date avec `.isel(time=0).compute()`.

---

### 05 — Masquer les nuages avec la couche SCL
**`02_odc_stac/02_masque_nuages_scl.py`**

La Scene Classification Layer (SCL) de Sentinel-2 attribue
une classe à chaque pixel (végétation, sol, eau, nuage, ombre…).
Ce script montre comment lire la SCL, construire un masque
valide, appliquer ce masque avant le calcul du NDVI, et
visualiser l'effet du masquage sur les résultats. Sans ce
masque, les nuages faussent systématiquement les statistiques.

---

### 06 — Lazy evaluation et .compute()
**`03_dask/01_lazy_et_compute.py`**

Script fondamental pour comprendre Dask. Illustre concrètement
la différence entre NumPy (calcul immédiat) et Dask (graphe de
tâches différé). Montre que les opérations Dask ne font que
construire un graphe, que `.compute()` est le déclencheur unique,
et comment grouper plusieurs `.compute()` en un seul appel pour
économiser du temps. Inclut un benchmark sur l'impact de la
taille des chunks.

---

### 07 — Schedulers et dashboard
**`03_dask/02_schedulers.py`**

Compare les trois modes d'exécution (synchrone, threads,
processus) et montre quand utiliser chacun. Introduit
`dask.distributed` pour créer un cluster local avec dashboard
(`localhost:8787`). Explique l'utilité de `client.persist()`
pour éviter de recalculer un résultat intermédiaire utilisé
plusieurs fois. Le dashboard est l'outil de diagnostic
indispensable pour tout calcul qui dépasse quelques secondes.

---

### 08 — Construire et analyser une série temporelle xarray
**`04_xarray_dask/01_serie_temporelle.py`**

Construit un DataArray xarray avec dimensions nommées (`time`,
`y`, `x`) et Dask comme backend. Illustre les opérations
temporelles essentielles : moyenne annuelle, greenest pixel
composite (`max(dim="time")`), variabilité (`std`), analyse
saisonnière (`groupby("time.season")`), et extraction d'un
profil temporel pour un pixel donné. Toutes les opérations
sont lazy jusqu'au `xr.compute()` final.

---

### 09 — Sauvegarder et lire en format Zarr
**`04_xarray_dask/02_zarr_format.py`**

Présente Zarr comme le format natif du big data raster :
chunké, compressé (Blosc/LZ4), et compatible cloud sans
modification de code. Compare la taille sur disque avec NetCDF,
benchmark les temps d'accès pour une tuile spatiale vs une
série temporelle d'un pixel, et inspecte la structure interne
d'un fichier Zarr. Montre que la syntaxe est identique pour
un Zarr local et un Zarr sur S3.

---

### 10 — Pipeline complet STAC → odc-stac → Dask → NDVI
**`05_pipeline/01_pipeline_ndvi.py`**

Script de synthèse qui assemble toutes les notions du cours
en un pipeline de bout en bout, commenté étape par étape.
Produit un composite NDVI "greenest pixel" sur Sherbrooke
à partir des scènes Sentinel-2 disponibles, sauvegarde
le résultat en Zarr et génère une carte PNG. Le point clé :
les étapes 2 à 4 sont entièrement **lazy** — un seul
`.compute()` à l'étape 5 déclenche téléchargement, masquage
et calcul simultanément.

---

## Ordre d'exécution suggéré

```
01_stac/01_explorer_catalogue.py      ← comprendre le catalogue
01_stac/02_rechercher_scenes.py       ← trouver des scènes
01_stac/03_inspecter_item.py          ← inspecter un Item
02_odc_stac/01_charger_xarray.py      ← charger en xarray
02_odc_stac/02_masque_nuages_scl.py   ← masquer les nuages
03_dask/01_lazy_et_compute.py         ← comprendre Dask
03_dask/02_schedulers.py              ← schedulers + dashboard
04_xarray_dask/01_serie_temporelle.py ← série temporelle
04_xarray_dask/02_zarr_format.py      ← format Zarr
05_pipeline/01_pipeline_ndvi.py       ← pipeline complet ★
```

Le script `05_pipeline/01_pipeline_ndvi.py` peut être exécuté
seul — il reprend toutes les étapes de façon autonome.