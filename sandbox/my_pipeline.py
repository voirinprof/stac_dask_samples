"""
=========================================
Pipeline : Série temporelle NDVI et détection de changement
------------------------------------------------------------
Objectif : construire de bout en bout un pipeline qui :
  1. Recherche des scènes Sentinel-2 sur deux périodes
  2. Charge les données en xarray lazy (odc-stac + Dask)
  3. Masque les nuages avec la couche SCL
  4. Calcule le NDVI pour chaque période
  5. Produit un composite par période
  6. Détecte les zones de changement significatif
  7. Génère une carte et un rapport de statistiques

Zone d'étude : Sherbrooke, Québec
Capteur      : Sentinel-2 L2A

"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import dask
import xarray as xr

from pystac_client import Client
import odc.stac

# ── Paramètres — ne pas modifier ─────────────────────────────────────────────
CATALOGUE_URL  = "https://earth-search.aws.element84.com/v1"
COLLECTION     = "sentinel-2-l2a"
BBOX           = [-72.15, 45.30, -71.65, 45.65]   # Sherbrooke
CRS            = "EPSG:32198"
RESOLUTION     = 10    # mètres
MAX_CLOUD      = 25    # % couverture nuageuse maximale
VALID_SCL      = [4, 5, 6, 7]   # végétation, sol, eau, non classifié
NDVI_THRESHOLD = 0.35  # seuil végétation
CHANGE_THRESHOLD = 0.10  # différence NDVI considérée comme un changement significatif

# Deux périodes à comparer
PERIODE_A = "2023-06-01/2023-08-31"   # été 2023
PERIODE_B = "2024-06-01/2024-08-31"   # été 2024

CHUNKS   = {"x": 512, "y": 512, "time": 1}
OUTPUT_DIR = Path(__file__).parent

## Fonction utilitaire pour trier les scènes par couverture nuageuse
def search_items(client, periode, max_cloud):
    """Rechercher et trier les scènes par couverture nuageuse."""
    items = sorted(
        client.search(
            collections=[COLLECTION],
            bbox=BBOX,
            datetime=periode,
            query={"eo:cloud_cover": {"lt": max_cloud}}
        ).items(),
        key=lambda i: i.properties.get("eo:cloud_cover", 100)
    )
    return items

## Fonction utilitaire pour charger un dataset xarray lazy depuis une liste d'Items
def load_dataset(items, label):
    """Charger une liste d'Items en Dataset xarray lazy."""
    ds = odc.stac.load(
        items,
        bands=["red", "nir", "scl"],
        crs=CRS,
        resolution=RESOLUTION,
        bbox=BBOX,
        chunks=CHUNKS,
        groupby="solar_day"   # fusionne les tuiles du même jour orbital
    )
    print(f"\nDataset {label} (lazy) :")
    print(f"  Dimensions : {dict(ds.dims)}")
    print(f"  Dates      : {[str(t)[:10] for t in ds.time.values]}")
    return ds

## fonction utilitaire pour appliquer le masque SCL et calculer le NDVI lazy
def compute_ndvi_masked(ds):
    """
    Appliquer le masque SCL et calculer le NDVI.
    Retourne un DataArray NDVI lazy avec NaN sur les pixels invalides.
    """
    # Masque valide : True = pixel sans nuage
    valid_mask = ds["scl"].isin(VALID_SCL)

    # Normaliser les valeurs S2 L2A (× 10 000 → 0–1)
    # et appliquer le masque en une seule opération
    red = (ds["red"].astype("float32") / 10000).where(valid_mask)
    nir = (ds["nir"].astype("float32") / 10000).where(valid_mask)

    # Calcul NDVI — reste entièrement lazy
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi

# Fonction utilitaire pour afficher les statistiques d'un composite NDVI
def print_composite_stats(composite, label, periode):
    """Afficher les statistiques d'un composite NDVI."""
    vals = composite.values
    print(f"\nComposite {label} ({periode}) :")
    print(f"  NDVI moyen   : {np.nanmean(vals):.4f}")
    print(f"  NDVI médian  : {np.nanmedian(vals):.4f}")
    pct = np.nanmean(vals > NDVI_THRESHOLD) * 100
    print(f"  % végétation (> {NDVI_THRESHOLD}) : {pct:.1f}%")
    pct_nan = np.isnan(vals).mean() * 100
    print(f"  % pixels NaN : {pct_nan:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Connexion et recherche STAC
# ══════════════════════════════════════════════════════════════════════════════
# Consigne : connectez-vous au catalogue STAC et recherchez les scènes
#            disponibles pour chacune des deux périodes.
#
# À compléter :
#   - Créer le client STAC
#   - Lancer une recherche pour PERIODE_A avec le filtre MAX_CLOUD
#   - Lancer une recherche pour PERIODE_B avec le filtre MAX_CLOUD
#   - Trier chaque liste par couverture nuageuse croissante
#   - Afficher le nombre de scènes trouvées et les 3 meilleures par période
#
# Indice : client.search(..., query={"eo:cloud_cover": {"lt": MAX_CLOUD}})

print("=" * 60)
print("ÉTAPE 1 — Recherche STAC")
print("=" * 60)

# Créer le client STAC
client = Client.open(CATALOGUE_URL)

# Rechercher les scènes pour chaque période
items_a = []           # TODO : recherche période A
items_b = []           # TODO : recherche période B


# Afficher le résumé des résultats
for label, items, periode in [("A", items_a, PERIODE_A),
                                ("B", items_b, PERIODE_B)]:
    print(f"\nPériode {label} ({periode}) : {len(items)} scènes")
    for item in items[:3]:
        date  = str(item.datetime.date())
        cloud = item.properties.get("eo:cloud_cover", "?")
        print(f"  {date} - {cloud:.1f}%  {item.id[:38]}")

if not items_a or not items_b:
    print("\n Aucune scène trouvée — vérifier les paramètres ou la connexion.")
    exit()


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Chargement odc-stac (lazy)
# ══════════════════════════════════════════════════════════════════════════════
# Consigne : chargez les deux séries de scènes en xarray Dataset lazy.
#            Utiliser les mêmes paramètres CRS/résolution/chunks pour
#            garantir que les deux datasets sont alignés spatialement.
#
# À compléter :
#   - Appeler odc.stac.load() pour items_a → ds_a
#   - Appeler odc.stac.load() pour items_b → ds_b
#   - Afficher les dimensions et les dates de chaque dataset
#   - Vérifier que les deux datasets ont le même CRS et la même résolution
#
# Indice : groupby="solar_day" fusionne les tuiles du même jour

print("\n" + "=" * 60)
print("ÉTAPE 2 — Chargement odc-stac")
print("=" * 60)

# charger les datasets pour la période A et B
ds_a = None   # TODO : charger les datasets pour la période A
ds_b = None   # TODO : charger les datasets pour la période B


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Masque nuages et calcul NDVI (lazy)
# ══════════════════════════════════════════════════════════════════════════════
# Consigne : pour chaque dataset, appliquer le masque SCL et calculer
#            le NDVI. Toutes ces opérations doivent rester LAZY
#            (aucun .compute() à cette étape).
#
# À compléter :
#   - Construire le masque valide depuis la bande "scl" pour ds_a et ds_b
#   - Normaliser les bandes red et nir (valeurs S2 × 10 000)
#   - Appliquer le masque (.where(valid_mask))
#   - Calculer le NDVI : (nir - red) / (nir + red + 1e-10)
#
# Rappel SCL :  4=végétation  5=sol  6=eau  7=non classifié
#               8/9=nuage     3=ombre nuage
# Indice : ds["scl"].isin(VALID_SCL) retourne un masque booléen

print("\n" + "=" * 60)
print("ÉTAPE 3 — Masque nuages et calcul NDVI (lazy)")
print("=" * 60)

# on doit calculer un NDVI lazy pour chaque période, avec NaN sur les pixels invalides
ndvi_a = None   # TODO : NDVI lazy pour la période A
ndvi_b = None   # TODO : NDVI lazy pour la période B

print("NDVI A lazy :", ndvi_a)
print("NDVI B lazy :", ndvi_b)

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Composite par période (.compute())
# ══════════════════════════════════════════════════════════════════════════════
# Consigne : produire un composite "best pixel" pour chaque période en
#            prenant le NDVI maximum sur la dimension temporelle.
#            C'est ici qu'on déclenche le calcul avec .compute().
#
# À compléter :
#   - Calculer le composite NDVI max pour la période A et B
#   - Calculer le nombre de scènes valides par pixel pour chaque période
#   - Utiliser dask.compute() pour déclencher les deux en un seul appel
#   - Afficher les statistiques du composite (min, max, mean, % végétation)
#
# Indice : ndvi.max(dim="time") donne le greenest pixel composite
# Indice : dask.compute(a, b, c) calcule plusieurs objets en une passe

print("\n" + "=" * 60)
print("ÉTAPE 4 — Composites par période (.compute())")
print("=" * 60)

# on doit créer une image composite par période
# avec un pixel ayant le NDVI maximum sur la dimension temporelle
composite_a  = None   # TODO : NDVI max période A
composite_b  = None   # TODO : NDVI max période B

# on doit aussi calculer le nombre de scènes valides par pixel
n_valid_a    = None   # TODO : nombre de scènes valides par pixel, période A
n_valid_b    = None   # TODO : nombre de scènes valides par pixel, période B
# ----------------------

# Un seul compute pour les quatre résultats — optimal
print("Calcul en cours (téléchargement + NDVI + max temporel)...")
composite_a, composite_b, n_valid_a, n_valid_b = dask.compute(
    composite_a,
    composite_b,
    n_valid_a,
    n_valid_b
)
print_composite_stats(composite_a, "A", PERIODE_A)
print_composite_stats(composite_b, "B", PERIODE_B)

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Détection de changement
# ══════════════════════════════════════════════════════════════════════════════
# Consigne : calculer la différence de NDVI entre les deux périodes et
#            classifier les changements en trois catégories.
#
# À compléter :
#   - Calculer la différence : delta = composite_b - composite_a
#   - Créer une carte de changement avec 3 classes :
#       +1  = gain de végétation  (delta > +CHANGE_THRESHOLD)
#        0  = stable              (|delta| <= CHANGE_THRESHOLD)
#       -1  = perte de végétation (delta < -CHANGE_THRESHOLD)
#   - Calculer le % de pixels dans chaque classe (ignorer les NaN)
#   - Afficher un rapport texte
#
# Indice : np.where(condition, valeur_si_vrai, valeur_si_faux)
# Indice : np.nansum() et np.isfinite() pour ignorer les NaN

print("\n" + "=" * 60)
print("ÉTAPE 5 — Détection de changement")
print("=" * 60)

# Statistiques à calculer et afficher :
# pct_gain   = % de pixels avec gain de végétation
# pct_stable = % de pixels stables
# pct_loss   = % de pixels avec perte de végétation

# Différence NDVI entre les deux périodes
# Un pixel est NaN si l'un des deux composites est NaN
delta = None   # TODO : différence composite_b - composite_a

# Classifier les changements en trois catégories
# +1 = gain de végétation, 0 = stable, -1 = perte de végétation
# on peut utiliser le principe de IF imbriqués avec np.where
change_map = None   # TODO : carte de changement (-1, 0, +1)


# Pixels invalides (NaN dans delta) → NaN dans la carte
change_map = change_map.astype("float32")
# pour éviter de compter les pixels invalides dans les stats, on met NaN dans change_map
change_map[np.isnan(delta)] = np.nan

# Calculer les statistiques sur les pixels valides seulement
valid_pixels = np.isfinite(change_map)
n_valid_total = valid_pixels.sum()

# Calculer le pourcentage de pixels dans chaque classe
pct_gain   = (change_map[valid_pixels] == 1).sum()  / n_valid_total * 100
pct_stable = (change_map[valid_pixels] == 0).sum()  / n_valid_total * 100
pct_loss   = (change_map[valid_pixels] == -1).sum() / n_valid_total * 100
delta_mean = np.nanmean(delta)

print(f"\nSeuil de changement : ±{CHANGE_THRESHOLD}")
print(f"\nRésultats :")
print(f"  Gain de végétation  : {pct_gain:.1f}%  "
      f"({(change_map[valid_pixels]==1).sum():,} pixels)")
print(f"  Stable              : {pct_stable:.1f}%  "
      f"({(change_map[valid_pixels]==0).sum():,} pixels)")
print(f"  Perte de végétation : {pct_loss:.1f}%  "
      f"({(change_map[valid_pixels]==-1).sum():,} pixels)")
print(f"  ΔNDVI moyen         : {delta_mean:+.4f}")

if abs(delta_mean) < 0.02:
    trend = "stable"
elif delta_mean > 0:
    trend = "légère augmentation de végétation"
else:
    trend = "légère diminution de végétation"
print(f"\nTendance générale : {trend}")

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Carte et rapport
# ══════════════════════════════════════════════════════════════════════════════
# Consigne : produire une figure avec 4 panneaux et imprimer un rapport
#            de synthèse dans le terminal.
#
# Figure attendue (4 panneaux) :
#   [0] NDVI composite — période A    (cmap RdYlGn, vmin=-0.1, vmax=0.8)
#   [1] NDVI composite — période B    (cmap RdYlGn, vmin=-0.1, vmax=0.8)
#   [2] Différence ΔNDVI (B - A)      (cmap RdBu,   vmin=-0.4, vmax=0.4)
#   [3] Carte de changement           (3 couleurs : rouge/gris/vert)
#
# Rapport terminal attendu :
#   - Période A : NDVI moyen, % végétation
#   - Période B : NDVI moyen, % végétation
#   - Changement : % gain, % stable, % perte, ΔNDVI moyen
#
# Indice : pour la carte de changement, utiliser une ListedColormap
#   cmap_change = mcolors.ListedColormap(["#d73027", "#ffffbf", "#1a9850"])
#   et norm = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap_change.N)

print("\n" + "=" * 60)
print("ÉTAPE 6 — Carte et rapport")
print("=" * 60)

# Rapport de synthèse
print("\n" + "─" * 50)
print("RAPPORT DE CHANGEMENT NDVI — Sherbrooke")
print("─" * 50)
print(f"Zone      : {BBOX}")
print(f"Période A : {PERIODE_A}")
print(f"Période B : {PERIODE_B}")
print(f"Capteur   : {COLLECTION}")
print(f"\nPériode A — NDVI moyen : {np.nanmean(composite_a.values):.4f} | "
      f"végétation : {np.nanmean(composite_a.values > NDVI_THRESHOLD)*100:.1f}%")
print(f"Période B — NDVI moyen : {np.nanmean(composite_b.values):.4f} | "
      f"végétation : {np.nanmean(composite_b.values > NDVI_THRESHOLD)*100:.1f}%")
print(f"\nChangement :")
print(f"  Gain   : {pct_gain:.1f}%")
print(f"  Stable : {pct_stable:.1f}%")
print(f"  Perte  : {pct_loss:.1f}%")
print(f"  ΔNDVI  : {delta_mean:+.4f}")
print("─" * 50)

# Carte 4 panneaux
fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")

# Colormap pour la carte de changement
cmap_change = mcolors.ListedColormap(["#d73027", "#ffffbf", "#1a9850"])
norm_change  = mcolors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap_change.N)

# Panneau 0 : NDVI composite A
im0 = axes[0, 0].imshow(composite_a.values, cmap="RdYlGn",
                          vmin=-0.1, vmax=0.8, interpolation="nearest")
plt.colorbar(im0, ax=axes[0, 0], shrink=0.7, label="NDVI")
axes[0, 0].set_title(f"NDVI composite — Période A\n{PERIODE_A}", fontsize=10)
axes[0, 0].set_axis_off()

# Panneau 1 : NDVI composite B
im1 = axes[0, 1].imshow(composite_b.values, cmap="RdYlGn",
                          vmin=-0.1, vmax=0.8, interpolation="nearest")
plt.colorbar(im1, ax=axes[0, 1], shrink=0.7, label="NDVI")
axes[0, 1].set_title(f"NDVI composite — Période B\n{PERIODE_B}", fontsize=10)
axes[0, 1].set_axis_off()

# Panneau 2 : Différence ΔNDVI
im2 = axes[1, 0].imshow(delta, cmap="RdBu",
                          vmin=-0.4, vmax=0.4, interpolation="nearest")
plt.colorbar(im2, ax=axes[1, 0], shrink=0.7, label="ΔNDVI (B − A)")
axes[1, 0].set_title(f"Différence NDVI (B − A)\nΔNDVI moyen : {delta_mean:+.4f}",
                      fontsize=10)
axes[1, 0].set_axis_off()

# Panneau 3 : Carte de changement
im3 = axes[1, 1].imshow(change_map, cmap=cmap_change,
                          norm=norm_change, interpolation="nearest")
cbar = plt.colorbar(im3, ax=axes[1, 1], shrink=0.7,
                     ticks=[-1, 0, 1])
cbar.set_ticklabels([f"Perte ({pct_loss:.0f}%)",
                      f"Stable ({pct_stable:.0f}%)",
                      f"Gain ({pct_gain:.0f}%)"])
axes[1, 1].set_title(f"Carte de changement (seuil ±{CHANGE_THRESHOLD})\n"
                      f"Rouge=perte | Jaune=stable | Vert=gain", fontsize=10)
axes[1, 1].set_axis_off()

fig.suptitle(
    f"Détection de changement NDVI — Sherbrooke\n"
    f"Sentinel-2 L2A | {PERIODE_A}  vs  {PERIODE_B} | CRS : {CRS}",
    fontsize=11, fontweight="bold"
)

output_path = OUTPUT_DIR / "exercice_changement.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nCarte sauvegardée : {output_path.name}")
plt.show()