import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import uuid
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# ============================================
# CONFIG
# ============================================

st.set_page_config(layout="wide", page_title="Climat Multi-Scénarios")
st.title("🌍 Analyse climatique : Grille 8km")

DOSSIER = "Données"

# ============================================
# LECTURE ET AGRÉGATION
# ============================================

def lire_fichier(path):
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        comment="#",
        skip_blank_lines=True
    )

@st.cache_data
def charger_donnees(dossier):
    """
    Agrège tous les fichiers du dossier dans une table unique.
    Identifiant unique : Point, Contexte, Période.
    """
    if not os.path.exists(dossier):
        return None

    all_dfs = []
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    for f in os.listdir(dossier):
        if not f.endswith(".txt"):
            continue

        try:
            df = lire_fichier(os.path.join(dossier, f))
            # Suppression colonnes parasites
            cols_unmatch = [c for c in df.columns if "Unnamed" in c]
            df = df.drop(columns=cols_unmatch)
            df.columns = [c.strip() for c in df.columns]

            # Nettoyage et conversion numérique
            for c in df.columns:
                if c in latlon_cols:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
                elif c not in id_cols:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
            
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Erreur lecture fichier {f}: {e}")
            continue

    if not all_dfs:
        return None

    # 1. On empile
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 2. On agrège (Lat/Lon ne changent pas par point, on prend la moyenne des valeurs si doublons)
    agg_dict = {col: "first" for col in combined_df.columns if col not in id_cols}
    final_df = combined_df.groupby(id_cols, as_index=False).agg(agg_dict)

    return final_df

# ============================================
# CHARGEMENT DONNÉES
# ============================================

data = charger_donnees(DOSSIER)

if data is None:
    st.error("Aucune donnée trouvée.")
    st.stop()

# ============================================
# FILTRES
# ============================================

with st.sidebar:
    st.header("🎛️ Filtres")

    scenario = st.selectbox("Scénario", sorted(data["Contexte"].dropna().unique()))
    df1 = data[data["Contexte"] == scenario]

    horizon = st.selectbox("Horizon", sorted(df1["Période"].dropna().unique()))
    df2 = df1[df1["Période"] == horizon]

    meta = ["Latitude", "Longitude", "Point", "Contexte", "Période"]
    # On ne garde que les colonnes numériques
    variables = [c for c in df2.columns if c not in meta and pd.api.types.is_numeric_dtype(df2[c])]

    if not variables:
        st.error("Pas de variables numériques.")
        st.stop()

    var = st.selectbox("Variable", variables)

    # Nettoyage final pour la carte
    df2 = df2.dropna(subset=["Latitude", "Longitude", var])

    # Calcul des bornes (Quantiles pour éviter que les extrêmes écrasent les couleurs)
    vmin = df2[var].quantile(0.02)
    vmax = df2[var].quantile(0.98)
    
    st.divider()
    
    # Légende
    st.write(f"**Légende : {var}**")
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(4, 0.4))
    norm_legend = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_legend, cmap=cmap), cax=ax, orientation='horizontal')
    ax.set_axis_off()
    st.pyplot(fig)

# ============================================
# GÉOCODAGE ROBUSTE (NOMINATIM)
# ============================================

@st.cache_data(show_spinner=False)
def geocode(address):
    """Utilise Nominatim avec un User-Agent unique pour ne pas être bloqué"""
    try:
        agent = f"app_clim_pixel_{uuid.uuid4()}"
        geolocator = Nominatim(user_agent=agent, timeout=5)
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except:
        return None, None
    return None, None

col_s, col_b = st.columns([3, 1])
with col_s:
    adr = st.text_input("🔍 Adresse", placeholder="Ex: Toulouse, France")

u_lat, u_lon = None, None
if adr:
    u_lat, u_lon = geocode(adr)
    if u_lat:
        st.success(f"📍 Localisé : {u_lat:.4f}, {u_lon:.4f}")
    else:
        st.warning("Adresse introuvable.")

# ============================================
# PRÉPARATION COULEURS
# ============================================

cmap = plt.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
# On convertit en 0-255
rgb = (cmap(norm(df2[var].values))[:, :3] * 255).astype(int)

df2 = df2.copy()
df2["r"], df2["g"], df2["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

# ============================================
# CARTE PIXELS (GRID CELL LAYER)
# ============================================


layers = []

# 1. Le calque de données en PIXELS
# GridCellLayer est très performant pour dessiner des carrés
pixel_layer = pdk.Layer(
    "GridCellLayer",
    data=df2,
    get_position="[Longitude, Latitude]",
    get_color="[r, g, b, 170]", # 170 = Transparence pour voir le fond
    cell_size=8000,             # 8000 mètres = 8km
    extruded=False,             # False = Plat (Pixel 2D), True = 3D. False est plus performant.
    pickable=True,
    auto_highlight=True,
)
layers.append(pixel_layer)

# 2. Le calque utilisateur (Gros point vert)
if u_lat:
    user_layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame({"lat": [u_lat], "lon": [u_lon]}),
        get_position="[lon, lat]",
        get_color="[0, 255, 0]",
        get_radius=5000,
        stroked=True,
        get_line_color=[0,0,0],
        line_width_min_pixels=3
    )
    layers.append(user_layer)
    view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=9)
else:
    view = pdk.ViewState(
        latitude=df2["Latitude"].mean(), 
        longitude=df2["Longitude"].mean(), 
        zoom=5.5
    )

st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9", # Fond clair propre
        layers=layers,
        initial_view_state=view,
        tooltip={"html": f"<b>Station:</b> {{Point}}<br><b>{var}:</b> {{{var}}}"},
    )
)

# ============================================
# TABLEAUX + INTERPOLATION
# ============================================
if u_lat:
    st.divider()
    
    # Calcul distance
    df2["dist_km"] = df2.apply(
        lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
    )

    voisins = df2.nsmallest(5, "dist_km")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📍 Pixel le plus proche")
        station = voisins.iloc[0]
        st.info(f"ID: {station['Point']} (à {station['dist_km']:.1f} km)")
        st.metric(f"Valeur réelle ({var})", f"{station[var]:.2f}")

    with c2:
        st.subheader("🧮 Interpolation locale")
        # Inverse Distance Weighting
        W = 1 / (voisins["dist_km"] + 0.01) ** 2
        vals = voisins[var].values
        
        # On calcule l'estimation pondérée
        val_est = np.sum(vals * W.values) / np.sum(W.values)
        
        st.success(f"Estimation pour votre adresse")
        st.metric(f"Valeur interpolée ({var})", f"{val_est:.2f}")

    st.caption("Données sources (5 pixels les plus proches) :")
    cols_show = ["Point", var, "dist_km"]
    st.dataframe(voisins[cols_show])
