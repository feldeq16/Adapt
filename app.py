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
# CONFIGURATION
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
        skip_blank_lines=True,
        encoding='utf-8' # Force l'encodage (ou 'latin-1' selon vos fichiers)
    )

@st.cache_data
def charger_donnees(dossier):
    all_dfs = []
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    if not os.path.exists(dossier):
        return None

    for f in os.listdir(dossier):
        if not f.endswith(".txt"):
            continue

        try:
            df = lire_fichier(os.path.join(dossier, f))
            # Nettoyage colonnes
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = [c.strip() for c in df.columns]

            # Conversion numérique
            for c in df.columns:
                if c in latlon_cols:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
                elif c not in id_cols:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
            
            all_dfs.append(df)
        except Exception as e:
            st.error(f"Erreur lecture {f}: {e}")
            continue

    if not all_dfs:
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Agrégation (first pour les coords, mean pour les valeurs si doublons)
    agg_dict = {col: "first" for col in combined_df.columns if col not in id_cols}
    final_df = combined_df.groupby(id_cols, as_index=False).agg(agg_dict)

    return final_df

# ============================================
# DONNÉES
# ============================================

data = charger_donnees(DOSSIER)

if data is None:
    st.error("Aucune donnée chargée.")
    st.stop()

# ============================================
# BARRE LATÉRALE (FILTRES & STYLE)
# ============================================

with st.sidebar:
    st.header("🎛️ Filtres Données")

    # 1. Filtres Données
    scenario = st.selectbox("Scénario", sorted(data["Contexte"].dropna().unique()))
    df1 = data[data["Contexte"] == scenario]

    horizon = st.selectbox("Horizon", sorted(df1["Période"].dropna().unique()))
    df2 = df1[df1["Période"] == horizon]

    meta = ["Latitude", "Longitude", "Point", "Contexte", "Période"]
    variables = [c for c in df2.columns if c not in meta and pd.api.types.is_numeric_dtype(df2[c])]

    var = st.selectbox("Variable", variables)

    # Nettoyage final pour la carte
    df2 = df2.dropna(subset=["Latitude", "Longitude", var])
    
    # Calcul des bornes pour la couleur
    vmin = df2[var].quantile(0.02)
    vmax = df2[var].quantile(0.98)

    st.divider()

    # 2. Filtres Carte
    st.header("🗺️ Fond de Carte")
    styles_map = {
        "Clair (Light)": "mapbox://styles/mapbox/light-v9",
        "Sombre (Dark)": "mapbox://styles/mapbox/dark-v9",
        "Satellite": "mapbox://styles/mapbox/satellite-v9",
        "Outdoors": "mapbox://styles/mapbox/outdoors-v11",
        "Rues": "mapbox://styles/mapbox/streets-v11"
    }
    
    style_choisi = st.selectbox("Style :", list(styles_map.keys()))
    map_style_url = styles_map[style_choisi]
    
    # Légende
    st.write(f"**Légende : {var}**")
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(4, 0.4))
    norm_legend = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_legend, cmap=cmap), cax=ax, orientation='horizontal')
    cb.outline.set_visible(False)
    ax.set_axis_off()
    st.pyplot(fig)

# ============================================
# GÉOCODAGE (Nominatim Gratuit)
# ============================================

@st.cache_data(show_spinner=False)
def geocode(address):
    try:
        # Agent unique pour éviter blocage
        agent = f"app_clim_{uuid.uuid4()}"
        geolocator = Nominatim(user_agent=agent, timeout=5)
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
    except:
        pass
    return None, None

adr = st.text_input("🔍 Rechercher une adresse", placeholder="Ex: Place Bellecour, Lyon")
u_lat, u_lon = None, None

if adr:
    u_lat, u_lon = geocode(adr)
    if u_lat:
        st.success(f"📍 Localisé : {u_lat:.4f}, {u_lon:.4f}")
    else:
        st.warning("Adresse introuvable.")

# ============================================
# PRÉPARATION COULEURS & CARTE
# ============================================

# Calcul des couleurs RGB
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
rgb = (cmap(norm(df2[var].values))[:, :3] * 255).astype(int)

df2 = df2.copy()
df2["r"], df2["g"], df2["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

layers = []

# 1. CALQUE PRINCIPAL : PIXELS CARRÉS (GridCellLayer)
# C'est ce calque qui remplace les points par des carrés de 8km
pixel_layer = pdk.Layer(
    "GridCellLayer",
    data=df2,
    get_position="[Longitude, Latitude]",
    get_color="[r, g, b, 180]", # 180 = Transparence pour voir le fond de carte
    cell_size=8000,             # 8000 mètres = 8km de côté
    extruded=False,             # False = Plat (Pixel), True = 3D (Histogramme)
    pickable=True,
    auto_highlight=True,
)
layers.append(pixel_layer)

# 2. CALQUE UTILISATEUR (PIN)
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
    view_state = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=9)
else:
    view_state = pdk.ViewState(
        latitude=df2["Latitude"].mean(), 
        longitude=df2["Longitude"].mean(), 
        zoom=5.5
    )

# RENDU DE LA CARTE
st.pydeck_chart(
    pdk.Deck(
        map_style=map_style_url, # Fond de carte dynamique
        layers=layers,
        initial_view_state=view_state,
        tooltip={"html": f"<b>Station:</b> {{Point}}<br><b>{var}:</b> {{{var}}}"},
    )
)

# ============================================
# TABLEAUX & INTERPOLATION
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
        st.subheader("📍 Point de grille le plus proche")
        plus_proche = voisins.iloc[0]
        st.info(f"Station : {plus_proche['Point']} (à {plus_proche['dist_km']:.1f} km)")
        
        # Petit tableau propre
        val_reelle = plus_proche[var]
        st.metric(f"Valeur réelle ({var})", f"{val_reelle:.2f}")

    with c2:
        st.subheader("🧮 Estimation Interpolée")
        
        # Interpolation IDW (Inverse Distance Weighting)
        weights = 1 / (voisins["dist_km"] + 0.01) ** 2
        
        # On ne calcule que pour la variable sélectionnée pour aller vite
        vals = voisins[var].values
        val_estimee = np.sum(vals * weights) / np.sum(weights)
        
        st.success(f"Pour votre adresse exacte")
        st.metric(f"Estimation ({var})", f"{val_estimee:.2f}")
    
    # Tableau de détail
    st.caption("Détail des 5 points de grille utilisés pour le calcul :")
    cols_show = ["Point", var, "dist_km", "Latitude", "Longitude"]
    st.dataframe(voisins[cols_show].style.format({var: "{:.2f}", "dist_km": "{:.2f} km"}))
