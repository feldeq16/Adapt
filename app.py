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
print("--- Démarrage du script ---") # REGARDEZ LE TERMINAL
st.set_page_config(layout="wide", page_title="Climat Multi-Scénarios")
st.title("🌍 Analyse climatique : Grille 8km")

DOSSIER = "Données"

# ============================================
# LECTURE ET AGRÉGATION
# ============================================

def lire_fichier(path):
    # Lecture optimisée : on ne lit que les colonnes utiles si possible
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        comment="#",
        skip_blank_lines=True,
        encoding='latin-1' # Souvent plus sûr pour les accents français
    )

@st.cache_data
def charger_donnees(dossier):
    print("Début chargement données...")
    all_dfs = []
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    if not os.path.exists(dossier):
        return None

    fichiers = [f for f in os.listdir(dossier) if f.endswith(".txt")]
    print(f"{len(fichiers)} fichiers trouvés.")

    # LIMITATION DE SÉCURITÉ (A enlever plus tard)
    # On ne lit que les 5 premiers fichiers pour voir si ça débloque l'affichage
    # fichiers = fichiers[:5] 

    for i, f in enumerate(fichiers):
        if i % 10 == 0: print(f"Lecture fichier {i}/{len(fichiers)}...") # Pour voir l'avancement
        
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
            print(f"Erreur sur {f}: {e}")
            continue

    if not all_dfs:
        return None

    print("Concaténation...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print("Groupby...")
    agg_dict = {col: "first" for col in combined_df.columns if col not in id_cols}
    final_df = combined_df.groupby(id_cols, as_index=False).agg(agg_dict)
    
    print("Chargement terminé.")
    return final_df

# ============================================
# DONNÉES
# ============================================

data = charger_donnees(DOSSIER)
st.dataframe(data)

if data is None:
    st.error("Aucune donnée chargée. Vérifiez le dossier.")
    st.stop()

# ============================================
# BARRE LATÉRALE (FILTRES & STYLE)
# ============================================

with st.sidebar:
    st.header("🎛️ Filtres Données")

    # Filtres
    scenario = st.selectbox("Scénario", sorted(data["Contexte"].dropna().unique()))
    df1 = data[data["Contexte"] == scenario]

    horizon = st.selectbox("Horizon", sorted(df1["Période"].dropna().unique()))
    df2 = df1[df1["Période"] == horizon]

    meta = ["Latitude", "Longitude", "Point", "Contexte", "Période"]
    variables = [c for c in df2.columns if c not in meta and pd.api.types.is_numeric_dtype(df2[c])]

    if not variables:
        st.error("Pas de variables numériques trouvées.")
        st.stop()

    var = st.selectbox("Variable", variables)

    # Nettoyage
    df2 = df2.dropna(subset=["Latitude", "Longitude", var])
    
    # Bornes
    vmin = df2[var].quantile(0.02)
    vmax = df2[var].quantile(0.98)

    st.divider()

    # Style Carte
    st.header("🗺️ Fond de Carte")
    styles_map = {
        "Clair (Light)": "mapbox://styles/mapbox/light-v9",
        "Sombre (Dark)": "mapbox://styles/mapbox/dark-v9",
        "Satellite": "mapbox://styles/mapbox/satellite-v9",
        "Outdoors": "mapbox://styles/mapbox/outdoors-v11",
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
# GÉOCODAGE
# ============================================

@st.cache_data(show_spinner=False)
def geocode(address):
    try:
        agent = f"app_clim_{uuid.uuid4()}"
        geolocator = Nominatim(user_agent=agent, timeout=2) # Timeout court pour ne pas bloquer
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
# CARTE
# ============================================

# Couleurs
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
rgb = (cmap(norm(df2[var].values))[:, :3] * 255).astype(int)

df2 = df2.copy()
df2["r"], df2["g"], df2["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

layers = []

# OPTIMISATION : Si plus de 10 000 points, on passe en Scatterplot (points)
# Sinon on reste en GridCell (carrés) car c'est trop lourd pour le navigateur
if len(df2) > 10000:
    st.caption("⚠️ Mode 'Points' activé (trop de données pour les carrés)")
    pixel_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df2,
        get_position="[Longitude, Latitude]",
        get_color="[r, g, b, 180]",
        get_radius=3000,
        pickable=True
    )
else:
    pixel_layer = pdk.Layer(
        "GridCellLayer",
        data=df2,
        get_position="[Longitude, Latitude]",
        get_color="[r, g, b, 180]",
        cell_size=8000,
        extruded=False,
        pickable=True
    )
layers.append(pixel_layer)

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

st.pydeck_chart(
    pdk.Deck(
        map_style=map_style_url,
        layers=layers,
        initial_view_state=view_state,
        tooltip={"html": f"<b>Station:</b> {{Point}}<br><b>{var}:</b> {{{var}}}"},
    )
)

# ============================================
# TABLEAUX
# ============================================

if u_lat:
    st.divider()
    df2["dist_km"] = df2.apply(lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1)
    voisins = df2.nsmallest(5, "dist_km")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📍 Station la plus proche")
        plus_proche = voisins.iloc[0]
        st.info(f"Station : {plus_proche['Point']} (à {plus_proche['dist_km']:.1f} km)")
        val_reelle = plus_proche[var]
        st.metric(f"Valeur réelle ({var})", f"{val_reelle:.2f}")

    with c2:
        st.subheader("🧮 Estimation Interpolée")
        weights = 1 / (voisins["dist_km"] + 0.01) ** 2
        vals = voisins[var].values
        val_estimee = np.sum(vals * weights) / np.sum(weights)
        st.success(f"Pour votre adresse exacte")
        st.metric(f"Estimation ({var})", f"{val_estimee:.2f}")
    
    st.caption("Détail des points utilisés :")
    cols_show = ["Point", var, "dist_km"]
    st.dataframe(voisins[cols_show])
