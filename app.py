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
# 1. CONFIGURATION
# ============================================
st.set_page_config(layout="wide", page_title="Observatoire Climatique", page_icon="🌍")

st.title("🌍 Observatoire Climatique Multi-Scénarios")
st.markdown("---")

DOSSIER = "Données"
FICHIER_DEFINITIONS = "name.txt"
FICHIER_CATEGORIES = "category.txt"

# ============================================
# 2. FONCTIONS DE LECTURE & CHARGEMENT
# ============================================

def lire_dict_fichier(path):
    """Lit un fichier clé:valeur (name.txt ou category.txt)"""
    d = {}
    if not os.path.exists(path): return d
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    d[key] = val
    except: pass
    return d

def lire_fichier_data(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
    except:
        return None

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier): return None, None

    all_dfs = []
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    # 1. Lecture des fichiers
    for f in os.listdir(dossier):
        if not f.endswith(".txt"): continue
        
        df = lire_fichier_data(os.path.join(dossier, f))
        if df is None: continue

        # Nettoyage
        df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
        df.columns = [c.strip() for c in df.columns]

        # Conversion numérique
        for c in df.columns:
            if c in latlon_cols:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
            elif c not in id_cols:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
        
        all_dfs.append(df)

    if not all_dfs: return None, None

    # 2. Agrégation
    combined = pd.concat(all_dfs, ignore_index=True)
    agg_dict = {c: "first" for c in combined.columns if c not in id_cols}
    final_df = combined.groupby(id_cols, as_index=False).agg(agg_dict)

    # 3. Calcul Echelles Globales
    numeric_vars = [c for c in final_df.columns if c not in id_cols + latlon_cols and pd.api.types.is_numeric_dtype(final_df[c])]
    
    global_scales = {}
    for v in numeric_vars:
        vmin = final_df[v].min()
        vmax = final_df[v].max()
        global_scales[v] = (vmin, vmax)

    return final_df, global_scales

# ============================================
# 3. CHARGEMENT INITIAL
# ============================================

data, echelles_globales = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
categories = lire_dict_fichier(FICHIER_CATEGORIES)

if data is None:
    st.error("❌ Aucune donnée trouvée. Vérifiez le dossier 'Données'.")
    st.stop()

# Fonction helper pour l'affichage dans le selectbox
def format_func_var(option):
    desc = descriptions.get(option, "")
    if desc:
        return f"{option} - {desc[:50]}..."
    return option

# ============================================
# 4. SIDEBAR : FILTRES
# ============================================
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    # 1. Filtre par Catégorie
    liste_cats = sorted(list(set(categories.values())))
    if liste_cats:
        liste_cats.insert(0, "Toutes les catégories")
        choix_cat = st.selectbox("Filtrer par thème", liste_cats)
    else:
        choix_cat = "Toutes les catégories"

    # 2. Filtre Variable
    variables_dispos = sorted(list(echelles_globales.keys()))
    
    # Application du filtre catégorie
    if choix_cat != "Toutes les catégories":
        variables_dispos = [v for v in variables_dispos if categories.get(v) == choix_cat]

    if not variables_dispos:
        st.warning("Aucune variable pour cette catégorie.")
        st.stop()

    choix_var = st.selectbox("Variable à analyser", variables_dispos, format_func=format_func_var)
    
    # Info bulle description
    if choix_var in descriptions:
        st.info(f"**Définition :** {descriptions[choix_var]}")
    
    st.divider()
    
    # 3. Scénario & Horizon
    scenarios = sorted(data["Contexte"].unique())
    choix_scenario = st.selectbox("Scénario (RCP)", scenarios)
    
    df_step1 = data[data["Contexte"] == choix_scenario]
    horizons = sorted(df_step1["Période"].unique())
    choix_horizon = st.selectbox("Période / Horizon", horizons)
    
    st.divider()
    
    # 4. Style
    styles_map = {
        "Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
    }
    style_choisi = st.selectbox("Fond de carte", list(styles_map.keys()))

# ============================================
# 5. PRÉPARATION CARTE
# ============================================

df_map = df_step1[df_step1["Période"] == choix_horizon].copy()

# Sécurité Variable
if choix_var not in df_map.columns or df_map[choix_var].isna().all():
    st.warning(f"⚠️ Variable **{choix_var}** indisponible pour {choix_scenario} / {choix_horizon}.")
    st.stop()

df_map = df_map.dropna(subset=["Latitude", "Longitude", choix_var])

# --- GÉOCODAGE ---
@st.cache_data(show_spinner=False)
def geocode_safe(address):
    try:
        agent = f"app_climat_{uuid.uuid4()}"
        geolocator = Nominatim(user_agent=agent, timeout=3)
        loc = geolocator.geocode(address)
        if loc: return loc.latitude, loc.longitude
    except: pass
    return None, None

col_search, col_kpi = st.columns([2, 1])

with col_search:
    adr = st.text_input("📍 Rechercher une adresse", placeholder="Ex: Toulouse, France")
    u_lat, u_lon = None, None
    if adr:
        u_lat, u_lon = geocode_safe(adr)
        if not u_lat: st.warning("Adresse introuvable.")

with col_kpi:
    avg_val = df_map[choix_var].mean()
    st.metric(f"Moyenne Nationale ({choix_scenario})", f"{avg_val:.2f}")

# --- COULEURS (Centrées sur Blanc) ---
vmin_glob, vmax_glob = echelles_globales[choix_var]
vcenter = (vmin_glob + vmax_glob) / 2
norm_fixe = mcolors.TwoSlopeNorm(vmin=vmin_glob, vcenter=vcenter, vmax=vmax_glob)
cmap = plt.get_cmap("coolwarm")

rgb = (cmap(norm_fixe(df_map[choix_var].values))[:, :3] * 255).astype(int)
df_map["r"], df_map["g"], df_map["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

# --- AFFICHAGE CARTE ---
layers = []

# Calque Pixels
grid_layer = pdk.Layer(
    "GridCellLayer",
    data=df_map,
    get_position="[Longitude, Latitude]",
    get_color="[r, g, b, 170]",
    cell_size=8000,
    extruded=False,
    pickable=True,
    auto_highlight=True
)
layers.append(grid_layer)

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
    view_state = pdk.ViewState(latitude=46.6, longitude=2.0, zoom=5.5)

st.pydeck_chart(pdk.Deck(
    map_style=styles_map[style_choisi],
    initial_view_state=view_state,
    layers=layers,
    tooltip={"html": f"<b>{choix_var}:</b> {{{choix_var}}}<br><i>Station: {{Point}}</i>"}
))

# --- LÉGENDE (SOUS LA CARTE) ---
col_leg1, col_leg2, col_leg3 = st.columns([1, 6, 1])
with col_leg2:
    st.caption(f"Échelle : {choix_var} (Min: {vmin_glob:.1f} | Blanc: {vcenter:.1f} | Max: {vmax_glob:.1f})")
    fig, ax = plt.subplots(figsize=(10, 0.5))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_fixe, cmap=cmap), cax=ax, orientation='horizontal')
    cb.outline.set_visible(False)
    ax.set_axis_off()
    st.pyplot(fig)

# ============================================
# 6. ANALYSE DÉTAILLÉE
# ============================================

if u_lat:
    st.divider()
    st.subheader("🔍 Analyse Complète du Pixel")
    
    # 1. Pixel le plus proche
    df_map["dist_km"] = df_map.apply(
        lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
    )
    # On prend le pixel le plus proche (le premier)
    pixel_cible = df_map.nsmallest(1, "dist_km").iloc[0]
    
    st.success(f"📍 Données pour la station la plus proche : **{pixel_cible['Point']}** (à {pixel_cible['dist_km']:.2f} km)")
    
    # 2. Préparation du tableau complet des variables
    # On récupère toutes les colonnes numériques qui sont dans echelles_globales
    vars_to_show = [c for c in pixel_cible.index if c in echelles_globales]
    
    # Création d'un DataFrame propre pour l'affichage
    data_list = []
    for v in vars_to_show:
        cat = categories.get(v, "Autre")
        desc = descriptions.get(v, v)
        val = pixel_cible[v]
        data_list.append({"Catégorie": cat, "Code": v, "Description": desc, "Valeur": val})
    
    df_detail = pd.DataFrame(data_list)
    
    # Tri par catégorie puis par code
    df_detail = df_detail.sort_values(by=["Catégorie", "Code"])
    
    # 3. Affichage Interactif
    # On permet de filtrer le tableau
    filtre_cat_tab = st.selectbox("Filtrer le tableau par catégorie", ["Tout"] + sorted(list(set(categories.values()))), key="filtre_tab")
    
    if filtre_cat_tab != "Tout":
        df_final_tab = df_detail[df_detail["Catégorie"] == filtre_cat_tab]
    else:
        df_final_tab = df_detail
    
    # Affichage du tableau stylisé
    st.dataframe(
        df_final_tab.style.format({"Valeur": "{:.2f}"})
        .background_gradient(subset=["Valeur"], cmap="coolwarm")
    )
