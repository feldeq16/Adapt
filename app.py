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

DOSSIER = "Donnéest"
FICHIER_DEFINITIONS = "name.txt"

# ============================================
# 2. CHARGEMENT ET TRAITEMENT
# ============================================

def lire_descriptions_variables(path):
    """Lit le fichier name.txt pour extraire les descriptions"""
    desc = {}
    if not os.path.exists(path):
        return desc
    try:
        with open(path, "r", encoding="utf-8") as f: # ou latin-1 selon votre fichier
            for line in f:
                if ":" in line:
                    parts = line.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    desc[key] = val
    except Exception as e:
        st.warning(f"Impossible de lire les descriptions : {e}")
    return desc

def lire_fichier_safe(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
    except:
        return None

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier):
        return None, None

    all_dfs = []
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    # 1. Lecture
    for f in os.listdir(dossier):
        if not f.endswith(".txt"): continue
        
        df = lire_fichier_safe(os.path.join(dossier, f))
        if df is None: continue

        # Nettoyage colonnes
        df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
        df.columns = [c.strip() for c in df.columns]

        # Conversion numérique forcée
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

    # 3. Calcul des échelles globales (Min/Max par variable)
    numeric_vars = [c for c in final_df.columns if c not in id_cols + latlon_cols and pd.api.types.is_numeric_dtype(final_df[c])]
    
    global_scales = {}
    for v in numeric_vars:
        vmin = final_df[v].min()
        vmax = final_df[v].max()
        global_scales[v] = (vmin, vmax)

    return final_df, global_scales

# ============================================
# 3. LOGIQUE APP
# ============================================

# Chargement des données et des descriptions
data, echelles_globales = charger_donnees_globales(DOSSIER)
descriptions = lire_descriptions_variables(FICHIER_DEFINITIONS)

if data is None:
    st.error("❌ Aucune donnée trouvée. Vérifiez le dossier 'Données'.")
    st.stop()

# --- TABLEAU RÉCAPITULATIF ---
with st.expander("📊 Disponibilité des variables par Scénario", expanded=False):
    dispo = data.groupby("Contexte").count()
    vars_cols = [c for c in dispo.columns if c in echelles_globales.keys()]
    dispo = dispo[vars_cols].T
    
    # Ajout d'une colonne avec la description lisible
    dispo["Définition"] = [descriptions.get(idx, "") for idx in dispo.index]
    
    # Réorganisation pour mettre la définition au début
    cols = ["Définition"] + [c for c in dispo.columns if c != "Définition"]
    dispo = dispo[cols]

    dispo_clean = dispo.applymap(lambda x: "✅" if x == "✅" else ("✅" if isinstance(x, int) and x > 0 else (x if isinstance(x, str) else "❌")))
    st.dataframe(dispo_clean)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    # Choix Variable AVEC DESCRIPTION
    variables_dispos = sorted(list(echelles_globales.keys()))
    if not variables_dispos:
        st.error("Aucune variable numérique détectée.")
        st.stop()
    
    # Fonction pour afficher le nom propre dans le menu
    def format_func_var(option):
        desc = descriptions.get(option, "")
        if desc:
            # On tronque si c'est trop long pour la sidebar
            return f"{option} - {desc[:40]}..."
        return option

    choix_var = st.selectbox("Variable à analyser", variables_dispos, format_func=format_func_var)
    
    # Affichage de la description complète sous le sélecteur
    if choix_var in descriptions:
        st.info(f"**{choix_var}** : {descriptions[choix_var]}")
    
    st.divider()
    
    # Choix Scénario & Horizon
    scenarios = sorted(data["Contexte"].unique())
    choix_scenario = st.selectbox("Scénario (RCP)", scenarios)
    
    df_step1 = data[data["Contexte"] == choix_scenario]
    horizons = sorted(df_step1["Période"].unique())
    choix_horizon = st.selectbox("Période / Horizon", horizons)
    
    st.divider()
    
    # Style Carte
    st.subheader("🎨 Apparence")
    styles_map = {
        "Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
    }
    style_choisi = st.selectbox("Fond de carte", list(styles_map.keys()))
    
    # Légende Globale Fixe CENTRÉE SUR LE BLANC
    vmin_glob, vmax_glob = echelles_globales[choix_var]
    
    # Calcul du point milieu pour le blanc
    vcenter = (vmin_glob + vmax_glob) / 2
    
    # TwoSlopeNorm permet de forcer le blanc (valeur médiane de la colormap) à une valeur précise (vcenter)
    # Ici, coolwarm va du bleu (min) au blanc (center) au rouge (max)
    norm_fixe = mcolors.TwoSlopeNorm(vmin=vmin_glob, vcenter=vcenter, vmax=vmax_glob)

    st.caption(f"Échelle fixe : {choix_var}")
    
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(4, 0.4))
    
    # On applique la norme centrée à la légende
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_fixe, cmap=cmap), cax=ax, orientation='horizontal')
    cb.outline.set_visible(False)
    ax.set_axis_off()
    st.pyplot(fig)
    st.write(f"Min: **{vmin_glob:.2f}** | Moy: **{vcenter:.2f}** | Max: **{vmax_glob:.2f}**")

# --- PRÉPARATION DONNÉES CARTE ---

df_map = df_step1[df_step1["Période"] == choix_horizon].copy()

# Sécurité : Vérifier si la variable existe pour cette sélection
if choix_var not in df_map.columns or df_map[choix_var].isna().all():
    st.warning(f"⚠️ Donnée indisponible : La variable **{choix_var}** n'existe pas pour {choix_scenario} / {choix_horizon}.")
    st.stop()

# Nettoyage des NaN pour la carte
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
    adr = st.text_input("📍 Rechercher une localisation", placeholder="Ex: Toulouse, France")
    u_lat, u_lon = None, None
    if adr:
        u_lat, u_lon = geocode_safe(adr)
        if not u_lat:
            st.warning("Adresse introuvable.")

with col_kpi:
    avg_val = df_map[choix_var].mean()
    st.metric(f"Moyenne Nationale ({choix_scenario})", f"{avg_val:.2f}")

# --- RENDU CARTE (PIXELS) ---

# Application des couleurs selon l'échelle GLOBALE et CENTRÉE
# On réutilise 'norm_fixe' calculé plus haut
rgb = (cmap(norm_fixe(df_map[choix_var].values))[:, :3] * 255).astype(int)
df_map["r"], df_map["g"], df_map["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

layers = []

# Calque Pixels (8km)
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
    tooltip={"html": f"<b>{choix_var}:</b> {{{choix_var}}}<br><i>(Station: {{Point}})</i>"}
))

# --- ANALYSE LOCALE ---

if u_lat:
    st.divider()
    st.subheader("🔍 Analyse Locale")
    
    # 1. Calcul distances
    df_map["dist_km"] = df_map.apply(
        lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
    )
    
    voisins = df_map.nsmallest(5, "dist_km")
    
    col_g, col_d = st.columns(2)
    
    with col_g:
        st.info("📍 Pixel le plus proche")
        proche = voisins.iloc[0]
        st.write(f"**Identifiant :** {proche['Point']}")
        st.write(f"**Distance :** {proche['dist_km']:.2f} km")
        st.metric(f"Valeur brute", f"{proche[choix_var]:.2f}")

    with col_d:
        st.success("🧮 Estimation Interpolée")
        weights = 1 / (voisins["dist_km"] + 0.01)**2
        val_est = np.sum(voisins[choix_var] * weights) / np.sum(weights)
        st.metric(f"Valeur pondérée", f"{val_est:.2f}")

    st.write("---")
    st.write("**Détail des données utilisées :**")
    
    cols_to_show = ["Point", choix_var, "dist_km"]
    st.dataframe(voisins[cols_to_show].style.format({choix_var: "{:.2f}", "dist_km": "{:.2f} km"}))
