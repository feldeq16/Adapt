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

# ============================================
# 2. CHARGEMENT ET TRAITEMENT (Avec Métadonnées)
# ============================================

def lire_metadonnees_et_data(path):
    """
    Lit un fichier pour extraire :
    1. Les descriptions des variables (lignes commençant par #)
    2. Le DataFrame des données
    """
    description_map = {}
    try:
        # 1. Lecture des commentaires (Méta-données)
        # On utilise latin-1 car c'est souvent l'encodage par défaut des fichiers Météo-France/Windows
        # Si vos accents ne s'affichent pas bien, essayez 'utf-8'
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    # On cherche le pattern "CODE : Description"
                    if ":" in line:
                        parts = line.replace("#", "").split(":", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip()
                            description_map[key] = val
                else:
                    # Dès qu'on tombe sur une ligne sans #, c'est que les données commencent
                    break
        
        # 2. Lecture des données
        df = pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True, encoding='latin-1')
        return df, description_map
    except Exception as e:
        # st.warning(f"Erreur de lecture sur {path}: {e}")
        return None, {}

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier):
        return None, None, {}

    all_dfs = []
    global_descriptions = {} # Pour stocker les définitions (NORTAV : Température...)
    
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    # 1. Lecture
    for f in os.listdir(dossier):
        if not f.endswith(".txt"): continue
        
        df, metas = lire_metadonnees_et_data(os.path.join(dossier, f))
        if df is None: continue
        
        # On met à jour le dictionnaire global des descriptions
        global_descriptions.update(metas)

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

    if not all_dfs: return None, None, {}

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

    return final_df, global_scales, global_descriptions

# ============================================
# 3. LOGIQUE APP
# ============================================

data, echelles_globales, descriptions = charger_donnees_globales(DOSSIER)

if data is None:
    st.error("❌ Aucune donnée trouvée. Vérifiez le dossier 'Données'.")
    st.stop()

# --- TABLEAU RÉCAPITULATIF ---
with st.expander("📊 Disponibilité des variables (Tableau de synthèse)", expanded=False):
    dispo = data.groupby("Contexte").count()
    vars_cols = [c for c in dispo.columns if c in echelles_globales.keys()]
    dispo = dispo[vars_cols].T 
    dispo_clean = dispo.applymap(lambda x: "✅" if x > 0 else "❌")
    
    # Ajout d'une colonne description dans le tableau récap
    dispo_clean.insert(0, "Description", [descriptions.get(idx, "") for idx in dispo_clean.index])
    
    st.dataframe(dispo_clean)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    # Choix Variable AVEC DESCRIPTION
    variables_dispos = sorted(list(echelles_globales.keys()))
    if not variables_dispos:
        st.error("Aucune variable numérique détectée.")
        st.stop()
    
    # Fonction de formatage pour afficher "CODE : Description" dans le menu
    def format_variable(code):
        desc = descriptions.get(code, "Description inconnue")
        # On coupe si c'est trop long pour la sidebar
        if len(desc) > 50: desc = desc[:47] + "..."
        return f"{code} - {desc}"

    choix_var = st.selectbox("Variable à analyser", variables_dispos, format_func=format_variable)
    
    # Affichage de la description complète juste en dessous pour être sûr
    st.info(f"**Définition :** {descriptions.get(choix_var, 'Pas de description disponible')}")

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
    
    # Légende Globale Fixe & Centrée sur le Blanc
    vmin_glob, vmax_glob = echelles_globales[choix_var]
    
    # CALCUL DU CENTRE EXACT POUR QUE LE BLANC SOIT AU MILIEU
    milieu = (vmin_glob + vmax_glob) / 2
    
    # Utilisation de TwoSlopeNorm pour forcer le centre
    # Cela garantit que la couleur blanche est exactement à 'milieu'
    norm_legend = mcolors.TwoSlopeNorm(vmin=vmin_glob, vcenter=milieu, vmax=vmax_glob)
    
    st.caption(f"Échelle : {choix_var}")
    
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(4, 0.4))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_legend, cmap=cmap), cax=ax, orientation='horizontal')
    cb.outline.set_visible(False)
    ax.set_axis_off()
    st.pyplot(fig)
    st.write(f"Min: **{vmin_glob:.2f}** | Milieu (Blanc): **{milieu:.2f}** | Max: **{vmax_glob:.2f}**")

# --- PRÉPARATION DONNÉES CARTE ---

df_map = df_step1[df_step1["Période"] == choix_horizon].copy()

# Sécurité Variable
if choix_var not in df_map.columns or df_map[choix_var].isna().all():
    st.warning(f"⚠️ Donnée indisponible : La variable **{choix_var}** n'existe pas pour {choix_scenario} / {choix_horizon}.")
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
    adr = st.text_input("📍 Rechercher une localisation", placeholder="Ex: Toulouse, France")
    u_lat, u_lon = None, None
    if adr:
        u_lat, u_lon = geocode_safe(adr)
        if not u_lat: st.warning("Adresse introuvable.")

with col_kpi:
    avg_val = df_map[choix_var].mean()
    st.metric(f"Moyenne Nationale ({choix_scenario})", f"{avg_val:.2f}")

# --- RENDU CARTE ---

# Application des couleurs avec le gradient centré
# TwoSlopeNorm permet d'assurer que vcenter est le point blanc
norm = mcolors.TwoSlopeNorm(vmin=vmin_glob, vcenter=milieu, vmax=vmax_glob)
rgb = (cmap(norm(df_map[choix_var].values))[:, :3] * 255).astype(int)
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
    tooltip={"html": f"<b>{choix_var}:</b> {{{choix_var}}}<br><i>Station: {{Point}}</i>"}
))

# --- ANALYSE LOCALE ---

if u_lat:
    st.divider()
    st.subheader("🔍 Analyse Locale")
    
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
        st.metric(f"Valeur réelle", f"{proche[choix_var]:.2f}")

    with col_d:
        st.success("🧮 Estimation Interpolée")
        weights = 1 / (voisins["dist_km"] + 0.01)**2
        val_est = np.sum(voisins[choix_var] * weights) / np.sum(weights)
        st.metric(f"Valeur estimée", f"{val_est:.2f}")

    st.write("---")
    st.write("**Détail des données utilisées :**")
    
    cols_to_show = ["Point", choix_var, "dist_km"]
    st.dataframe(voisins[cols_to_show].style.format({choix_var: "{:.2f}", "dist_km": "{:.2f} km"}))
