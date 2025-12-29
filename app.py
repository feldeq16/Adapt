import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import uuid
import re
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# ============================================
# 1. CONFIGURATION ET STYLE
# ============================================
st.set_page_config(layout="wide", page_title="Observatoire Climatique", page_icon="🌍")

st.markdown("""
<style>
    /* Style pour le Dashboard */
    .block-container {padding-top: 2rem;}
    
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        color: #333;
    }

    h4 {
        margin-top: 30px;
        padding-bottom: 5px;
        border-bottom: 2px solid #f0f2f6;
        color: #0068c9;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Observatoire Climatique Multi-Scénarios")
st.markdown("---")

DOSSIER = "Données"
FICHIER_DEFINITIONS = "name.txt"
FICHIER_CATEGORIES = "category.txt"

# ============================================
# 2. FONCTIONS DE TRAITEMENT
# ============================================

def lire_dict_fichier(path):
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

def extraire_unite(description):
    if not description: return ""
    match = re.search(r"\((.*?)\)$", description.strip())
    return match.group(1) if match else ""

def nettoyer_nom_variable(description):
    if not description: return ""
    return re.sub(r"\s*\(.*?\)$", "", description).strip()

def lire_fichier_data(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
    except:
        return None

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier): return None, None, None

    all_dfs = []
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    for f in os.listdir(dossier):
        if not f.endswith(".txt"): continue
        df = lire_fichier_data(os.path.join(dossier, f))
        if df is None: continue

        df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
        df.columns = [c.strip() for c in df.columns]

        for c in df.columns:
            if c in latlon_cols:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
            elif c not in id_cols:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
        all_dfs.append(df)

    if not all_dfs: return None, None, None

    combined = pd.concat(all_dfs, ignore_index=True)
    agg_dict = {c: "first" for c in combined.columns if c not in id_cols}
    final_df = combined.groupby(id_cols, as_index=False).agg(agg_dict)

    numeric_vars = [c for c in final_df.columns if c not in id_cols + latlon_cols and pd.api.types.is_numeric_dtype(final_df[c])]
    
    global_scales = {}
    for v in numeric_vars:
        vmin = final_df[v].min()
        vmax = final_df[v].max()
        global_scales[v] = (vmin, vmax)

    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")

    return final_df, global_scales, national_means

# ============================================
# 3. LOGIQUE APP
# ============================================

data, echelles_globales, national_means = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
categories = lire_dict_fichier(FICHIER_CATEGORIES)

if data is None:
    st.error("❌ Aucune donnée trouvée.")
    st.stop()

def format_func_var(option):
    desc = descriptions.get(option, "")
    if desc: return f"{option} - {desc[:45]}..."
    return option

# ============================================
# 4. SIDEBAR
# ============================================
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    # Filtres
    liste_cats = sorted(list(set(categories.values())))
    if liste_cats:
        liste_cats.insert(0, "Toutes les catégories")
        choix_cat = st.selectbox("Filtrer par thème", liste_cats)
    else:
        choix_cat = "Toutes les catégories"

    variables_dispos = sorted(list(echelles_globales.keys()))
    if choix_cat != "Toutes les catégories":
        variables_dispos = [v for v in variables_dispos if categories.get(v) == choix_cat]

    if not variables_dispos:
        st.warning("Aucune variable.")
        st.stop()

    choix_var = st.selectbox("Variable à analyser", variables_dispos, format_func=format_func_var)
    
    st.divider()
    
    scenarios = sorted(data["Contexte"].unique())
    choix_scenario = st.selectbox("Scénario (RCP)", scenarios)
    
    df_step1 = data[data["Contexte"] == choix_scenario]
    horizons = sorted(df_step1["Période"].unique())
    choix_horizon = st.selectbox("Période / Horizon", horizons)
    
    st.divider()
    
    styles_map = {
        "Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
    }
    style_choisi = st.selectbox("Fond de carte", list(styles_map.keys()))

# ============================================
# 5. PRÉPARATION ET COULEURS (LOGIQUE CORRIGÉE)
# ============================================

df_map = df_step1[df_step1["Période"] == choix_horizon].copy()

if choix_var not in df_map.columns:
    st.warning("Variable indisponible.")
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
    adr = st.text_input("📍 Rechercher une adresse", placeholder="Ex: Place Bellecour, Lyon")
    u_lat, u_lon = None, None
    if adr:
        u_lat, u_lon = geocode_safe(adr)
        if not u_lat: st.warning("Adresse introuvable.")

# Infos Variables
moyenne_nat = national_means[choix_scenario].get(choix_var, 0)
desc_brute = descriptions.get(choix_var, choix_var)
unit_var = extraire_unite(desc_brute)

with col_kpi:
    st.metric(f"Moyenne France ({choix_scenario})", f"{moyenne_nat:.2f} {unit_var}")

# --- COULEURS ROBUSTES ---
vmin_glob, vmax_glob = echelles_globales[choix_var]

# LOGIQUE DE COULEUR SANS ERREUR
if vmin_glob >= 0:
    # 1. Tout POSITIF (0 à 30) -> Blanc à Rouge
    norm = mcolors.Normalize(vmin=0, vmax=vmax_glob)
    cmap = plt.get_cmap("Reds") # Blanc -> Rouge
    # Légende : de 0 à Max

elif vmax_glob <= 0:
    # 2. Tout NÉGATIF (-30 à 0) -> Bleu à Blanc
    norm = mcolors.Normalize(vmin=vmin_glob, vmax=0)
    cmap = plt.get_cmap("Blues_r") # Bleu -> Blanc
    # Légende : de Min à 0

else:
    # 3. MIXTE (-10 à 20) -> Bleu à Blanc à Rouge
    # Ici TwoSlopeNorm est sûr de fonctionner car vmin < 0 < vmax
    norm = mcolors.TwoSlopeNorm(vmin=vmin_glob, vcenter=0, vmax=vmax_glob)
    cmap = plt.get_cmap("coolwarm")

# Application
rgb = (cmap(norm(df_map[choix_var].values))[:, :3] * 255).astype(int)
df_map["r"], df_map["g"], df_map["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

# --- CARTE ---
layers = [
    pdk.Layer(
        "GridCellLayer",
        data=df_map,
        get_position="[Longitude, Latitude]",
        get_color="[r, g, b, 170]",
        cell_size=8000,
        extruded=False,
        pickable=True,
        auto_highlight=True
    )
]

if u_lat:
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame({"lat": [u_lat], "lon": [u_lon]}),
        get_position="[lon, lat]",
        get_color="[0, 255, 0]",
        get_radius=5000,
        stroked=True,
        get_line_color=[0,0,0],
        line_width_min_pixels=3
    ))
    view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=9)
else:
    view = pdk.ViewState(latitude=46.6, longitude=2.0, zoom=5.5)

st.pydeck_chart(pdk.Deck(
    map_style=styles_map[style_choisi],
    initial_view_state=view,
    layers=layers,
    tooltip={"html": f"<b>{choix_var}</b>: {{{choix_var}}} {unit_var}<br><i>{{Point}}</i>"}
))

# --- LÉGENDE ADAPTÉE ---
col_leg1, col_leg2, col_leg3 = st.columns([1, 6, 1])
with col_leg2:
    st.caption(f"Légende : {nettoyer_nom_variable(desc_brute)}")
    
    fig, ax = plt.subplots(figsize=(10, 0.4))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb.outline.set_visible(False)
    ax.set_axis_off()
    st.pyplot(fig)
    
    # Indicateurs sous la légende selon le cas
    c1, c2, c3 = st.columns(3)
    
    if vmin_glob >= 0:
        # Positif : 0 -> Max
        c1.markdown("<div style='text-align: left'><b>0</b></div>", unsafe_allow_html=True)
        c3.markdown(f"<div style='text-align: right'><b>{vmax_glob:.1f}</b></div>", unsafe_allow_html=True)
    elif vmax_glob <= 0:
        # Négatif : Min -> 0
        c1.markdown(f"<div style='text-align: left'><b>{vmin_glob:.1f}</b></div>", unsafe_allow_html=True)
        c3.markdown("<div style='text-align: right'><b>0</b></div>", unsafe_allow_html=True)
    else:
        # Mixte : Min -> 0 -> Max
        c1.markdown(f"<div style='text-align: left'><b>{vmin_glob:.1f}</b></div>", unsafe_allow_html=True)
        c2.markdown("<div style='text-align: center'><b>0</b></div>", unsafe_allow_html=True)
        c3.markdown(f"<div style='text-align: right'><b>{vmax_glob:.1f}</b></div>", unsafe_allow_html=True)

# ============================================
# 6. DASHBOARD ANALYTIQUE
# ============================================

if u_lat:
    st.divider()
    st.subheader(f"📍 Tableau de Bord : {adr}")
    
    # 1. Calculs Voisins
    df_map["dist_km"] = df_map.apply(
        lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
    )
    voisins = df_map.nsmallest(5, "dist_km")
    
    weights = 1 / (voisins["dist_km"] + 0.01)**2
    val_focus = np.sum(voisins[choix_var] * weights) / np.sum(weights)

    # 2. FOCUS VARIABLE
    delta_focus = val_focus - moyenne_nat
    col_f1, col_f2 = st.columns([2, 1])
    
    with col_f1:
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 25px; border-radius: 12px; border-left: 5px solid #0068c9;">
            <h3 style="margin:0; color:#31333F;">{nettoyer_nom_variable(desc_brute)}</h3>
            <h1 style="margin:10px 0; font-size: 3.5em; color:#0068c9;">{val_focus:.2f} <span style="font-size: 0.4em; color:#666;">{unit_var}</span></h1>
            <p style="margin:0; color:#555; font-size: 1.1em;">
                Moyenne Nationale : <b>{moyenne_nat:.2f}</b> {unit_var} 
                <span style="color: {'green' if delta_focus < 0 else 'red'}; margin-left: 10px;">
                    ({delta_focus:+.2f})
                </span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_f2:
        st.info("ℹ️ Données calculées par interpolation des 5 stations climatiques les plus proches.")
        st.caption(f"Station la plus proche : **{voisins.iloc[0]['Point']}** ({voisins.iloc[0]['dist_km']:.1f} km)")

    # 3. GRILLE DE DASHBOARD PAR CATÉGORIE
    st.markdown("### 📊 Indicateurs Détaillés")
    
    pixel_ref = voisins.iloc[0]
    all_vars = [c for c in pixel_ref.index if c in echelles_globales]
    
    grouped_vars = {}
    for var in all_vars:
        cat = categories.get(var, "Autres Indicateurs")
        if cat not in grouped_vars: grouped_vars[cat] = []
        grouped_vars[cat].append(var)
    
    # Tri des catégories
    sorted_cats = sorted(grouped_vars.keys())
    if "Autres Indicateurs" in sorted_cats:
        sorted_cats.remove("Autres Indicateurs")
        sorted_cats.append("Autres Indicateurs")

    for cat in sorted_cats:
        st.markdown(f"#### {cat}")
        
        vars_in_cat = grouped_vars[cat]
        
        # Grid layout (3 colonnes)
        cols = st.columns(3)
        
        for i, var_code in enumerate(vars_in_cat):
            # Valeurs
            val_locale = pixel_ref[var_code]
            mean_nat = national_means[choix_scenario].get(var_code, 0)
            
            # Textes
            full_desc = descriptions.get(var_code, var_code)
            clean_name = nettoyer_nom_variable(full_desc)
            if len(clean_name) > 35: clean_name = clean_name[:32] + "..."
            unit = extraire_unite(full_desc)
            
            delta = val_locale - mean_nat
            
            # Affichage Metric
            with cols[i % 3]:
                st.metric(
                    label=clean_name,
                    value=f"{val_locale:.2f} {unit}",
                    delta=f"{delta:+.2f} vs Moy. Nat.",
                    delta_color="inverse" # Rouge si ça augmente, Vert si ça baisse (généralement mieux pour le climat)
                )
