import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import plotly.express as px
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import uuid
import re
import requests
from geopy.distance import geodesic

# ============================================
# 1. CONFIGURATION ET STYLE
# ============================================
st.set_page_config(layout="wide", page_title="Observatoire Climatique", page_icon="🌍")

st.markdown("""
<style>
    .stApp { color: #31333F; }
    
    /* Style des Metrics du Dashboard */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #4b5563 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-size: 1.6rem !important;
    }

    /* Style des Onglets */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        color: #374151 !important;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0068c9 !important;
        border: 1px solid #0068c9 !important;
        font-weight: bold;
    }
    
    /* Titres des sous-catégories dans le dashboard */
    .subcat-header {
        color: #6b7280;
        font-size: 1.1rem;
        margin-top: 10px;
        border-left: 3px solid #0068c9;
        padding-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Observatoire Climatique")
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
                    d[parts[0].strip()] = parts[1].strip()
    except: pass
    return d

def lire_dict_categories_double(path):
    d = {}
    if not os.path.exists(path): return d
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    if "|" in val:
                        theme, dtype = val.split("|")
                        d[key.strip()] = {"theme": theme.strip(), "type": dtype.strip()}
                    else:
                        d[key.strip()] = {"theme": val.strip(), "type": "Valeurs"}
    except: pass
    return d

def extraire_unite(description):
    match = re.search(r"\((.*?)\)$", description.strip())
    return match.group(1) if match else ""

def nettoyer_nom_variable(description):
    return re.sub(r"\s*\(.*?\)$", "", description).strip()

def format_func_clean(option):
    return descriptions.get(option, option)

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier): return None, None, None
    all_dfs = []
    id_cols = ["Point", "Contexte", "Période"]
    for f in os.listdir(dossier):
        if not f.endswith(".txt"): continue
        df = pd.read_csv(os.path.join(dossier, f), sep=None, engine="python", comment="#")
        df.columns = [c.strip() for c in df.columns]
        for c in df.columns:
            if c not in id_cols:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True)
    agg_dict = {c: "first" for c in combined.columns if c not in id_cols}
    final_df = combined.groupby(id_cols, as_index=False).agg(agg_dict)
    
    numeric_vars = [c for c in final_df.columns if c not in id_cols + ["Latitude", "Longitude"]]
    global_scales = {v: (final_df[v].min(), final_df[v].max()) for v in numeric_vars}
    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")
    return final_df, global_scales, national_means

# ============================================
# 3. CHARGEMENT ET SIDEBAR
# ============================================
data, echelles_globales, national_means = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
cat_dict = lire_dict_categories_double(FICHIER_CATEGORIES)

with st.sidebar:
    st.header("🎛️ Paramètres")
    adr_input = st.text_input("📍 Rechercher une adresse", placeholder="Ex: Lyon")
    u_lat, u_lon, adr_full = None, None, None
    if adr_input and len(adr_input) > 2:
        try:
            r = requests.get(f"https://api-adresse.data.gouv.fr/search/?q={adr_input}&limit=1").json()
            if r['features']:
                u_lon, u_lat = r['features'][0]['geometry']['coordinates']
                adr_full = r['features'][0]['properties']['label']
                st.success(f"Trouvé : {adr_full}")
        except: pass

    st.divider()
    choix_scenario = st.selectbox("Scénario RCP", sorted(data["Contexte"].unique()))
    df_step = data[data["Contexte"] == choix_scenario]
    choix_horizon = st.selectbox("Période", sorted(df_step["Période"].unique()))
    df_map = df_step[df_step["Période"] == choix_horizon].copy()

# ============================================
# 4. ONGLETS
# ============================================
tab1, tab2, tab3 = st.tabs(["🗺️ Carte", "📊 Dashboard", "📈 Graphique"])

# --- ONGLET 1 : CARTE ---
with tab1:
    col_sel, col_viz = st.columns([1, 3])
    with col_sel:
        st.markdown("#### Sélection")
        themes_map = sorted(list(set(v["theme"] for v in cat_dict.values())))
        t_map = st.selectbox("Thématique", themes_map, key="t_map")
        
        types_map = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == t_map)))
        ty_map = st.selectbox("Type de donnée", types_map, key="ty_map")
        
        vars_map = [k for k, v in cat_dict.items() if v["theme"] == t_map and v["type"] == ty_map]
        choix_var = st.selectbox("Indicateur", vars_map, format_func=format_func_clean)
        
        style_map = st.selectbox("Fond de carte", ["Voyager", "Clair", "Sombre"])
        styles = {"Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                  "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                  "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"}

    with col_viz:
        vmin, vmax = echelles_globales[choix_var]
        if vmin >= 0:
            norm = mcolors.Normalize(vmin=0, vmax=vmax); cmap = plt.get_cmap("Reds")
        elif vmax <= 0:
            norm = mcolors.Normalize(vmin=vmin, vmax=0); cmap = plt.get_cmap("Blues_r")
        else:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax); cmap = plt.get_cmap("coolwarm")

        df_plot = df_map.dropna(subset=[choix_var]).copy()
        colors = (cmap(norm(df_plot[choix_var].values))[:, :3] * 255).astype(int)
        df_plot["r"], df_plot["g"], df_plot["b"] = colors[:, 0], colors[:, 1], colors[:, 2]

        layer = pdk.Layer("GridCellLayer", df_plot, get_position="[Longitude, Latitude]",
                          get_color="[r, g, b, 180]", cell_size=8000, pickable=True)
        view = pdk.ViewState(latitude=u_lat if u_lat else 46.5, longitude=u_lon if u_lon else 2.5, zoom=8 if u_lat else 5)
        st.pydeck_chart(pdk.Deck(map_style=styles[style_map], initial_view_state=view, layers=[layer],
                                 tooltip={"html": f"<b>{format_func_clean(choix_var)}</b>: {{{choix_var}}}"}))
        
        fig, ax = plt.subplots(figsize=(6, 0.4))
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        ax.set_axis_off()
        st.pyplot(fig)
        c_min, c_mid, c_max = st.columns(3)
        c_min.markdown(f"<small>Min: {vmin:.1f}</small>", unsafe_allow_html=True)
        if vmin < 0 < vmax: c_mid.markdown("<center><small>0 (Blanc)</small></center>", unsafe_allow_html=True)
        c_max.markdown(f"<div style='text-align:right'><small>Max: {vmax:.1f}</small></div>", unsafe_allow_html=True)

# --- ONGLET 2 : DASHBOARD ---
with tab2:
    if u_lat:
        st.subheader(f"📍 Analyse locale : {adr_full}")
        df_map["dist"] = df_map.apply(lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1)
        voisins = df_map.nsmallest(5, "dist")
        w = 1 / (voisins["dist"] + 0.1)**2

        # Construction de la hiérarchie : Thème -> Type -> Liste Variables
        hierarchy = {}
        for var, info in cat_dict.items():
            if var in df_map.columns:
                t, ty = info["theme"], info["type"]
                if t not in hierarchy: hierarchy[t] = {}
                if ty not in hierarchy[t]: hierarchy[t][ty] = []
                hierarchy[t][ty].append(var)

        for theme in sorted(hierarchy.keys()):
            st.markdown(f"### {theme}")
            for dtype in sorted(hierarchy[theme].keys()):
                st.markdown(f"<div class='subcat-header'>{dtype}</div>", unsafe_allow_html=True)
                cols = st.columns(3)
                for i, v in enumerate(hierarchy[theme][dtype]):
                    val_interp = np.sum(voisins[v] * w) / np.sum(w)
                    m_nat = national_means[choix_scenario].get(v, 0)
                    unit = extraire_unite(descriptions.get(v, ""))
                    with cols[i % 3]:
                        st.metric(label=nettoyer_nom_variable(descriptions.get(v, v)), 
                                  value=f"{val_interp:.2f} {unit}",
                                  delta=f"{val_interp - m_nat:+.2f} vs Moy. Nat.",
                                  delta_color="inverse")
            st.divider()
    else:
        st.info("Recherchez une adresse dans la barre latérale pour activer le dashboard.")

# --- ONGLET 3 : GRAPHIQUE ---
with tab3:
    st.subheader("📈 Comparateur de variables")
    col_x, col_y = st.columns(2)
    
    themes_list = sorted(list(set(v["theme"] for v in cat_dict.values())))
    
    with col_x:
        st.markdown("**Axe X**")
        tx = st.selectbox("Thème X", themes_list, key="tx")
        types_x = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == tx)))
        tyx = st.selectbox("Type X", types_x, key="tyx")
        vars_x = [k for k, v in cat_dict.items() if v["theme"] == tx and v["type"] == tyx]
        var_x = st.selectbox("Variable X", vars_x, format_func=format_func_clean, key="vx")

    with col_y:
        st.markdown("**Axe Y**")
        ty = st.selectbox("Thème Y", themes_list, key="ty")
        types_y = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == ty)))
        tyy = st.selectbox("Type Y", types_y, key="tyy")
        vars_y = [k for k, v in cat_dict.items() if v["theme"] == ty and v["type"] == tyy]
        var_y = st.selectbox("Variable Y", vars_y, format_func=format_func_clean, key="vy")

    df_scat = df_map.dropna(subset=[var_x, var_y]).copy()
    df_scat["Légende"] = "Points du territoire"
    
    if u_lat:
        val_x = np.sum(voisins[var_x] * w) / np.sum(w)
        val_y = np.sum(voisins[var_y] * w) / np.sum(w)
        user_pt = pd.DataFrame({var_x: [val_x], var_y: [val_y], "Légende": ["📍 Votre Adresse"]})
        df_scat = pd.concat([df_scat, user_pt], ignore_index=True)

    fig = px.scatter(df_scat, x=var_x, y=var_y, color="Légende",
                     color_discrete_map={"Points du territoire": "rgba(100,100,100,0.3)", "📍 Votre Adresse": "#e74c3c"},
                     labels={var_x: format_func_clean(var_x), var_y: format_func_clean(var_y)},
                     template="plotly_white")
    
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color='white')), selector=dict(mode='markers'))
    st.plotly_chart(fig, use_container_width=True)
