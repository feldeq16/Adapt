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
# 1. CONFIGURATION ET STYLE (CORRECTION BLANC SUR BLANC)
# ============================================
st.set_page_config(layout="wide", page_title="Observatoire Climatique", page_icon="🌍")

st.markdown("""
<style>
    /* Force le texte en noir/gris foncé dans tout le dashboard pour éviter le blanc sur blanc */
    .stApp {
        color: #31333F;
    }
    
    /* Style des Metrics du Dashboard */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Force la couleur du texte des labels et valeurs des metrics */
    div[data-testid="stMetricLabel"] {
        color: #4b5563 !important; /* Gris foncé */
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #1f2937 !important; /* Presque noir */
        font-size: 1.8rem !important;
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
</style>
""", unsafe_allow_html=True)

st.title("🌍 Observatoire Climatique")
st.markdown("---")

# Chemins fichiers
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
    final_df = combined.groupby(id_cols, as_index=False).agg({c: "first" for c in combined.columns if c not in id_cols})
    
    numeric_vars = [c for c in final_df.columns if c not in id_cols + ["Latitude", "Longitude"]]
    global_scales = {v: (final_df[v].min(), final_df[v].max()) for v in numeric_vars}
    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")
    return final_df, global_scales, national_means

# ============================================
# 3. LOGIQUE GLOBALE & RECHERCHE
# ============================================
data, echelles_globales, national_means = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
categories = lire_dict_fichier(FICHIER_CATEGORIES)

with st.sidebar:
    st.header("🎛️ Paramètres")
    # Recherche adresse simple pour l'exemple
    adr_input = st.text_input("📍 Rechercher une adresse", placeholder="Ex: Lyon")
    
    # Utilisation d'une API de recherche simple (BAN France)
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
        all_vars = sorted(list(echelles_globales.keys()))
        choix_var = st.selectbox("Indicateur", all_vars, format_func=format_func_clean)
        style_map = st.selectbox("Fond", ["Voyager", "Clair", "Sombre"])
        styles = {"Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                  "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                  "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"}

    with col_viz:
        vmin, vmax = echelles_globales[choix_var]
        
        # --- LOGIQUE DE COULEUR ROBUSTE (Fix ValueError) ---
        if vmin >= 0:
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
            cmap = plt.get_cmap("Reds")
        elif vmax <= 0:
            norm = mcolors.Normalize(vmin=vmin, vmax=0)
            cmap = plt.get_cmap("Blues_r")
        else:
            # MIXTE : On force le centre à 0 (Valeur blanche)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cmap = plt.get_cmap("coolwarm")

        # Application
        df_plot = df_map.dropna(subset=[choix_var]).copy()
        colors = (cmap(norm(df_plot[choix_var].values))[:, :3] * 255).astype(int)
        df_plot["r"], df_plot["g"], df_plot["b"] = colors[:, 0], colors[:, 1], colors[:, 2]

        layer = pdk.Layer("GridCellLayer", df_plot, get_position="[Longitude, Latitude]",
                          get_color="[r, g, b, 180]", cell_size=8000, pickable=True)
        
        view = pdk.ViewState(latitude=46.5, longitude=2.5, zoom=5)
        if u_lat:
            view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=8)

        st.pydeck_chart(pdk.Deck(map_style=styles[style_map], initial_view_state=view, layers=[layer],
                                 tooltip={"html": f"<b>{choix_var}</b>: {{{choix_var}}}"}))
        
        # --- LÉGENDE AVEC MIN/MAX AFFICHÉS ---
        st.write(f"**Légende : {nettoyer_nom_variable(format_func_clean(choix_var))}**")
        fig, ax = plt.subplots(figsize=(6, 0.4))
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
        ax.set_axis_off()
        st.pyplot(fig)
        
        col_min, col_mid, col_max = st.columns(3)
        # ---col_min.caption(f"Min: {vmin:.1f}")
        col_min.markdown(f"<div style='text-align:left'><small>Min: {vmin:.1f}</small></div>", unsafe_allow_html=True)
        if vmin < 0 < vmax: col_mid.markdown("<center><small>0</small></center>", unsafe_allow_html=True)
        col_max.markdown(f"<div style='text-align:right'><small>Max: {vmax:.1f}</small></div>", unsafe_allow_html=True)

# --- ONGLET 2 : DASHBOARD ---
with tab2:
    if u_lat:
        st.subheader(f"📍 Indicateurs pour : {adr_full}")
        # Interpolation simple (5 voisins)
        df_map["dist"] = df_map.apply(lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1)
        voisins = df_map.nsmallest(5, "dist")
        w = 1 / (voisins["dist"] + 0.1)**2

        # Groupement par catégories
        vars_avail = [v for v in df_map.columns if v in echelles_globales]
        grouped = {}
        for v in vars_avail:
            cat = categories.get(v, "Indicateurs")
            if cat not in grouped: grouped[cat] = []
            grouped[cat].append(v)

        for cat, v_list in grouped.items():
            st.markdown(f"#### {cat}")
            cols = st.columns(3)
            for i, v in enumerate(v_list):
                val_interp = np.sum(voisins[v] * w) / np.sum(w)
                m_nat = national_means[choix_scenario].get(v, 0)
                unit = extraire_unite(descriptions.get(v, ""))
                with cols[i % 3]:
                    st.metric(label=nettoyer_nom_variable(descriptions.get(v, v)), 
                              value=f"{val_interp:.2f} {unit}",
                              delta=f"{val_interp - m_nat:+.2f} vs Moy. Nat.",
                              delta_color="inverse")
    else:
        st.info("Recherchez une adresse dans la barre latérale.")

# --- ONGLET 3 : GRAPHIQUE ---
with tab3:
    st.subheader("📈 Corrélation et positionnement")
    col_x, col_y = st.columns(2)
    with col_x:
        var_x = st.selectbox("Variable X", all_vars, index=0, format_func=format_func_clean, key="x")
    with col_y:
        var_y = st.selectbox("Variable Y", all_vars, index=1, format_func=format_func_clean, key="y")

    df_scat = df_map.dropna(subset=[var_x, var_y]).copy()
    df_scat["Point d'intérêt"] = "Autres points"
    
    if u_lat:
        val_x = np.sum(voisins[var_x] * w) / np.sum(w)
        val_y = np.sum(voisins[var_y] * w) / np.sum(w)
        # Ajout du point interpolé
        new_row = pd.DataFrame({var_x: [val_x], var_y: [val_y], "Point d'intérêt": ["📍 Votre Adresse"]})
        df_scat = pd.concat([df_scat, new_row], ignore_index=True)

    fig = px.scatter(df_scat, x=var_x, y=var_y, color="Point d'intérêt",
                     color_discrete_map={"Autres points": "rgba(100,100,100,0.3)", "📍 Votre Adresse": "red"},
                     title=f"{nettoyer_nom_variable(format_func_clean(var_x))} vs {nettoyer_nom_variable(format_func_clean(var_y))}",
                     template="plotly_white")
    
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    st.plotly_chart(fig, use_container_width=True)
