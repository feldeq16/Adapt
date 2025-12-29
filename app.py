import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import plotly.express as px  # Remplacement de matplotlib par Plotly pour le graph
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt # Gardé uniquement pour la colormap de la carte
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
    /* Style général */
    .main .block-container {padding-top: 2rem;}
    
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e6e6e6;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    h4 {
        margin-top: 30px;
        padding-bottom: 5px;
        border-bottom: 2px solid #f0f2f6;
        color: #0068c9;
    }
    
    /* Tabs Style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #eef0f4;
        color: #555;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 15px;
        border: none;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0068c9 !important;
        border-top: 3px solid #0068c9 !important;
        font-weight: bold;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Observatoire Climatique Multi-Scénarios")
st.markdown("---")

DOSSIER = "Données"
FICHIER_DEFINITIONS = "name.txt"
FICHIER_CATEGORIES = "category.txt"

# ============================================
# 2. FONCTIONS UTILITAIRES
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
    # Enlève les parenthèses à la fin (souvent l'unité) pour le titre propre
    return re.sub(r"\s*\(.*?\)$", "", description).strip()

def lire_fichier_data(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
    except:
        return None

# --- MODIFICATION: AFFICHER JUSTE LE NOM (PLUS D'ACRONYME) ---
def format_func_clean(option):
    """Retourne uniquement la description humaine de la variable."""
    desc = descriptions.get(option, "")
    if desc:
        return desc # On ne retourne que le nom complet, pas l'ID technique
    return option

@st.cache_data(show_spinner=False, ttl=3600)
def search_address_gouv(query):
    if not query or len(query) < 3: return []
    try:
        url = f"https://api-adresse.data.gouv.fr/search/?q={query}&limit=5&autocomplete=1"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            results = []
            for feature in data.get('features', []):
                label = feature['properties'].get('label')
                coords = feature['geometry']['coordinates']
                if label and coords:
                    results.append((label, coords[1], coords[0]))
            return results
    except: pass
    return []

# ============================================
# 3. CHARGEMENT DES DONNÉES
# ============================================
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
        global_scales[v] = (final_df[v].min(), final_df[v].max())
    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")
    return final_df, global_scales, national_means

data, echelles_globales, national_means = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
categories = lire_dict_fichier(FICHIER_CATEGORIES)

if data is None:
    st.error("❌ Aucune donnée trouvée.")
    st.stop()

# ============================================
# 4. SIDEBAR
# ============================================
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    st.subheader("📍 Localisation")
    if 'search_query' not in st.session_state: st.session_state.search_query = ''
    if 'selected_coords' not in st.session_state: st.session_state.selected_coords = (None, None)
    if 'final_address_label' not in st.session_state: st.session_state.final_address_label = None

    query = st.text_input("Adresse :", value=st.session_state.search_query, placeholder="Ex: Paris", key="input_addr")
    
    suggestions = []
    if query and len(query) > 3:
        suggestions = search_address_gouv(query)

    u_lat, u_lon = st.session_state.selected_coords
    adr_label = st.session_state.final_address_label

    if suggestions:
        options_map = {item[0]: (item[1], item[2]) for item in suggestions}
        selected_option = st.selectbox("Suggestions :", list(options_map.keys()), key="sb_addr")
        if selected_option:
            u_lat, u_lon = options_map[selected_option]
            adr_label = selected_option
            st.session_state.selected_coords = (u_lat, u_lon)
            st.session_state.final_address_label = adr_label
            st.session_state.search_query = selected_option
            
    if u_lat: st.success(f"Validé : {adr_label}")

    st.divider()

    st.subheader("📅 Contexte")
    scenarios = sorted(data["Contexte"].unique())
    choix_scenario = st.selectbox("Scénario", scenarios)
    
    df_context = data[data["Contexte"] == choix_scenario]
    horizons = sorted(df_context["Période"].unique())
    choix_horizon = st.selectbox("Horizon", horizons)

    df_map = df_context[df_context["Période"] == choix_horizon].copy()
    
    voisins, weights = None, None
    if u_lat and not df_map.empty:
        df_map = df_map.dropna(subset=["Latitude", "Longitude"])
        df_map["dist_km"] = df_map.apply(lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1)
        voisins = df_map.nsmallest(5, "dist_km")
        weights = 1 / (voisins["dist_km"] + 0.1)**2

# ============================================
# 5. ONGLETS
# ============================================
tab_map, tab_dash, tab_scatter = st.tabs(["🗺️ Carte", "📊 Dashboard", "📈 Comparateur"])

# --- ONGLET 1 : CARTE ---
with tab_map:
    col_ctrl, col_map = st.columns([1, 3], gap="medium")
    with col_ctrl:
        st.markdown("#### Configuration")
        liste_cats = sorted(list(set(categories.values())))
        if liste_cats:
            liste_cats.insert(0, "Toutes")
            choix_cat = st.selectbox("Thème", liste_cats)
        else: choix_cat = "Toutes"

        variables_dispos = sorted(list(echelles_globales.keys()))
        if choix_cat != "Toutes":
            variables_dispos = [v for v in variables_dispos if categories.get(v) == choix_cat]
        
        if variables_dispos:
            # Utilisation de format_func_clean pour n'afficher que le nom
            choix_var = st.selectbox("Indicateur", variables_dispos, format_func=format_func_clean)
            style_choisi = st.selectbox("Fond", ["Clair", "Sombre", "Voyager"])
            styles_map = {"Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                          "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                          "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"}

    with col_map:
        if variables_dispos and choix_var in df_map.columns:
            df_disp = df_map.dropna(subset=[choix_var]).copy()
            vmin, vmax = echelles_globales[choix_var]
            
            if vmin >= 0: 
                norm, cmap = mcolors.Normalize(0, vmax), plt.get_cmap("Reds")
            elif vmax <= 0: 
                norm, cmap = mcolors.Normalize(vmin, 0), plt.get_cmap("Blues_r")
            else: 
                norm, cmap = mcolors.TwoSlopeNorm(vmin, 0, vmax), plt.get_cmap("coolwarm")

            rgb = (cmap(norm(df_disp[choix_var].values))[:, :3] * 255).astype(int)
            df_disp["r"], df_disp["g"], df_disp["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

            layers = [pdk.Layer("GridCellLayer", data=df_disp, get_position="[Longitude, Latitude]",
                                get_color="[r, g, b, 170]", cell_size=8000, pickable=True)]

            if u_lat:
                layers.append(pdk.Layer("ScatterplotLayer", data=pd.DataFrame({"lat": [u_lat], "lon": [u_lon]}),
                                        get_position="[lon, lat]", get_color="[0, 255, 0]", get_radius=4000, stroked=True,
                                        get_line_color=[0,0,0], line_width_min_pixels=3))
                view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=8)
            else:
                view = pdk.ViewState(latitude=46.6, longitude=2.0, zoom=5.5)
            
            desc_clean = descriptions.get(choix_var, choix_var)
            unit = extraire_unite(desc_clean)
            
            st.pydeck_chart(pdk.Deck(map_style=styles_map[style_choisi], initial_view_state=view, layers=layers,
                                     tooltip={"html": f"<b>{desc_clean}</b>: {{{choix_var}}} {unit}"}))
            
            # Légende simple
            fig, ax = plt.subplots(figsize=(6, 0.3))
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
            cb.outline.set_visible(False)
            ax.set_axis_off()
            st.pyplot(fig, use_container_width=False)
            st.caption(f"Echelle : {nettoyer_nom_variable(desc_clean)}")

# --- ONGLET 2 : DASHBOARD ---
with tab_dash:
    if not u_lat or voisins is None:
        st.info("Veuillez sélectionner une adresse.")
    else:
        st.markdown(f"### 📍 Analyse locale : {adr_label}")
        
        # Logique de regroupement
        pixel_ref = voisins.iloc[0]
        vars_avail = [c for c in pixel_ref.index if c in echelles_globales]
        grouped = {}
        for v in vars_avail:
            c = categories.get(v, "Autres")
            if c not in grouped: grouped[c] = []
            grouped[c].append(v)
        
        for cat in sorted(grouped.keys()):
            st.markdown(f"#### {cat}")
            cols = st.columns(3, gap="large")
            for i, var in enumerate(grouped[cat]):
                val = np.sum(voisins[var] * weights) / np.sum(weights)
                mean = national_means[choix_scenario].get(var, 0)
                desc = descriptions.get(var, var)
                name_clean = nettoyer_nom_variable(desc)
                
                with cols[i % 3]:
                    st.metric(label=name_clean, value=f"{val:.2f} {extraire_unite(desc)}",
                              delta=f"{val-mean:+.2f} vs Moy.", delta_color="inverse")

# --- ONGLET 3 : COMPARATEUR (PLOTLY) ---
with tab_scatter:
    st.markdown("### 📈 Analyse Croisée (Interactive)")
    
    # 1. Sélection CATEORISÉE pour X et Y
    c1, c2 = st.columns(2)
    
    all_vars = sorted(list(echelles_globales.keys()))
    cats_unique = sorted(list(set([categories.get(v, "Autres") for v in all_vars])))
    
    with c1:
        st.markdown("**Axe Horizontal (X)**")
        cat_x = st.selectbox("Catégorie X", cats_unique, key="cat_x")
        # Filtrer les variables selon la catégorie X
        vars_x = [v for v in all_vars if categories.get(v, "Autres") == cat_x]
        var_x = st.selectbox("Variable X", vars_x, format_func=format_func_clean, key="var_x")

    with c2:
        st.markdown("**Axe Vertical (Y)**")
        # Essayer de pré-sélectionner une autre catégorie si possible
        idx_c_y = 1 if len(cats_unique) > 1 else 0
        cat_y = st.selectbox("Catégorie Y", cats_unique, index=idx_c_y, key="cat_y")
        # Filtrer les variables selon la catégorie Y
        vars_y = [v for v in all_vars if categories.get(v, "Autres") == cat_y]
        var_y = st.selectbox("Variable Y", vars_y, format_func=format_func_clean, key="var_y")

    # 2. Construction du Graphique Plotly
    if var_x and var_y and not df_map.empty:
        # Préparation dataframe : on garde tout le monde
        df_plot = df_map.dropna(subset=[var_x, var_y]).copy()
        df_plot["Type"] = "Territoire National"
        df_plot["Taille"] = 2 # Taille petite pour le fond
        
        # Ajout point adresse
        val_x_usr, val_y_usr = None, None
        if u_lat and voisins is not None:
            val_x_usr = np.sum(voisins[var_x] * weights) / np.sum(weights)
            val_y_usr = np.sum(voisins[var_y] * weights) / np.sum(weights)
            
            # On crée une ligne data pour l'utilisateur
            user_row = {
                var_x: val_x_usr,
                var_y: val_y_usr,
                "Type": "📍 Votre Adresse",
                "Taille": 15 # Plus gros
            }
            # Concaténation propre via DataFrame
            df_user = pd.DataFrame([user_row])
            df_plot = pd.concat([df_plot, df_user], ignore_index=True)

        # Labels propres pour les axes
        label_x = f"{nettoyer_nom_variable(descriptions.get(var_x, var_x))} ({extraire_unite(descriptions.get(var_x, ''))})"
        label_y = f"{nettoyer_nom_variable(descriptions.get(var_y, var_y))} ({extraire_unite(descriptions.get(var_y, ''))})"

        # Création Figure Plotly
        fig = px.scatter(
            df_plot, 
            x=var_x, 
            y=var_y, 
            color="Type",
            size="Taille", # Pour rendre le point utilisateur bien visible
            size_max=15,
            color_discrete_map={"Territoire National": "#BDC3C7", "📍 Votre Adresse": "#E74C3C"},
            labels={var_x: label_x, var_y: label_y},
            title=f"Comparaison : {nettoyer_nom_variable(descriptions.get(var_x))} vs {nettoyer_nom_variable(descriptions.get(var_y))}",
            hover_data={"Taille": False, "Type": True}
        )
        
        fig.update_layout(
            template="plotly_white",
            legend_title_text="",
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        if u_lat:
            st.info(f"Position interpolée : X = {val_x_usr:.2f} | Y = {val_y_usr:.2f}")
    else:
        st.warning("Données insuffisantes.")
