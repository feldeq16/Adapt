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
    /* Style général */
    .block-container {padding-top: 2rem;}
    
    /* Metrics */
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

    /* Titres */
    h4 {
        margin-top: 30px;
        padding-bottom: 5px;
        border-bottom: 2px solid #f0f2f6;
        color: #0068c9;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        border-top: 3px solid #0068c9;
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

@st.cache_data(show_spinner=False)
def geocode_safe(address):
    try:
        agent = f"app_climat_{uuid.uuid4()}"
        geolocator = Nominatim(user_agent=agent, timeout=3)
        loc = geolocator.geocode(address)
        if loc: return loc.latitude, loc.longitude
    except: pass
    return None, None

def format_func_var(option):
    desc = descriptions.get(option, "")
    if desc: return f"{option} - {desc[:45]}..."
    return option

# ============================================
# 3. CHARGEMENT DONNÉES
# ============================================

data, echelles_globales, national_means = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
categories = lire_dict_fichier(FICHIER_CATEGORIES)

if data is None:
    st.error("❌ Aucune donnée trouvée. Vérifiez le dossier 'Données'.")
    st.stop()

# ============================================
# 4. SIDEBAR (FILTRES GLOBAUX)
# ============================================
with st.sidebar:
    st.header("🎛️ Paramètres Globaux")
    
    # --- 1. Adresse (Globale pour Dashboard et Comparateur) ---
    st.subheader("📍 Localisation")
    adr = st.text_input("Rechercher une adresse", placeholder="Ex: Place Bellecour, Lyon")
    u_lat, u_lon = None, None
    if adr:
        u_lat, u_lon = geocode_safe(adr)
        if u_lat:
            st.success("Adresse trouvée !")
        else:
            st.warning("Adresse introuvable.")
    
    st.divider()

    # --- 2. Scénarios ---
    st.subheader("📅 Contexte")
    scenarios = sorted(data["Contexte"].unique())
    choix_scenario = st.selectbox("Scénario (RCP)", scenarios)
    
    df_context = data[data["Contexte"] == choix_scenario]
    horizons = sorted(df_context["Période"].unique())
    choix_horizon = st.selectbox("Période / Horizon", horizons)

    # Préparation des données filtrées pour tous les onglets
    df_map = df_context[df_context["Période"] == choix_horizon].copy()
    
    # Calcul des voisins (Interpolation) fait une seule fois ici si adresse dispo
    voisins = None
    weights = None
    if u_lat and not df_map.empty:
        df_map = df_map.dropna(subset=["Latitude", "Longitude"])
        # Calcul distance
        df_map["dist_km"] = df_map.apply(
            lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
        )
        voisins = df_map.nsmallest(5, "dist_km")
        weights = 1 / (voisins["dist_km"] + 0.01)**2

# ============================================
# 5. ONGLETS PRINCIPAUX
# ============================================

tab_map, tab_dash, tab_scatter = st.tabs(["🗺️ Visualisateur Carte", "📊 Dashboard Adresse", "📈 Comparateur Variable"])

# --------------------------------------------
# ONGLET 1 : VISUALISATEUR DE CARTE
# --------------------------------------------
with tab_map:
    
    col_ctrl, col_map = st.columns([1, 3])
    
    with col_ctrl:
        st.markdown("#### Configuration Carte")
        # Filtre Catégorie
        liste_cats = sorted(list(set(categories.values())))
        if liste_cats:
            liste_cats.insert(0, "Toutes les catégories")
            choix_cat = st.selectbox("Filtrer par thème", liste_cats, key="map_cat")
        else:
            choix_cat = "Toutes les catégories"

        # Filtre Variable
        variables_dispos = sorted(list(echelles_globales.keys()))
        if choix_cat != "Toutes les catégories":
            variables_dispos = [v for v in variables_dispos if categories.get(v) == choix_cat]
        
        if not variables_dispos:
            st.warning("Aucune variable.")
        else:
            choix_var = st.selectbox("Variable à afficher", variables_dispos, format_func=format_func_var, key="map_var")
            
            # Style Map
            styles_map = {
                "Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
            }
            style_choisi = st.selectbox("Fond de carte", list(styles_map.keys()))

    with col_map:
        if variables_dispos and choix_var in df_map.columns:
            # Nettoyage
            df_display = df_map.dropna(subset=["Latitude", "Longitude", choix_var]).copy()

            # --- COULEURS ROBUSTES ---
            vmin_glob, vmax_glob = echelles_globales[choix_var]
            
            if vmin_glob >= 0:
                norm = mcolors.Normalize(vmin=0, vmax=vmax_glob)
                cmap = plt.get_cmap("Reds")
            elif vmax_glob <= 0:
                norm = mcolors.Normalize(vmin=vmin_glob, vmax=0)
                cmap = plt.get_cmap("Blues_r")
            else:
                norm = mcolors.TwoSlopeNorm(vmin=vmin_glob, vcenter=0, vmax=vmax_glob)
                cmap = plt.get_cmap("coolwarm")

            rgb = (cmap(norm(df_display[choix_var].values))[:, :3] * 255).astype(int)
            df_display["r"], df_display["g"], df_display["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

            # --- CARTE ---
            layers = [
                pdk.Layer(
                    "GridCellLayer",
                    data=df_display,
                    get_position="[Longitude, Latitude]",
                    get_color="[r, g, b, 170]",
                    cell_size=8000,
                    extruded=False,
                    pickable=True,
                    auto_highlight=True
                )
            ]

            # Ajout point adresse si existe
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
                view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=8)
            else:
                view = pdk.ViewState(latitude=46.6, longitude=2.0, zoom=5.5)
            
            desc_brute = descriptions.get(choix_var, choix_var)
            unit_var = extraire_unite(desc_brute)

            st.pydeck_chart(pdk.Deck(
                map_style=styles_map[style_choisi],
                initial_view_state=view,
                layers=layers,
                tooltip={"html": f"<b>{choix_var}</b>: {{{choix_var}}} {unit_var}<br><i>{{Point}}</i>"}
            ))

            # Légende
            st.caption(f"Légende : {nettoyer_nom_variable(desc_brute)}")
            fig, ax = plt.subplots(figsize=(8, 0.3))
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
            cb.outline.set_visible(False)
            ax.set_axis_off()
            st.pyplot(fig)


# --------------------------------------------
# ONGLET 2 : DASHBOARD DONNÉES À L'ADRESSE
# --------------------------------------------
with tab_dash:
    if not u_lat:
        st.info("👋 Veuillez entrer une adresse dans la barre latérale pour voir le tableau de bord.")
    else:
        st.markdown(f"### 📍 Analyse locale : {adr}")
        st.caption(f"Données interpolées basées sur les 5 stations les plus proches (Dist. min: {voisins.iloc[0]['dist_km']:.1f} km)")
        
        pixel_ref = voisins.iloc[0]
        all_vars = [c for c in pixel_ref.index if c in echelles_globales]
        
        # Groupement par catégorie
        grouped_vars = {}
        for var in all_vars:
            cat = categories.get(var, "Autres Indicateurs")
            if cat not in grouped_vars: grouped_vars[cat] = []
            grouped_vars[cat].append(var)
        
        sorted_cats = sorted(grouped_vars.keys())
        if "Autres Indicateurs" in sorted_cats:
            sorted_cats.remove("Autres Indicateurs")
            sorted_cats.append("Autres Indicateurs")

        for cat in sorted_cats:
            st.markdown(f"#### {cat}")
            vars_in_cat = grouped_vars[cat]
            cols = st.columns(3)
            
            for i, var_code in enumerate(vars_in_cat):
                # Interpolation Pondérée
                val_interp = np.sum(voisins[var_code] * weights) / np.sum(weights)
                mean_nat = national_means[choix_scenario].get(var_code, 0)
                
                full_desc = descriptions.get(var_code, var_code)
                clean_name = nettoyer_nom_variable(full_desc)
                unit = extraire_unite(full_desc)
                delta = val_interp - mean_nat
                
                with cols[i % 3]:
                    st.metric(
                        label=clean_name[:35] + ("..." if len(clean_name)>35 else ""),
                        value=f"{val_interp:.2f} {unit}",
                        delta=f"{delta:+.2f} vs Moy. Nat.",
                        delta_color="inverse"
                    )

# --------------------------------------------
# ONGLET 3 : COMPARATEUR (NUAGE DE POINTS)
# --------------------------------------------
with tab_scatter:
    
    st.markdown("### 📈 Analyse Croisée (Nuage de points)")
    st.markdown("Sélectionnez deux variables pour comparer l'ensemble des points géographiques et situer votre adresse.")

    c1, c2 = st.columns(2)
    vars_scatter = sorted(list(echelles_globales.keys()))
    
    with c1:
        var_x = st.selectbox("Variable Axe X", vars_scatter, index=0, format_func=format_func_var)
    with c2:
        # Essayer de prendre une variable différente par défaut pour Y
        idx_y = 1 if len(vars_scatter) > 1 else 0
        var_y = st.selectbox("Variable Axe Y", vars_scatter, index=idx_y, format_func=format_func_var)

    if var_x and var_y:
        # Création du plot avec Matplotlib
        fig_scat, ax_scat = plt.subplots(figsize=(10, 6))
        
        # 1. Tous les points (Gris/Bleu léger)
        ax_scat.scatter(
            df_map[var_x], 
            df_map[var_y], 
            alpha=0.4, 
            c="#0068c9", 
            edgecolors='none', 
            s=30, 
            label="Ensemble du territoire"
        )
        
        # 2. Point interpolé (si adresse présente)
        if u_lat and voisins is not None:
            val_x_interp = np.sum(voisins[var_x] * weights) / np.sum(weights)
            val_y_interp = np.sum(voisins[var_y] * weights) / np.sum(weights)
            
            ax_scat.scatter(
                [val_x_interp], 
                [val_y_interp], 
                c="red", 
                s=200, 
                marker="*", 
                edgecolors="black", 
                label=f"📍 {adr[:20]}..."
            )
            # Ligne pointillée pour repérer
            ax_scat.axvline(x=val_x_interp, color='red', linestyle=':', alpha=0.5)
            ax_scat.axhline(y=val_y_interp, color='red', linestyle=':', alpha=0.5)

        # Style du graphique
        ax_scat.set_xlabel(f"{var_x} ({extraire_unite(descriptions.get(var_x, ''))})")
        ax_scat.set_ylabel(f"{var_y} ({extraire_unite(descriptions.get(var_y, ''))})")
        ax_scat.set_title(f"Corrélation : {var_x} vs {var_y}")
        ax_scat.grid(True, linestyle='--', alpha=0.3)
        ax_scat.legend()
        
        # Affichage
        st.pyplot(fig_scat)
        
        if u_lat:
            st.info(f"Votre position se situe à **X={val_x_interp:.2f}** / **Y={val_y_interp:.2f}**")
        else:
            st.warning("Entrez une adresse dans la barre latérale pour voir votre position sur le graphique.")
