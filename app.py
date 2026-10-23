import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import uuid
import re
import requests
from geopy.distance import geodesic

# ============================================
# 1. CONFIGURATION ET STYLE
# ============================================
st.set_page_config(
    layout="wide",
    page_title="Observatoire & Résilience Agro-Climatique",
    page_icon="🌍"
)

st.markdown("""
<style>
    .stApp { color: #31333F; }
    
    /* Style des Metrics du Dashboard */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        padding: 18px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #4b5563 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-size: 1.5rem !important;
    }

    /* Style des Onglets */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f3f4f6;
        color: #374151 !important;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0d9488 !important;
        border: 1px solid #0d9488 !important;
        font-weight: bold;
    }
    
    /* Titres des sous-catégories dans le dashboard */
    .subcat-header {
        color: #374151;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 12px;
        margin-bottom: 8px;
        border-left: 4px solid #0d9488;
        padding-left: 10px;
    }

    /* Encadrés de synthèse business */
    .highlight-box {
        background-color: #f0fdfa;
        border-left: 5px solid #0d9488;
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    .warning-box {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("🌍 Observatoire Climatique & Décision Agro-Économique")
    st.caption("Plateforme de modélisation des risques physiques, d'adaptation agronomique et d'ingénierie financière territoriale")
with col_badge:
    st.markdown(
        "<div style='text-align:right; margin-top:15px;'>"
        "<span style='background:#0d9488; color:white; padding:6px 12px; border-radius:20px; font-size:0.85rem; font-weight:600;'>"
        "Démonstrateur Stratégique</span></div>",
        unsafe_allow_html=True
    )
st.markdown("---")

DOSSIER = "Données"
FICHIER_DEFINITIONS = "name.txt"
FICHIER_CATEGORIES = "category.txt"

# ============================================
# 2. FONCTIONS DE TRAITEMENT & SECOURS
# ============================================

def lire_dict_fichier(path):
    d = {}
    if not os.path.exists(path):
        return d
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    parts = line.split(":", 1)
                    d[parts[0].strip()] = parts[1].strip()
    except Exception:
        pass
    return d

def lire_dict_categories_double(path):
    d = {}
    if not os.path.exists(path):
        return d
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    if "|" in val:
                        theme, dtype = val.split("|", 1)
                        d[key.strip()] = {"theme": theme.strip(), "type": dtype.strip()}
                    else:
                        d[key.strip()] = {"theme": val.strip(), "type": "Valeurs"}
    except Exception:
        pass
    return d

def extraire_unite(description):
    match = re.search(r"\((.*?)\)$", description.strip())
    return match.group(1) if match else ""

def nettoyer_nom_variable(description):
    return re.sub(r"\s*\(.*?\)$", "", description).strip()

def generer_donnees_secours():
    """Génère un jeu de données de démonstration si le dossier 'Données' est absent."""
    np.random.seed(42)
    lats = np.linspace(43.0, 50.0, 12)
    lons = np.linspace(-1.5, 6.5, 12)
    rows = []
    pt = 1
    for lat in lats:
        for lon in lons:
            base_temp = 13.5 - (lat - 43.0) * 0.45
            base_rain = 750 + (lon + 1.5) * 20
            for ctx, mult in [("RCP 4.5", 1.0), ("RCP 8.5", 1.45)]:
                for per, t_add in [("Référence", 0.0), ("Horizon 2030", 1.2 * mult), ("Horizon 2050", 2.4 * mult)]:
                    rows.append({
                        "Point": pt,
                        "Contexte": ctx,
                        "Période": per,
                        "Latitude": round(lat, 3),
                        "Longitude": round(lon, 3),
                        "TM_ann": round(base_temp + t_add + np.random.normal(0, 0.2), 2),
                        "Jours_Chauds": max(0, round((18 - (lat - 43.0) * 1.8) + t_add * 8 + np.random.normal(0, 1.5), 1)),
                        "Deficit_Hydrique": round(max(20, 140 + t_add * 45 - (lat - 43.0) * 10 + np.random.normal(0, 8)), 1),
                        "Pluie_Ete": round(max(40, base_rain * 0.25 * (1 - 0.08 * t_add) + np.random.normal(0, 5)), 1),
                        "Jours_Gel_Printemps": max(0, round(6 - t_add * 0.6 + np.random.normal(0, 0.5), 1))
                    })
            pt += 1
    df = pd.DataFrame(rows)
    desc_secours = {
        "TM_ann": "Température moyenne annuelle (°C)",
        "Jours_Chauds": "Jours de stress thermique > 30°C (jours)",
        "Deficit_Hydrique": "Déficit hydrique estival cumulé (mm)",
        "Pluie_Ete": "Cumul de précipitations estivales (mm)",
        "Jours_Gel_Printemps": "Risque de gel tardif printanier (jours)"
    }
    cat_secours = {
        "TM_ann": {"theme": "Température", "type": "Moyennes"},
        "Jours_Chauds": {"theme": "Température", "type": "Extrêmes"},
        "Deficit_Hydrique": {"theme": "Eau & Sécheresse", "type": "Bilan hydrique"},
        "Pluie_Ete": {"theme": "Eau & Sécheresse", "type": "Précipitations"},
        "Jours_Gel_Printemps": {"theme": "Agro-météorologie", "type": "Extrêmes"}
    }
    return df, desc_secours, cat_secours

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier):
        return None, None, None
    all_dfs = []
    id_cols = ["Point", "Contexte", "Période"]
    for f in os.listdir(dossier):
        if not f.endswith(".txt"):
            continue
        try:
            df = pd.read_csv(os.path.join(dossier, f), sep=None, engine="python", comment="#")
            df.columns = [c.strip() for c in df.columns]
            for c in df.columns:
                if c not in id_cols:
                    df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
            all_dfs.append(df)
        except Exception:
            continue
    if not all_dfs:
        return None, None, None
    combined = pd.concat(all_dfs, ignore_index=True)
    agg_dict = {c: "first" for c in combined.columns if c not in id_cols}
    final_df = combined.groupby(id_cols, as_index=False).agg(agg_dict)
    
    numeric_vars = [c for c in final_df.columns if c not in id_cols + ["Latitude", "Longitude"]]
    global_scales = {v: (float(final_df[v].min()), float(final_df[v].max())) for v in numeric_vars}
    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")
    return final_df, global_scales, national_means

# ============================================
# 3. CHARGEMENT ET SYNCHRONISATION
# ============================================
data, echelles_globales, national_means = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
cat_dict = lire_dict_categories_double(FICHIER_CATEGORIES)

# Bascule automatique en mode démonstration si les fichiers locaux ne sont pas trouvés
mode_demo_secours = False
if data is None or data.empty:
    mode_demo_secours = True
    data, desc_sec, cat_sec = generer_donnees_secours()
    id_cols = ["Point", "Contexte", "Période"]
    numeric_vars = [c for c in data.columns if c not in id_cols + ["Latitude", "Longitude"]]
    echelles_globales = {v: (float(data[v].min()), float(data[v].max())) for v in numeric_vars}
    national_means = data.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")
    if not descriptions:
        descriptions = desc_sec
    if not cat_dict:
        cat_dict = cat_sec

# Garantir que toutes les variables de cat_dict existent bien dans data (évite tout KeyError)
vars_disponibles = [c for c in data.columns if c not in ["Point", "Contexte", "Période", "Latitude", "Longitude"]]
cat_dict_valide = {k: v for k, v in cat_dict.items() if k in vars_disponibles}
if not cat_dict_valide:
    cat_dict_valide = {v: {"theme": "Indicateurs Climatiques", "type": "Valeurs"} for v in vars_disponibles}
cat_dict = cat_dict_valide

def format_func_clean(option):
    return descriptions.get(option, option)

# ============================================
# 4. SIDEBAR & GÉOLOCALISATION
# ============================================
with st.sidebar:
    st.header("🎛️ Paramètres Territoriaux")
    if mode_demo_secours:
        st.caption("ℹ️ *Données de démonstration actives (Dossier `Données` non détecté).*")
    
    adr_input = st.text_input("📍 Rechercher une commune / ferme", value="Beaune", placeholder="Ex: Lyon, Chartres, Beaune...")
    u_lat, u_lon, adr_full = None, None, None
    if adr_input and len(adr_input.strip()) > 2:
        try:
            r = requests.get(
                f"https://api-adresse.data.gouv.fr/search/?q={adr_input}&limit=1",
                timeout=3
            ).json()
            if r.get("features"):
                u_lon, u_lat = r["features"][0]["geometry"]["coordinates"]
                adr_full = r["features"][0]["properties"]["label"]
                st.success(f"📍 Localisé : {adr_full}")
        except Exception:
            pass
    
    # Coordonnées de secours si l'API est hors-ligne ou le champ vide
    if u_lat is None or u_lon is None:
        u_lat = float(data["Latitude"].median())
        u_lon = float(data["Longitude"].median())
        adr_full = adr_input if adr_input else "Point central du territoire (Défaut)"
        st.info(f"Position : {adr_full} ({u_lat:.2f}°N, {u_lon:.2f}°E)")

    st.divider()
    st.subheader("🌡️ Scénario Climatique")
    scenarios_dispo = sorted(data["Contexte"].unique())
    choix_scenario = st.selectbox("Scénario RCP / SSP", scenarios_dispo, index=len(scenarios_dispo)-1)
    df_step = data[data["Contexte"] == choix_scenario]
    
    periodes_dispo = sorted(df_step["Période"].unique())
    choix_horizon = st.selectbox("Horizon temporel", periodes_dispo, index=len(periodes_dispo)-1)
    df_map = df_step[df_step["Période"] == choix_horizon].copy()

    st.divider()
    st.markdown(
        "**💡 Mode Présentation :**\n"
        "- **Onglets 1 à 3 :** Exploration scientifique des aléas.\n"
        "- **Onglet 4 :** Stress-test financier de l'exploitation.\n"
        "- **Onglet 5 :** Solutions (Agroécologie, Agrivoltaïsme, Biochar) & Financements.\n"
        "- **Onglet 6 :** Génération automatique du rapport client."
    )

# Pré-calcul de l'interpolation spatiale pour l'adresse sélectionnée (utilisé dans tous les onglets)
df_map["dist"] = df_map.apply(
    lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
)
voisins = df_map.nsmallest(5, "dist")
w = 1.0 / (voisins["dist"] + 0.1) ** 2

def interpoler_valeur(df_source, var_name):
    """Interpole une variable sur les 5 points les plus proches."""
    if var_name not in df_source.columns:
        return 0.0
    sub = df_source.dropna(subset=[var_name]).copy()
    if sub.empty:
        return 0.0
    sub["dist"] = sub.apply(lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1)
    v_loc = sub.nsmallest(5, "dist")
    w_loc = 1.0 / (v_loc["dist"] + 0.1) ** 2
    return float(np.sum(v_loc[var_name] * w_loc) / np.sum(w_loc))

# ============================================
# 5. ONGLETS DE L'APPLICATION
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Carte des Aléas",
    "📊 Diagnostic Local & Trajectoire",
    "📈 Analyse Croisée",
    "🌾 Stress-Test Agro-Financier",
    "⚡ Solutions & Financements",
    "📄 Rapport Client"
])

# --------------------------------------------
# ONGLET 1 : CARTE
# --------------------------------------------
with tab1:
    col_sel, col_viz = st.columns([1, 3])
    with col_sel:
        st.markdown("#### Sélection de l'aléa")
        themes_map = sorted(list(set(v["theme"] for v in cat_dict.values())))
        t_map = st.selectbox("Thématique", themes_map, key="t_map")
        
        types_map = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == t_map)))
        ty_map = st.selectbox("Type de donnée", types_map, key="ty_map")
        
        vars_map = [k for k, v in cat_dict.items() if v["theme"] == t_map and v["type"] == ty_map]
        choix_var = st.selectbox("Indicateur", vars_map, format_func=format_func_clean)
        
        style_map = st.selectbox("Fond de carte", ["Voyager", "Clair", "Sombre"])
        styles = {
            "Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
        }

        # Valeur locale rapide dans le panneau gauche
        val_loc_carte = np.sum(voisins[choix_var] * w) / np.sum(w)
        unit_carte = extraire_unite(descriptions.get(choix_var, ""))
        st.divider()
        st.metric(
            label=f"📍 Valeur à {adr_full[:22]}",
            value=f"{val_loc_carte:.2f} {unit_carte}",
            delta=f"{val_loc_carte - national_means[choix_scenario].get(choix_var, 0):+.2f} vs Moy. Nat.",
            delta_color="inverse"
        )

    with col_viz:
        vmin, vmax = echelles_globales.get(choix_var, (0.0, 1.0))
        if np.isnan(vmin) or np.isnan(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0

        if vmin >= 0:
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
            cmap = plt.get_cmap("Reds")
        elif vmax <= 0:
            norm = mcolors.Normalize(vmin=vmin, vmax=0)
            cmap = plt.get_cmap("Blues_r")
        else:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cmap = plt.get_cmap("coolwarm")

        df_plot = df_map.dropna(subset=[choix_var]).copy()
        colors = (cmap(norm(df_plot[choix_var].values))[:, :3] * 255).astype(int)
        df_plot["r"], df_plot["g"], df_plot["b"] = colors[:, 0], colors[:, 1], colors[:, 2]

        grid_layer = pdk.Layer(
            "GridCellLayer",
            df_plot,
            get_position="[Longitude, Latitude]",
            get_color="[r, g, b, 180]",
            cell_size=8000,
            pickable=True
        )
        
        # Marqueur de l'adresse recherchée sur la carte
        df_pin = pd.DataFrame([{
            "Latitude": u_lat,
            "Longitude": u_lon,
            "Nom": adr_full,
            choix_var: round(val_loc_carte, 2)
        }])
        pin_layer = pdk.Layer(
            "ScatterplotLayer",
            df_pin,
            get_position="[Longitude, Latitude]",
            get_fill_color="[13, 148, 136, 255]",
            get_line_color="[255, 255, 255, 255]",
            get_radius=6500,
            stroked=True,
            line_width_min_pixels=2,
            pickable=True
        )

        view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=7.5 if adr_input else 5.2)
        st.pydeck_chart(pdk.Deck(
            map_style=styles[style_map],
            initial_view_state=view,
            layers=[grid_layer, pin_layer],
            tooltip={"html": f"<b>{format_func_clean(choix_var)}</b>: {{{choix_var}}}"}
        ))
        
        fig_cb, ax = plt.subplots(figsize=(6, 0.35))
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal")
        ax.set_axis_off()
        st.pyplot(fig_cb)
        plt.close(fig_cb)

        c_min, c_mid, c_max = st.columns(3)
        c_min.markdown(f"<small>Min: {vmin:.1f}</small>", unsafe_allow_html=True)
        if vmin < 0 < vmax:
            c_mid.markdown("<center><small>0 (Neutre)</small></center>", unsafe_allow_html=True)
        c_max.markdown(f"<div style='text-align:right'><small>Max: {vmax:.1f}</small></div>", unsafe_allow_html=True)

# --------------------------------------------
# ONGLET 2 : DASHBOARD LOCAL & TRAJECTOIRE
# --------------------------------------------
with tab2:
    st.subheader(f"📍 Diagnostic Agro-Climatique Local : {adr_full}")
    
    # Nouveau bloc : Évolution temporelle multi-scénarios & multi-périodes
    with st.expander("📈 Voir la trajectoire d'évolution multi-horizons et multi-scénarios sur ce site", expanded=True):
        var_traj = st.selectbox(
            "Choisir l'indicateur à projeter dans le temps",
            vars_disponibles,
            index=vars_disponibles.index(choix_var) if choix_var in vars_disponibles else 0,
            format_func=format_func_clean,
            key="var_traj_tab2"
        )
        traj_rows = []
        for (ctx_i, per_i), grp in data.groupby(["Contexte", "Période"]):
            val_i = interpoler_valeur(grp, var_traj)
            traj_rows.append({"Scénario": ctx_i, "Période": str(per_i), "Valeur": val_i})
        df_traj = pd.DataFrame(traj_rows)
        
        if not df_traj.empty:
            fig_traj = px.line(
                df_traj, x="Période", y="Valeur", color="Scénario", markers=True,
                title=f"Trajectoire locale : {format_func_clean(var_traj)} ({adr_full})",
                template="plotly_white"
            )
            fig_traj.update_traces(line=dict(width=3), marker=dict(size=9))
            st.plotly_chart(fig_traj, use_container_width=True)

    st.markdown("#### 📋 Tableau de bord complet par thématique")
    hierarchy = {}
    for var, info in cat_dict.items():
        if var in df_map.columns:
            t, ty = info["theme"], info["type"]
            if t not in hierarchy:
                hierarchy[t] = {}
            if ty not in hierarchy[t]:
                hierarchy[t][ty] = []
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
                    st.metric(
                        label=nettoyer_nom_variable(descriptions.get(v, v)),
                        value=f"{val_interp:.2f} {unit}",
                        delta=f"{val_interp - m_nat:+.2f} vs Moy. Nat.",
                        delta_color="inverse"
                    )
        st.divider()

# --------------------------------------------
# ONGLET 3 : GRAPHIQUE CROISÉ
# --------------------------------------------
with tab3:
    st.subheader("📈 Comparateur territorial & Positionnement de l'exploitation")
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
        ty = st.selectbox("Thème Y", themes_list, index=min(1, len(themes_list)-1), key="ty")
        types_y = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == ty)))
        tyy = st.selectbox("Type Y", types_y, key="tyy")
        vars_y = [k for k, v in cat_dict.items() if v["theme"] == ty and v["type"] == tyy]
        var_y = st.selectbox("Variable Y", vars_y, index=min(1, len(vars_y)-1), format_func=format_func_clean, key="vy")

    df_scat = df_map.dropna(subset=[var_x, var_y]).copy()
    df_scat["Légende"] = "Mailles du territoire"
    
    val_x = np.sum(voisins[var_x] * w) / np.sum(w)
    val_y = np.sum(voisins[var_y] * w) / np.sum(w)
    user_pt = pd.DataFrame({var_x: [val_x], var_y: [val_y], "Légende": [f"📍 {adr_full[:25]}"]})
    df_scat_plot = pd.concat([df_scat[[var_x, var_y, "Légende"]], user_pt], ignore_index=True)

    fig_scat = px.scatter(
        df_scat_plot, x=var_x, y=var_y, color="Légende",
        color_discrete_map={
            "Mailles du territoire": "rgba(120,130,140,0.35)",
            f"📍 {adr_full[:25]}": "#e11d48"
        },
        labels={var_x: format_func_clean(var_x), var_y: format_func_clean(var_y)},
        template="plotly_white"
    )
    fig_scat.update_traces(marker=dict(size=11, line=dict(width=1, color="white")), selector=dict(mode="markers"))
    st.plotly_chart(fig_scat, use_container_width=True)

    # Calcul du centile d'exposition sur le territoire
    pct_x = (df_scat[var_x] < val_x).mean() * 100
    pct_y = (df_scat[var_y] < val_y).mean() * 100
    st.info(
        f"📊 **Positionnement statistique national ({choix_scenario} - {choix_horizon}) :** "
        f"Votre localisation se situe au **{pct_x:.0f}e centile** sur *{nettoyer_nom_variable(format_func_clean(var_x))}* "
        f"et au **{pct_y:.0f}e centile** sur *{nettoyer_nom_variable(format_func_clean(var_y))}*."
    )

# --------------------------------------------
# ONGLET 4 : STRESS-TEST AGRO-FINANCIER (NOUVEAU)
# --------------------------------------------
with tab4:
    st.subheader(f"🌾 Modélisation Technico-Économique & Stress-Test d'Exploitation ({adr_full})")
    st.markdown(
        "Cet outil traduit les **aléas climatiques physiques locaux** en **impacts agronomiques et financiers** "
        "(Chiffre d'affaires, EBE/EBITDA, capacité de remboursement d'emprunt et risque de défaut pour les banques, coopératives et filières)."
    )

    # Paramètres par défaut selon la filière choisie
    profils_filieres = {
        "Grandes Cultures (Blé, Maïs, Colza, Orge)": {"surf": 140, "ca_ha": 1650, "ebe_pct": 32, "dette_an": 42000, "sensibilite": 1.0},
        "Viticulture & Champagne (Haute Valeur Ajoutée)": {"surf": 18, "ca_ha": 24000, "ebe_pct": 38, "dette_an": 65000, "sensibilite": 1.35},
        "Arboriculture & Vergers Pérennes": {"surf": 35, "ca_ha": 9500, "ebe_pct": 30, "dette_an": 48000, "sensibilite": 1.30},
        "Polyculture - Élevage (Bovin Lait / Autonomie fourragère)": {"surf": 110, "ca_ha": 2200, "ebe_pct": 28, "dette_an": 38000, "sensibilite": 1.15},
        "Maraîchage & Légumes d'Industrie": {"surf": 45, "ca_ha": 6800, "ebe_pct": 31, "dette_an": 35000, "sensibilite": 1.20}
    }

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        filiere = st.selectbox("🌱 Orientation technico-économique (OTEX)", list(profils_filieres.keys()))
        p_def = profils_filieres[filiere]
        surface_ha = st.number_input("Surface exploitée (ha)", min_value=1, max_value=2000, value=p_def["surf"], step=5)
    with col_p2:
        ca_par_ha = st.number_input("Chiffre d'affaires historique (€ / ha)", min_value=200, max_value=80000, value=p_def["ca_ha"], step=100)
        marge_ebe_pct = st.slider("Taux d'EBE / EBITDA historique (% du CA)", min_value=10, max_value=65, value=p_def["ebe_pct"])
    with col_p3:
        var_aléa_ref = st.selectbox(
            "🔥 Variable climatique directrice du stress-test",
            vars_disponibles,
            index=vars_disponibles.index(choix_var) if choix_var in vars_disponibles else 0,
            format_func=format_func_clean
        )
        annuite_dette = st.number_input("Annuités d'emprunt bancaire (€ / an)", min_value=0, max_value=500000, value=p_def["dette_an"], step=2000)

    # Calcul de l'intensité du choc climatique local à partir des données réelles
    val_locale_stress = np.sum(voisins[var_aléa_ref] * w) / np.sum(w)
    vmin_s, vmax_s = echelles_globales.get(var_aléa_ref, (0.0, 1.0))
    span_s = max(abs(vmax_s - vmin_s), 1e-4)
    indice_exposition_local = np.clip(abs(val_locale_stress - vmin_s) / span_s, 0.08, 0.95)
    
    # Facteur aggravant selon le scénario et la sensibilité de la filière
    perte_rendement_moy_pct = round(indice_exposition_local * 26.0 * p_def["sensibilite"], 1)
    perte_rendement_crise_pct = round(min(75.0, perte_rendement_moy_pct * 2.1), 1)  # Année extrême (décennale)

    ca_initial = surface_ha * ca_par_ha
    ebe_initial = ca_initial * (marge_ebe_pct / 100.0)
    charges_fixes_est = ca_initial - ebe_initial

    # En agriculture, une baisse de rendement réduit le CA mais les charges fixes (foncier, matériel, main-d'œuvre) restent rigides
    perte_ca_moy = ca_initial * (perte_rendement_moy_pct / 100.0)
    ebe_projete_moy = ebe_initial - (perte_ca_moy * 0.78)  # 78% de rigidité des charges à court/moyen terme
    perte_ca_crise = ca_initial * (perte_rendement_crise_pct / 100.0)
    ebe_projete_crise = ebe_initial - (perte_ca_crise * 0.85)

    solde_treso_init = ebe_initial - annuite_dette
    solde_treso_proj = ebe_projete_moy - annuite_dette
    solde_treso_crise = ebe_projete_crise - annuite_dette

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CA Historique Ferme", f"{ca_initial:,.0f} €/an".replace(",", " "))
    k2.metric(
        f"Choc Rendement Moyen ({choix_horizon})",
        f"-{perte_rendement_moy_pct:.1f} %",
        delta=f"-{perte_rendement_crise_pct:.1f}% en année extrême",
        delta_color="inverse"
    )
    k3.metric(
        "EBE / EBITDA Projeté (Sans adaptation)",
        f"{ebe_projete_moy:,.0f} €/an".replace(",", " "),
        delta=f"{ebe_projete_moy - ebe_initial:+,.0f} €/an vs Hist.".replace(",", " ")
    )
    statut_dette = "Sain ✅" if solde_treso_proj > 15000 else ("Tension ⚠️" if solde_treso_proj > 0 else "Défaut Bancaire 🚨")
    k4.metric(
        "Couverture de la Dette (DSCR)",
        statut_dette,
        delta=f"Reste après dette : {solde_treso_proj:,.0f} €".replace(",", " "),
        delta_color="normal" if solde_treso_proj > 0 else "inverse"
    )

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        df_stress_bar = pd.DataFrame({
            "Situation": ["1. Référence Historique", f"2. Moyenne ({choix_horizon})", f"3. Année Extrême ({choix_horizon})"],
            "EBE / EBITDA (€)": [ebe_initial, ebe_projete_moy, ebe_projete_crise],
            "Seuil Annuités Bancaires (€)": [annuite_dette, annuite_dette, annuite_dette]
        })
        fig_stress = go.Figure()
        fig_stress.add_trace(go.Bar(
            x=df_stress_bar["Situation"],
            y=df_stress_bar["EBE / EBITDA (€)"],
            name="EBE / EBITDA Généré (€)",
            marker_color=["#0d9488", "#f59e0b", "#e11d48"],
            text=[f"{v:,.0f} €".replace(",", " ") for v in df_stress_bar["EBE / EBITDA (€)"]],
            textposition="auto"
        ))
        fig_stress.add_trace(go.Scatter(
            x=df_stress_bar["Situation"],
            y=df_stress_bar["Seuil Annuités Bancaires (€)"],
            name="Annuités d'emprunt à rembourser",
            mode="lines+markers",
            line=dict(color="#1f2937", width=3, dash="dash")
        ))
        fig_stress.update_layout(
            title="Stress-Test Financier : Érosion de l'EBE face aux engagements bancaires",
            template="plotly_white",
            yaxis_title="Montant annuel (€)"
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    with col_g2:
        st.markdown("#### 🔍 Diagnostic Risque Physique & CSRD")
        if solde_treso_proj <= 0:
            st.markdown(
                f"<div class='warning-box'><b>🚨 Alerte Vulnérabilité Critique :</b><br>"
                f"Sous le scénario <b>{choix_scenario} ({choix_horizon})</b>, la perte moyenne d'EBE "
                f"(<b>{ebe_initial - ebe_projete_moy:,.0f} €/an</b>) ne permet plus de couvrir les annuités bancaires "
                f"sans puiser dans la trésorerie ou réduire la rémunération de l'agriculteur.</div>".replace(",", " "),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='highlight-box'><b>⚠️ Fragilisation de la rentabilité :</b><br>"
                f"L'exploitation reste solvable en année moyenne mais perd <b>{(1 - ebe_projete_moy/max(ebe_initial,1))*100:.0f}% de son EBE</b>. "
                f"En cas d'aléa décennal, le déficit après dette atteint <b>{solde_treso_crise:,.0f} €</b>.</div>".replace(",", " "),
                unsafe_allow_html=True
            )
        st.markdown(
            "**Impacts pour la chaîne de valeur (Approche B2B2F) :**\n"
            f"- **Coopérative & Industriel Aval (Scope 3 CSRD) :** Risque de rupture d'approvisionnement estimé à **-{perte_rendement_moy_pct:.1f}%** en volume moyen.\n"
            f"- **Banque & Assurance :** Dégradation du ratio de couverture de la dette (DSCR passant de **{ebe_initial/max(annuite_dette,1):.2f}x** à **{ebe_projete_moy/max(annuite_dette,1):.2f}x**)."
        )

# --------------------------------------------
# ONGLET 5 : SOLUTIONS D'ADAPTATION, ÉNERGIE & FINANCEMENT (NOUVEAU)
# --------------------------------------------
with tab5:
    st.subheader("⚡ Simulateur de Solutions d'Adaptation, Systèmes Énergétiques & Ingénierie Financière")
    st.markdown(
        "Construisez le **mix technico-économique optimal** (agronomie, hydrologie, énergie décentralisée) "
        "et mobilisez les **coalitions de financement** (PSE, Primes Filières CSRD, Crédits Carbone, Subventions) "
        "pour traverser la transition et sécuriser le revenu de l'exploitation."
    )

    catalogue_solutions = {
        "🌱 Agriculture Régénérative & Hydrologie (Couverts, semis direct, design Keyline)": {
            "capex_ha": 320, "gain_resilience_pct": 35, "revenu_energie_ha": 0, "tco2_ha_an": 1.4
        },
        "🌳 Agroforesterie / Vitiforesterie & Haies brise-vent (Micro-climat & Biodiversité)": {
            "capex_ha": 650, "gain_resilience_pct": 25, "revenu_energie_ha": 40, "tco2_ha_an": 1.8
        },
        "💧 Irrigation de Précision Sobriété & Sondes Tensiométriques Pilotées": {
            "capex_ha": 1400, "gain_resilience_pct": 45, "revenu_energie_ha": -35, "tco2_ha_an": 0.1
        },
        "☀️ Agrivoltaïsme Dynamique Protecteur (Ombrage piloté anti-gel/grêle + Rente électrique)": {
            "capex_ha": 1800, "gain_resilience_pct": 40, "revenu_energie_ha": 420, "tco2_ha_an": 0.8
        },
        "🔥 Hub Biomasse Collectif : Pyrogazéification & Épandage de Biochar (Rétention d'eau + H2/Élec)": {
            "capex_ha": 950, "gain_resilience_pct": 38, "revenu_energie_ha": 260, "tco2_ha_an": 2.5
        }
    }

    col_sol, col_fin = st.columns([3, 2])
    with col_sol:
        st.markdown("#### 🛠️ 1. Leviers Technico-Agronomiques & Énergétiques")
        solutions_choisies = st.multiselect(
            "Sélectionnez les briques d'adaptation à déployer sur l'exploitation :",
            list(catalogue_solutions.keys()),
            default=[
                "🌱 Agriculture Régénérative & Hydrologie (Couverts, semis direct, design Keyline)",
                "🔥 Hub Biomasse Collectif : Pyrogazéification & Épandage de Biochar (Rétention d'eau + H2/Élec)"
            ]
        )
        part_surface_adaptee = st.slider(
            "Part de la surface de la ferme équipée / engagée (%)",
            min_value=10, max_value=100, value=60, step=10
        ) / 100.0

    with col_fin:
        st.markdown("#### 🤝 2. Coalition de Financement (B2B2F)")
        subv_capex_pct = st.slider("Subventions Publiques (PAC / Région / France 2030 - % CAPEX)", 0, 70, 35, 5)
        pse_eau_ha = st.slider("PSE Agence de l'Eau & Biodiversité (€ / ha engagé / an)", 0, 200, 95, 5)
        prime_filiere_ha = st.slider("Prime Filière Industrielle Scope 3 CSRD (€ / ha engagé / an)", 0, 250, 80, 10)
        prix_tco2 = st.slider("Valorisation Carbone / Biochar Label Bas Carbone (€ / tCO2e)", 0, 150, 55, 5)

    # Calculs d'ingénierie technico-financière
    surf_engagee = surface_ha * part_surface_adaptee
    capex_brut = sum(catalogue_solutions[s]["capex_ha"] for s in solutions_choisies) * surf_engagee
    capex_net = capex_brut * (1.0 - subv_capex_pct / 100.0)

    # Réduction du choc climatique (plafonnée à 88% de réduction des pertes climatiques)
    reduction_choc_pct = min(0.88, sum(catalogue_solutions[s]["gain_resilience_pct"] for s in solutions_choisies) / 100.0) * (0.4 + 0.6 * part_surface_adaptee)
    perte_evitee_an = (ebe_initial - ebe_projete_moy) * reduction_choc_pct

    # Revenus complémentaires énergétiques (Biomasse / Agrivoltaïsme) et environnementaux (PSE + Carbone + CSRD)
    revenu_energie_an = sum(catalogue_solutions[s]["revenu_energie_ha"] for s in solutions_choisies) * surf_engagee
    tonnes_co2_an = sum(catalogue_solutions[s]["tco2_ha_an"] for s in solutions_choisies) * surf_engagee
    revenu_coalition_an = surf_engagee * (pse_eau_ha + prime_filiere_ha) + tonnes_co2_an * prix_tco2

    gain_total_annuel = perte_evitee_an + revenu_energie_an + revenu_coalition_an
    ebe_adapte_complet = ebe_projete_moy + gain_total_annuel
    temps_retour_ans = capex_net / max(gain_total_annuel, 1.0)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAPEX Net Agriculteur", f"{capex_net:,.0f} €".replace(",", " "), delta=f"-{subv_capex_pct}% subventionné", delta_color="normal")
    m2.metric("Pertes Climatiques Évitées", f"+{perte_evitee_an:,.0f} €/an".replace(",", " "), delta=f"{reduction_choc_pct*100:.0f}% du risque absorbé")
    m3.metric("Revenus Énergie + Coalition", f"+{revenu_energie_an + revenu_coalition_an:,.0f} €/an".replace(",", " "), delta=f"{tonnes_co2_an:.0f} tCO2/an séquestrées")
    m4.metric(
        "Nouvel EBE Sécurisé",
        f"{ebe_adapte_complet:,.0f} €/an".replace(",", " "),
        delta=f"ROI en {temps_retour_ans:.1f} ans"
    )

    # Simulation de la courbe de trésorerie sur 15 ans (avec effet "vallée de la mort" les 2 premières années de transition)
    annees = np.arange(0, 16)
    flux_inaction = [0.0]
    flux_adapt_seule = [-capex_brut]
    flux_adapt_coalition = [-capex_net]

    for an in annees[1:]:
        # Inaction : dégradation progressive de l'EBE
        deg_progressive = (ebe_projete_moy - ebe_initial) * min(1.0, an / 10.0)
        flux_inaction.append(flux_inaction[-1] + deg_progressive)

        # Montée en charge agronomique (transition de 2 ans avant plein effet)
        facteur_maturite = 0.4 if an == 1 else (0.75 if an == 2 else 1.0)
        benef_agro_energie = deg_progressive + (perte_evitee_an + revenu_energie_an) * facteur_maturite
        flux_adapt_seule.append(flux_adapt_seule[-1] + benef_agro_energie)
        flux_adapt_coalition.append(flux_adapt_coalition[-1] + benef_agro_energie + revenu_coalition_an)

    df_cashflow = pd.DataFrame({
        "Année": np.tile(annees, 3),
        "Trésorerie Cumulée Différentielle (€)": np.concatenate([flux_inaction, flux_adapt_seule, flux_adapt_coalition]),
        "Trajectoire": (
            ["1. Statu Quo (Inaction climatique)"] * len(annees)
            + ["2. Adaptation autofinancée seule (Vallée de la mort)"] * len(annees)
            + ["3. Adaptation + Hub Énergie + Coalition B2B2F (PSE / CSRD / Carbone)"] * len(annees)
        )
    })

    fig_cf = px.line(
        df_cashflow,
        x="Année",
        y="Trésorerie Cumulée Différentielle (€)",
        color="Trajectoire",
        markers=True,
        color_discrete_map={
            "1. Statu Quo (Inaction climatique)": "#e11d48",
            "2. Adaptation autofinancée seule (Vallée de la mort)": "#f59e0b",
            "3. Adaptation + Hub Énergie + Coalition B2B2F (PSE / CSRD / Carbone)": "#0d9488"
        },
        title="Trajectoire Financière Cumulée sur 15 ans : Traverser la « Vallée de la Mort » grâce à l'ingénierie financière",
        template="plotly_white"
    )
    fig_cf.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_cf.update_traces(line=dict(width=3))
    st.plotly_chart(fig_cf, use_container_width=True)

# --------------------------------------------
# ONGLET 6 : LIVRABLE & RAPPORT CLIENT (NOUVEAU)
# --------------------------------------------
with tab6:
    st.subheader("📄 Générateur Automatique de Diagnostic & Feuille de Route")
    st.markdown(
        "Synthèse prête à l'emploi pour un **comité d'investissement bancaire**, une **coopérative agricole**, "
        "une **direction RSE/Achats agroalimentaire (Scope 3 CSRD)** ou un **exploitant**."
    )

    solutions_liste_str = "\n".join([f"- {s}" for s in solutions_choisies]) if solutions_choisies else "- Aucun levier sélectionné"
    rapport_md = f"""# DIAGNOSTIC DE RÉSILIENCE CLIMATIQUE & INGÉNIERIE D'ADAPTATION
**Localisation analysée :** {adr_full} ({u_lat:.3f}°N, {u_lon:.3f}°E)
**Scénario & Horizon de référence :** {choix_scenario} — {choix_horizon}
**Filière étudiée :** {filiere} ({surface_ha} ha)

---

### 1. Exposition Physique Locale (Interpolation sur les 5 mailles les plus proches)
- **Indicateur directeur :** {format_func_clean(var_aléa_ref)}
- **Valeur locale projetée :** {val_locale_stress:.2f} {extraire_unite(descriptions.get(var_aléa_ref, ""))} (Moyenne nationale : {national_means[choix_scenario].get(var_aléa_ref, 0):.2f})
- **Perte de rendement agronomique estimée :** -{perte_rendement_moy_pct:.1f}% en année moyenne (-{perte_rendement_crise_pct:.1f}% en année décennale extrême)

### 2. Stress-Test Financier (Coût de l'Inaction)
- **Chiffre d'Affaires historique :** {ca_initial:,.0f} € / an
- **EBE / EBITDA historique :** {ebe_initial:,.0f} € / an ({marge_ebe_pct}% du CA)
- **EBE projeté sans adaptation :** {ebe_projete_moy:,.0f} € / an (soit une perte sèche de {ebe_initial - ebe_projete_moy:,.0f} € / an)
- **Statut de couverture de la dette ({annuite_dette:,.0f} €/an) :** {statut_dette}

### 3. Plan d'Adaptation Technico-Énergétique ({int(part_surface_adaptee*100)}% de la surface engagée)
{solutions_liste_str}

### 4. Montage Financier B2B2F & Retour sur Investissement
- **Investissement initial (CAPEX Brut) :** {capex_brut:,.0f} €
- **CAPEX Net après subventions publiques ({subv_capex_pct}%) :** {capex_net:,.0f} €
- **Pertes agricoles évitées :** +{perte_evitee_an:,.0f} € / an
- **Revenus énergétiques décentralisés (Biomasse / Agrivoltaïsme) :** +{revenu_energie_an:,.0f} € / an
- **Financements contractuels (PSE Eau + Prime Filière CSRD + Carbone {tonnes_co2_an:.0f} tCO2/an) :** +{revenu_coalition_an:,.0f} € / an
- **Nouvel EBE sécurisé :** {ebe_adapte_complet:,.0f} € / an
- **Temps de retour sur investissement (ROI) :** {temps_retour_ans:.1f} ans
""".replace(",", " ")

    st.markdown(f"<div class='highlight-box'>{rapport_md}</div>", unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Télécharger la Note de Synthèse Stratégique (.md)",
            data=rapport_md.encode("utf-8"),
            file_name=f"Diagnostic_Resilience_{adr_input.replace(' ', '_')}_{choix_horizon}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_dl2:
        # Export CSV de toutes les variables locales interpolées pour l'adresse
        rows_export = []
        for v in vars_disponibles:
            val_v = np.sum(voisins[v] * w) / np.sum(w)
            rows_export.append({
                "Code_Variable": v,
                "Description": descriptions.get(v, v),
                "Theme": cat_dict.get(v, {}).get("theme", ""),
                "Type": cat_dict.get(v, {}).get("type", ""),
                "Valeur_Locale": round(val_v, 3),
                "Moyenne_Nationale": round(national_means[choix_scenario].get(v, 0), 3),
                "Scenario": choix_scenario,
                "Periode": choix_horizon,
                "Adresse": adr_full
            })
        csv_bytes = pd.DataFrame(rows_export).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📊 Exporter les Données Climatiques Locales (.csv)",
            data=csv_bytes,
            file_name=f"Donnees_Climat_{adr_input.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
