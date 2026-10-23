import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import re
import unicodedata
import requests

# ============================================
# 1. CONFIGURATION ET STYLE
# ============================================
st.set_page_config(layout="wide", page_title="Observatoire Climatique", page_icon="🌍")

fragment_rapide = getattr(st, "fragment", getattr(st, "experimental_fragment", lambda f: f))

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
        font-size: 0.95rem !important;
    }
    
    div[data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-size: 1.55rem !important;
    }

    /* Style des Onglets */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; flex-wrap: wrap; }
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
        margin-bottom: 6px;
        border-left: 3px solid #0068c9;
        padding-left: 10px;
    }

    .info-box {
        background-color: #f8fafc;
        border-left: 4px solid #0068c9;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
    }
    .alert-box {
        background-color: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 14px 18px;
        border-radius: 6px;
        margin-bottom: 14px;
    }
    .synth-card {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-top: 4px solid #0068c9;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌍 Observatoire Climatique")
st.markdown("---")

DOSSIER = "Données"
FICHIER_DEFINITIONS = "name.txt"
FICHIER_CATEGORIES = "category.txt"

# ============================================
# 2. FONCTIONS DE TRAITEMENT MISES EN CACHE
# ============================================

def normaliser_texte(texte):
    if not texte:
        return ""
    texte = unicodedata.normalize("NFD", str(texte))
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    return texte.lower().strip()

def renommer_horizon(val):
    """Remplace H1, H2, H3 par des horizons temporels explicites et triables chronologiquement."""
    s = str(val).strip()
    s_up = s.upper()
    mapping = {
        "REF": "1976-2005 (Référence)",
        "REFERENCE": "1976-2005 (Référence)",
        "RÉFÉRENCE": "1976-2005 (Référence)",
        "H1": "2021-2050 (Horizon proche)",
        "H2": "2041-2070 (Horizon moyen)",
        "H3": "2071-2100 (Horizon lointain)"
    }
    if s_up in mapping:
        return mapping[s_up]
    # Gestion si H1/H2/H3 est contenu dans une chaîne (ex: "Période H1")
    if re.search(r"\bH1\b", s_up):
        return "2021-2050 (Horizon proche)"
    if re.search(r"\bH2\b", s_up):
        return "2041-2070 (Horizon moyen)"
    if re.search(r"\bH3\b", s_up):
        return "2071-2100 (Horizon lointain)"
    return s

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner=False)
def trouver_variable_cible(cat_dict, descriptions, cible="ecart_precip"):
    """
    Recherche intelligente d'une variable dans name.txt et category.txt :
    - 'ecart_precip' -> Ecart de cumul de précipitation (mm)
    - 'periode_secheresse' -> Période de sécheresse (jour(s))
    """
    meilleure_var = None
    meilleur_score = -1

    for var, info in cat_dict.items():
        desc_brute = descriptions.get(var, var)
        desc_norm = normaliser_texte(desc_brute)
        theme_norm = normaliser_texte(info.get("theme", ""))
        type_norm = normaliser_texte(info.get("type", ""))
        texte_complet = f"{desc_norm} {theme_norm} {type_norm} {normaliser_texte(var)}"

        score = 0
        if cible == "ecart_precip":
            if "ecart de cumul de precipitation" in desc_norm:
                score += 20
            if "ecart" in texte_complet:
                score += 5
            if "cumul" in texte_complet:
                score += 4
            if "precip" in texte_complet or "pluie" in texte_complet:
                score += 5
            if "(mm)" in desc_norm:
                score += 2
        elif cible == "periode_secheresse":
            if "periode de secheresse" in desc_norm:
                score += 20
            if "periode" in desc_norm:
                score += 5
            if "secheresse" in texte_complet:
                score += 6
            if "jour" in desc_norm:
                score += 3
            if "ecart" not in desc_norm:
                score += 2

        if score > meilleur_score:
            meilleur_score = score
            meilleure_var = var

    return meilleure_var if meilleure_var else next(iter(cat_dict.keys()), None)

@st.cache_data(show_spinner=False)
def geocoder_adresse(adr_query):
    if adr_query and len(adr_query.strip()) > 2:
        try:
            r = requests.get(
                f"https://api-adresse.data.gouv.fr/search/?q={adr_query}&limit=1",
                timeout=3
            ).json()
            if r.get("features"):
                lon, lat = r["features"][0]["geometry"]["coordinates"]
                label = r["features"][0]["properties"]["label"]
                return float(lat), float(lon), label, True
        except Exception:
            pass
    return 50.3714, 2.3136, "Ramecourt 62130", False

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier):
        return None, None, None, None
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
        return None, None, None, None
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # Remplacement de H1, H2, H3 par les horizons temporels explicites dès le chargement
    combined["Période"] = combined["Période"].apply(renommer_horizon)
    
    agg_dict = {c: "first" for c in combined.columns if c not in id_cols}
    final_df = combined.groupby(id_cols, as_index=False).agg(agg_dict)
    
    numeric_vars = [c for c in final_df.columns if c not in id_cols + ["Latitude", "Longitude"]]
    global_scales = {v: (float(final_df[v].min()), float(final_df[v].max())) for v in numeric_vars}
    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")
    
    grille_points = final_df[["Point", "Latitude", "Longitude"]].drop_duplicates(subset=["Point"]).copy()
    return final_df, global_scales, national_means, grille_points

def haversine_vectorise(lat1, lon1, lats, lons):
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lats)
    dphi = np.radians(lats - lat1)
    dlambda = np.radians(lons - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

@st.cache_data(show_spinner=False)
def interpoler_toutes_variables_adresse(u_lat, u_lon, _data, _grille_points, numeric_vars):
    dists = haversine_vectorise(u_lat, u_lon, _grille_points["Latitude"].to_numpy(), _grille_points["Longitude"].to_numpy())
    idx_top5 = np.argsort(dists)[:5]
    pts_top5 = _grille_points.iloc[idx_top5]["Point"].to_numpy()
    w_top5 = 1.0 / (dists[idx_top5] + 0.1) ** 2
    w_norm = w_top5 / np.sum(w_top5)
    poids_map = dict(zip(pts_top5, w_norm))

    df_sub = _data[_data["Point"].isin(pts_top5)].copy()
    df_sub["_w"] = df_sub["Point"].map(poids_map)

    valeurs_ponderees = df_sub[numeric_vars].multiply(df_sub["_w"], axis=0)
    valeurs_ponderees["Contexte"] = df_sub["Contexte"]
    valeurs_ponderees["Période"] = df_sub["Période"]
    
    somme_ponderee = valeurs_ponderees.groupby(["Contexte", "Période"])[numeric_vars].sum()
    somme_poids = df_sub.groupby(["Contexte", "Période"])["_w"].sum()
    df_interp_global = somme_ponderee.div(somme_poids, axis=0)
    return df_interp_global

# ============================================
# 3. CHARGEMENT ET SIDEBAR
# ============================================
data, echelles_globales, national_means, grille_points = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
cat_dict = lire_dict_categories_double(FICHIER_CATEGORIES)

if data is None or data.empty:
    st.error("Le dossier 'Données' est introuvable ou vide.")
    st.stop()

vars_disponibles = [c for c in data.columns if c not in ["Point", "Contexte", "Période", "Latitude", "Longitude"]]
cat_dict_valide = {k: v for k, v in cat_dict.items() if k in vars_disponibles}
if not cat_dict_valide:
    cat_dict_valide = {v: {"theme": "Indicateurs", "type": "Valeurs"} for v in vars_disponibles}
cat_dict = cat_dict_valide

def format_func_clean(option):
    return descriptions.get(option, option)

# Variables par défaut :
# - Carte & Axe Y du graphique : Ecart de cumul de précipitation (mm)
# - Axe X du graphique : Période de sécheresse (jour(s))
var_defaut = trouver_variable_cible(cat_dict, descriptions, cible="ecart_precip")
theme_defaut = cat_dict[var_defaut]["theme"] if var_defaut in cat_dict else None
type_defaut = cat_dict[var_defaut]["type"] if var_defaut in cat_dict else None

var_secheresse_defaut = trouver_variable_cible(cat_dict, descriptions, cible="periode_secheresse")
theme_sech_defaut = cat_dict[var_secheresse_defaut]["theme"] if var_secheresse_defaut in cat_dict else None
type_sech_defaut = cat_dict[var_secheresse_defaut]["type"] if var_secheresse_defaut in cat_dict else None

if "var_active" not in st.session_state:
    st.session_state["var_active"] = var_defaut

with st.sidebar:
    st.header("🎛️ Paramètres")
    adr_input = st.text_input("📍 Rechercher une adresse", value="Ramecourt 62130", placeholder="Ex: Ramecourt 62130")
    u_lat, u_lon, adr_full, adr_ok = geocoder_adresse(adr_input)
    if adr_ok:
        st.success(f"Trouvé : {adr_full}")

    st.divider()
    scenarios_dispo = sorted(data["Contexte"].unique())
    choix_scenario = st.selectbox("Scénario RCP", scenarios_dispo)
    df_step = data[data["Contexte"] == choix_scenario]
    
    periodes_dispo = sorted(df_step["Période"].unique())
    choix_horizon = st.selectbox("Horizon temporel", periodes_dispo)
    df_map = df_step[df_step["Période"] == choix_horizon]

df_interp_adresse = interpoler_toutes_variables_adresse(u_lat, u_lon, data, grille_points, vars_disponibles)
valeurs_locales_actuelles = df_interp_adresse.loc[(choix_scenario, choix_horizon)].to_dict()

# ============================================
# 4. ONGLETS
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Carte",
    "📊 Dashboard",
    "📈 Graphique",
    "🌾 Impact Cultures & Finances",
    "⚡ Adaptation & Financement",
    "📄 Synthèse"
])

# --------------------------------------------
# ONGLET 1 : CARTE
# --------------------------------------------
@fragment_rapide
def afficher_onglet_carte():
    col_sel, col_viz = st.columns([1, 3])
    with col_sel:
        st.markdown("#### Sélection")
        themes_map = sorted(list(set(v["theme"] for v in cat_dict.values())))
        idx_t_map = themes_map.index(theme_defaut) if theme_defaut in themes_map else 0
        t_map = st.selectbox("Thématique", themes_map, index=idx_t_map, key="t_map")
        
        types_map = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == t_map)))
        idx_ty_map = types_map.index(type_defaut) if type_defaut in types_map else 0
        ty_map = st.selectbox("Type de donnée", types_map, index=idx_ty_map, key="ty_map")
        
        vars_map = [k for k, v in cat_dict.items() if v["theme"] == t_map and v["type"] == ty_map]
        idx_var_map = vars_map.index(var_defaut) if var_defaut in vars_map else 0
        choix_var = st.selectbox("Indicateur", vars_map, index=idx_var_map, format_func=format_func_clean, key="sel_var_carte")
        st.session_state["var_active"] = choix_var
        
        style_map = st.selectbox("Fond de carte", ["Voyager", "Clair", "Sombre"])
        styles = {
            "Clair": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            "Sombre": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
            "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json"
        }

        val_loc_carte = float(valeurs_locales_actuelles.get(choix_var, 0.0))
        unit_carte = extraire_unite(descriptions.get(choix_var, ""))
        st.divider()
        st.metric(
            label=f"📍 {adr_full}",
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
            css_grad = "linear-gradient(to right, #fff5f0, #fb6a4a, #67000d)"
        elif vmax <= 0:
            norm = mcolors.Normalize(vmin=vmin, vmax=0)
            cmap = plt.get_cmap("Blues_r")
            css_grad = "linear-gradient(to right, #08306b, #6baed6, #f7fbff)"
        else:
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cmap = plt.get_cmap("coolwarm")
            css_grad = "linear-gradient(to right, #3b4cc0, #f7f7f7, #b40426)"

        df_plot = df_map[["Longitude", "Latitude", choix_var]].dropna().copy()
        colors = (cmap(norm(df_plot[choix_var].to_numpy()))[:, :3] * 255).astype(np.uint8)
        df_plot["r"], df_plot["g"], df_plot["b"] = colors[:, 0], colors[:, 1], colors[:, 2]

        layer = pdk.Layer(
            "GridCellLayer",
            df_plot,
            get_position="[Longitude, Latitude]",
            get_color="[r, g, b, 180]",
            cell_size=8000,
            pickable=True
        )
        df_pin = pd.DataFrame([{
            "Latitude": u_lat,
            "Longitude": u_lon,
            choix_var: round(val_loc_carte, 2)
        }])
        pin_layer = pdk.Layer(
            "ScatterplotLayer",
            df_pin,
            get_position="[Longitude, Latitude]",
            get_fill_color="[0, 104, 201, 255]",
            get_line_color="[255, 255, 255, 255]",
            get_radius=5500,
            stroked=True,
            line_width_min_pixels=2,
            pickable=True
        )

        view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=7.5 if adr_input else 5)
        st.pydeck_chart(pdk.Deck(
            map_style=styles[style_map],
            initial_view_state=view,
            layers=[layer, pin_layer],
            tooltip={"html": f"<b>{format_func_clean(choix_var)}</b>: {{{choix_var}}}"}
        ))
        
        st.markdown(
            f"<div style='height:14px; width:100%; background:{css_grad}; border-radius:4px; border:1px solid #d1d5db; margin-top:6px;'></div>",
            unsafe_allow_html=True
        )
        c_min, c_mid, c_max = st.columns(3)
        c_min.markdown(f"<small>Min: {vmin:.1f}</small>", unsafe_allow_html=True)
        if vmin < 0 < vmax:
            c_mid.markdown("<center><small>0 (Blanc)</small></center>", unsafe_allow_html=True)
        c_max.markdown(f"<div style='text-align:right'><small>Max: {vmax:.1f}</small></div>", unsafe_allow_html=True)

with tab1:
    afficher_onglet_carte()

# --------------------------------------------
# ONGLET 2 : DASHBOARD
# --------------------------------------------
@fragment_rapide
def afficher_graphe_trajectoire():
    var_cible = st.session_state.get("var_active", var_defaut)
    idx_traj = vars_disponibles.index(var_cible) if var_cible in vars_disponibles else 0
    var_traj = st.selectbox(
        "Indicateur",
        vars_disponibles,
        index=idx_traj,
        format_func=format_func_clean,
        key="var_traj_tab2"
    )
    df_traj = df_interp_adresse[[var_traj]].reset_index()
    df_traj.columns = ["Scénario", "Horizon temporel", "Valeur"]
    df_traj = df_traj.sort_values(["Scénario", "Horizon temporel"])
    
    fig_traj = px.line(
        df_traj, x="Horizon temporel", y="Valeur", color="Scénario", markers=True,
        title=f"{format_func_clean(var_traj)} — {adr_full}",
        template="plotly_white"
    )
    fig_traj.update_traces(line=dict(width=2.5), marker=dict(size=8))
    st.plotly_chart(fig_traj, use_container_width=True)

with tab2:
    st.subheader(f"📍 Analyse locale : {adr_full}")

    with st.expander("Évolution selon les horizons temporels et scénarios RCP", expanded=True):
        afficher_graphe_trajectoire()

    hierarchy = {}
    for var, info in cat_dict.items():
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
                val_interp = float(valeurs_locales_actuelles.get(v, 0.0))
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
# ONGLET 3 : GRAPHIQUE
# X par défaut : Période de sécheresse (jour(s))
# Y par défaut : Ecart de cumul de précipitation (mm)
# --------------------------------------------
@fragment_rapide
def afficher_onglet_graphique():
    st.subheader("📈 Comparateur de variables")
    col_x, col_y = st.columns(2)
    
    themes_list = sorted(list(set(v["theme"] for v in cat_dict.values())))
    
    with col_x:
        st.markdown("**Axe X**")
        idx_tx_def = themes_list.index(theme_sech_defaut) if theme_sech_defaut in themes_list else 0
        tx = st.selectbox("Thème X", themes_list, index=idx_tx_def, key="tx")
        
        types_x = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == tx)))
        idx_tyx_def = types_x.index(type_sech_defaut) if type_sech_defaut in types_x else 0
        tyx = st.selectbox("Type X", types_x, index=idx_tyx_def, key="tyx")
        
        vars_x = [k for k, v in cat_dict.items() if v["theme"] == tx and v["type"] == tyx]
        idx_vx_def = vars_x.index(var_secheresse_defaut) if var_secheresse_defaut in vars_x else 0
        var_x = st.selectbox("Variable X", vars_x, index=idx_vx_def, format_func=format_func_clean, key="vx")

    with col_y:
        st.markdown("**Axe Y**")
        idx_ty_def = themes_list.index(theme_defaut) if theme_defaut in themes_list else 0
        ty = st.selectbox("Thème Y", themes_list, index=idx_ty_def, key="ty")
        
        types_y = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == ty)))
        idx_tyy_def = types_y.index(type_defaut) if type_defaut in types_y else 0
        tyy = st.selectbox("Type Y", types_y, index=idx_tyy_def, key="tyy")
        
        vars_y = [k for k, v in cat_dict.items() if v["theme"] == ty and v["type"] == tyy]
        idx_vy_def = vars_y.index(var_defaut) if var_defaut in vars_y else 0
        var_y = st.selectbox("Variable Y", vars_y, index=idx_vy_def, format_func=format_func_clean, key="vy")

    cols_subset = [var_x] if var_x == var_y else [var_x, var_y]
    df_scat = df_map[cols_subset].dropna()
    
    df_scat_plot = pd.DataFrame({
        "axe_x": df_scat[var_x].to_numpy(),
        "axe_y": df_scat[var_y].to_numpy(),
        "Légende": "Points du territoire"
    })
    
    val_x = float(valeurs_locales_actuelles.get(var_x, 0.0))
    val_y = float(valeurs_locales_actuelles.get(var_y, 0.0))
    user_pt = pd.DataFrame({
        "axe_x": [val_x],
        "axe_y": [val_y],
        "Légende": [f"📍 {adr_full}"]
    })
    df_scat_plot = pd.concat([df_scat_plot, user_pt], ignore_index=True)

    fig = px.scatter(
        df_scat_plot,
        x="axe_x",
        y="axe_y",
        color="Légende",
        color_discrete_map={
            "Points du territoire": "rgba(100,100,100,0.3)",
            f"📍 {adr_full}": "#e74c3c"
        },
        labels={"axe_x": format_func_clean(var_x), "axe_y": format_func_clean(var_y)},
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")), selector=dict(mode="markers"))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    afficher_onglet_graphique()

# --------------------------------------------
# ONGLET 4 : IMPACT CULTURES & FINANCES
# --------------------------------------------
with tab4:
    st.subheader(f"🌾 Diagnostic Agro-Climatique & Stress-Test d'Exploitation — {adr_full}")

    # Référentiel économique RICA / Agreste / Chambres d'Agriculture
    profils_filieres = {
        "Grandes Cultures & Cultures Industrielles (Blé, Betterave, Pomme de terre - Type Hauts-de-France)": {
            "surf": 135, "ca_ha": 2300, "ebe_pct": 33, "dette_an": 45000, "sensibilite": 1.10,
            "cultures": ["Blé tendre d'hiver", "Betterave sucrière / Pomme de terre", "Colza / Orge d'hiver"]
        },
        "Grandes Cultures Céréalières (Blé, Orge, Colza, Maïs, Tournesol)": {
            "surf": 140, "ca_ha": 1650, "ebe_pct": 32, "dette_an": 40000, "sensibilite": 1.00,
            "cultures": ["Blé tendre d'hiver", "Maïs grain / Tournesol", "Colza / Orge d'hiver"]
        },
        "Polyculture - Élevage (Bovins lait / Autonomie fourragère)": {
            "surf": 115, "ca_ha": 2150, "ebe_pct": 29, "dette_an": 38000, "sensibilite": 1.15,
            "cultures": ["Prairies & Fourrages (Luzerne/Ray-grass)", "Maïs ensilage", "Blé tendre d'hiver"]
        },
        "Viticulture (AOC / Haute valeur ajoutée)": {
            "surf": 20, "ca_ha": 18000, "ebe_pct": 36, "dette_an": 55000, "sensibilite": 1.35,
            "cultures": ["Vigne (Floraison & Véraison)"]
        },
        "Arboriculture & Maraîchage plein champ": {
            "surf": 40, "ca_ha": 8500, "ebe_pct": 30, "dette_an": 42000, "sensibilite": 1.25,
            "cultures": ["Vergers & Légumes plein champ"]
        }
    }

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        filiere = st.selectbox(
            "Orientation technico-économique (OTEX)",
            list(profils_filieres.keys()),
            help="Catégories issues du Réseau d'Information Comptable Agricole (RICA / Agreste)."
        )
        p_def = profils_filieres[filiere]
        surface_ha = st.number_input(
            "Surface Agricole Utile - SAU (ha)",
            min_value=1, max_value=2000, value=p_def["surf"], step=5,
            help="Moyenne nationale en grandes cultures : ~135 à 140 ha."
        )
    with col_p2:
        ca_par_ha = st.number_input(
            "Produit brut moyen (€ / ha / an)",
            min_value=200, max_value=80000, value=p_def["ca_ha"], step=100,
            help="Chiffre d'affaires par hectare (ventes de récoltes + aides découplées). Source : RICA / Chambres d'agriculture."
        )
        marge_ebe_pct = st.slider(
            "Ratio EBE / Produit brut historique (%)",
            min_value=10, max_value=65, value=p_def["ebe_pct"],
            help="L'Excédent Brut d'Exploitation (EBITDA) représente en moyenne 28% à 35% du produit brut en France (Source : RICA / Cerfrance)."
        )
    with col_p3:
        annuite_dette = st.number_input(
            "Annuités d'emprunt bancaire (€ / an)",
            min_value=0, max_value=500000, value=p_def["dette_an"], step=2000,
            help="Remboursement annuel du capital et des intérêts (matériel, foncier, bâtiments). Représente en moyenne 35% à 45% de l'EBE."
        )
        taux_mo_sol = st.slider(
            "Taux de Matière Organique du sol (% MO)",
            min_value=1.0, max_value=5.0, value=2.2, step=0.2,
            help="+1% de Matière Organique augmente la Réserve Utile du sol d'environ 16 mm (soit 160 m³ d'eau stockée/ha), ce qui amortit les sécheresses."
        )

    # Calcul de l'exposition agro-climatique locale (combine Sécheresse + Précipitations + Variable active)
    val_sech = float(valeurs_locales_actuelles.get(var_secheresse_defaut, 0.0))
    vmin_sech, vmax_sech = echelles_globales.get(var_secheresse_defaut, (0.0, 1.0))
    score_sech = np.clip(abs(val_sech - vmin_sech) / max(abs(vmax_sech - vmin_sech), 1e-4), 0.1, 0.95)

    val_prec = float(valeurs_locales_actuelles.get(var_defaut, 0.0))
    vmin_prec, vmax_prec = echelles_globales.get(var_defaut, (0.0, 1.0))
    score_prec = np.clip(abs(vmax_prec - val_prec) / max(abs(vmax_prec - vmin_prec), 1e-4), 0.1, 0.95)

    # Effet tampon agronomique de la Matière Organique (base référence = 2.0 % MO)
    gain_eau_mm = (taux_mo_sol - 2.0) * 16.0
    facteur_tampon_sol = np.clip(1.0 - (taux_mo_sol - 2.0) * 0.10, 0.70, 1.15)

    indice_exposition_local = (0.55 * score_sech + 0.45 * score_prec) * facteur_tampon_sol
    perte_rendement_moy_pct = round(indice_exposition_local * 22.0 * p_def["sensibilite"], 1)
    perte_rendement_crise_pct = round(min(65.0, perte_rendement_moy_pct * 2.0), 1)

    ca_initial = surface_ha * ca_par_ha
    ebe_initial = ca_initial * (marge_ebe_pct / 100.0)

    # Rigidité des charges : 80% des coûts (semences, engrais, fermage, matériel) sont déjà engagés avant l'aléa
    perte_ca_moy = ca_initial * (perte_rendement_moy_pct / 100.0)
    ebe_projete_moy = ebe_initial - (perte_ca_moy * 0.80)
    perte_ca_crise = ca_initial * (perte_rendement_crise_pct / 100.0)
    ebe_projete_crise = ebe_initial - (perte_ca_crise * 0.85)

    solde_treso_init = ebe_initial - annuite_dette
    solde_treso_proj = ebe_projete_moy - annuite_dette
    solde_treso_crise = ebe_projete_crise - annuite_dette

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Produit Brut (CA) Initial", f"{ca_initial:,.0f} €/an".replace(",", " "))
    k2.metric(
        f"Impact Rendement ({choix_horizon[:9]})",
        f"-{perte_rendement_moy_pct:.1f} %",
        delta=f"-{perte_rendement_crise_pct:.1f}% en année extrême",
        delta_color="inverse"
    )
    k3.metric(
        "EBE Projeté (Sans adaptation)",
        f"{ebe_projete_moy:,.0f} €/an".replace(",", " "),
        delta=f"{ebe_projete_moy - ebe_initial:+,.0f} €/an vs réf.".replace(",", " ")
    )
    statut_dette = "Couvert" if solde_treso_proj > 18000 else ("Sous tension" if solde_treso_proj > 0 else "Déficitaire")
    k4.metric(
        "Revenu dispo après annuités",
        f"{solde_treso_proj:,.0f} €/an".replace(",", " "),
        delta=f"{statut_dette} (Réf: {solde_treso_init:,.0f} €)".replace(",", " "),
        delta_color="normal" if solde_treso_proj > 15000 else "inverse"
    )

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        df_stress_bar = pd.DataFrame({
            "Situation": ["Référence historique", f"Moyenne {choix_horizon[:9]}", f"Année extrême {choix_horizon[:9]}"],
            "EBE (€)": [ebe_initial, ebe_projete_moy, ebe_projete_crise],
            "Annuités (€)": [annuite_dette, annuite_dette, annuite_dette]
        })
        fig_stress = go.Figure()
        fig_stress.add_trace(go.Bar(
            x=df_stress_bar["Situation"],
            y=df_stress_bar["EBE (€)"],
            name="EBE généré (€)",
            marker_color=["#0068c9", "#f59e0b", "#dc2626"],
            text=[f"{v:,.0f} €".replace(",", " ") for v in df_stress_bar["EBE (€)"]],
            textposition="auto"
        ))
        fig_stress.add_trace(go.Scatter(
            x=df_stress_bar["Situation"],
            y=df_stress_bar["Annuités (€)"],
            name="Seuil des annuités bancaires",
            mode="lines+markers",
            line=dict(color="#1f2937", width=2.5, dash="dash")
        ))
        fig_stress.update_layout(
            title="Effet ciseau : impact de la baisse de rendement sur l'EBE et le remboursement de la dette",
            template="plotly_white",
            yaxis_title="€ / an"
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    with col_g2:
        st.markdown("#### 💧 Réserve Utile du Sol & Mécanisme Financier")
        st.markdown(
            f"<div class='info-box'>"
            f"<b>Effet tampon du sol ({taux_mo_sol:.1f}% de Matière Organique) :</b><br>"
            f"Variation de réserve en eau utile : <b>{gain_eau_mm:+.0f} mm</b> "
            f"(soit <b>{gain_eau_mm * 10:+.0f} m³ d'eau/ha</b> stockés dans le sol).<br><br>"
            f"<b>Pourquoi l'EBE chute plus vite que le rendement ?</b><br>"
            f"En agriculture, ~80 % des charges (semences, engrais, fermage, mécanisation) sont engagées avant la récolte. "
            f"Une perte de <b>-{perte_rendement_moy_pct:.1f}%</b> de volume entraîne ici une chute de "
            f"<b>-{(1 - ebe_projete_moy/max(ebe_initial,1))*100:.0f}% d'EBE</b>."
            f"</div>",
            unsafe_allow_html=True
        )
        if solde_treso_proj < 20000:
            st.markdown(
                f"<div class='alert-box'><b>Tension sur la rémunération de l'exploitant :</b><br>"
                f"Une fois les annuités bancaires ({annuite_dette:,.0f} €) payées, il ne reste que "
                f"<b>{solde_treso_proj:,.0f} €/an</b> pour rémunérer le travail de l'agriculteur, payer les cotisations MSA et autofinancer la ferme.</div>".replace(",", " "),
                unsafe_allow_html=True
            )

    # Nouveau bloc : Matrice de vulnérabilité phénologique par culture
    st.markdown("#### 🌱 Calendrier de sensibilité phénologique des cultures aux aléas locaux")
    matrice_cultures = pd.DataFrame([
        {
            "Culture": "Blé tendre / Orge d'hiver",
            "Stade phénologique critique": "Montaison — Floraison — Remplissage du grain (Mai - Juin)",
            "Seuil physique de perte": "Échaudage thermique dès Tmax > 25-28°C & déficit hydrique printanier",
            "Levier agronomique d'adaptation": "Variétés plus précoces (esquive), semis sous couvert, réduction de la densité"
        },
        {
            "Culture": "Betterave / Pomme de terre",
            "Stade phénologique critique": "Tubérisation & Grossissement estival (Juillet - Août)",
            "Seuil physique de perte": "Arrêt de croissance dès Tmax > 30°C & sensibilité forte à la durée de sécheresse",
            "Levier agronomique d'adaptation": "Augmentation de la Matière Organique (+RU), binage, irrigation pilotée par sondes"
        },
        {
            "Culture": "Maïs / Tournesol / Sorgho",
            "Stade phénologique critique": "Floraison mâle/femelle (± 15 jours autour du 15 Juillet)",
            "Seuil physique de perte": "Avortement des grains en cas de stress hydrique + Tmax > 32°C à la floraison",
            "Levier agronomique d'adaptation": "Substitution partielle Maïs → Sorgho/Tournesol, indices de précocité adaptés"
        },
        {
            "Culture": "Colza & Légumineuses (Pois/Féverole)",
            "Stade phénologique critique": "Levée de fin d'été (Août-Sept) & Floraison (Avril-Mai)",
            "Seuil physique de perte": "Échec de levée si sécheresse en août ; avortement des siliques si gel tardif",
            "Levier agronomique d'adaptation": "Colza associé à des plantes compagnes (féverole/lentille), semis direct dès la moisson"
        },
        {
            "Culture": "Prairies & Fourrages",
            "Stade phénologique critique": "Pousse de printemps & Trou estival (Juin - Septembre)",
            "Seuil physique de perte": "Arrêt de pousse estivale (consommation précoce des stocks hivernaux)",
            "Levier agronomique d'adaptation": "Prairies multi-espèces à enracinement profond (Luzerne, Sainfoin, Dactyle, Fétuque)"
        }
    ])
    st.dataframe(matrice_cultures, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Sources et méthodologie des paramètres économiques (RICA / Agreste)"):
        st.markdown("""
        - **Produit brut et EBE :** Ordres de grandeur issus du *Réseau d'Information Comptable Agricole (RICA - Agreste)* et des observatoires *Cerfrance*. En grandes cultures et polyculture-élevage, le taux d'EBE/CA moyen se situe entre **28 % et 35 %**.
        - **Annuités et utilisation de l'EBE :** En comptabilité agricole, l'EBE sert à payer : (1) les **annuités d'emprunt** (~35 à 45 % de l'EBE), (2) les **prélèvements privés et cotisations MSA** de l'exploitant, et (3) l'**autofinancement**.
        - **Rigidité des charges :** Lorsqu'un aléa climatique survient en fin de printemps ou en été, **75 % à 85 % des charges annuelles** ont déjà été dépensées (semences, engrais, protection des cultures, carburant, fermage, assurances).
        """)

# --------------------------------------------
# ONGLET 5 : ADAPTATION & FINANCEMENT
# --------------------------------------------
with tab5:
    st.subheader("⚡ Leviers d'adaptation et plan de financement")

    catalogue_solutions = {
        "Agriculture de conservation des sols (couverts végétaux, semis direct, allongement des rotations)": {
            "capex_ha": 320, "gain_resilience_pct": 35, "revenu_comp_ha": 0, "tco2_ha_an": 1.3
        },
        "Hydrologie régénérative & agroforesterie (haies brise-vent, fascines, rétention d'eau)": {
            "capex_ha": 600, "gain_resilience_pct": 25, "revenu_comp_ha": 35, "tco2_ha_an": 1.6
        },
        "Diversification d'assolement (introduction de légumineuses / luzerne / sorgho sobres en intrants)": {
            "capex_ha": 180, "gain_resilience_pct": 28, "revenu_comp_ha": 45, "tco2_ha_an": 0.9
        },
        "Irrigation de précision & pilotage par sondes tensiométriques": {
            "capex_ha": 1350, "gain_resilience_pct": 42, "revenu_comp_ha": -30, "tco2_ha_an": 0.1
        },
        "Valorisation biomasse : pyrogazéification & amendement biochar (rétention hydrique + énergie)": {
            "capex_ha": 900, "gain_resilience_pct": 36, "revenu_comp_ha": 240, "tco2_ha_an": 2.4
        }
    }

    col_sol, col_fin = st.columns([3, 2])
    with col_sol:
        st.markdown("#### 1. Leviers agronomiques et énergétiques")
        solutions_choisies = st.multiselect(
            "Pratiques et équipements déployés sur l'exploitation :",
            list(catalogue_solutions.keys()),
            default=[
                "Agriculture de conservation des sols (couverts végétaux, semis direct, allongement des rotations)",
                "Diversification d'assolement (introduction de légumineuses / luzerne / sorgho sobres en intrants)"
            ]
        )
        part_surface_adaptee = st.slider(
            "Part de la SAU engagée (%)",
            min_value=10, max_value=100, value=65, step=5
        ) / 100.0

    with col_fin:
        st.markdown("#### 2. Aides publiques et co-financements privés")
        subv_capex_pct = st.slider(
            "Subventions investissement PCAE / Haies (% du CAPEX)",
            0, 70, 35, 5,
            help="Plan de Compétitivité et d'Adaptation des Exploitations (PCAE / FranceAgriMer / Régions) : taux standard de 30% à 40% sur le matériel d'adaptation, et jusqu'à 60-80% sur la plantation de haies."
        )
        pse_eau_ha = st.slider(
            "MAEC & PSE Agence de l'Eau (€ / ha engagé / an)",
            0, 200, 80, 5,
            help="Mesures Agro-Environnementales et Climatiques (PAC) et Paiements pour Services Environnementaux (Agences de l'Eau) : contrats de 5 ans rémunérés entre 60 € et 150 €/ha/an."
        )
        prime_filiere_ha = st.slider(
            "Prime filière aval Scope 3 CSRD (€ / ha engagé / an)",
            0, 250, 60, 10,
            help="Primes de transition versées par les coopératives et industriels agroalimentaires (sucreries, malteries, meuniers, laiteries) pour sécuriser leur approvisionnement bas-carbone."
        )
        prix_tco2 = st.slider(
            "Valorisation Label Bas Carbone (€ / tCO2e)",
            0, 150, 45, 5,
            help="Méthode Grandes Cultures / Haies du Label Bas Carbone : prix moyen observé sur le marché volontaire français entre 35 € et 55 €/tCO2e."
        )

    surf_engagee = surface_ha * part_surface_adaptee
    capex_brut = sum(catalogue_solutions[s]["capex_ha"] for s in solutions_choisies) * surf_engagee
    capex_net = capex_brut * (1.0 - subv_capex_pct / 100.0)

    reduction_choc_pct = min(0.85, sum(catalogue_solutions[s]["gain_resilience_pct"] for s in solutions_choisies) / 100.0) * (0.4 + 0.6 * part_surface_adaptee)
    perte_evitee_an = (ebe_initial - ebe_projete_moy) * reduction_choc_pct

    revenu_energie_an = sum(catalogue_solutions[s]["revenu_comp_ha"] for s in solutions_choisies) * surf_engagee
    tonnes_co2_an = sum(catalogue_solutions[s]["tco2_ha_an"] for s in solutions_choisies) * surf_engagee
    revenu_coalition_an = surf_engagee * (pse_eau_ha + prime_filiere_ha) + tonnes_co2_an * prix_tco2

    gain_total_annuel = perte_evitee_an + revenu_energie_an + revenu_coalition_an
    ebe_adapte_complet = ebe_projete_moy + gain_total_annuel
    solde_adapte_apres_dette = ebe_adapte_complet - annuite_dette
    temps_retour_ans = capex_net / max(gain_total_annuel, 1.0)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAPEX net exploitant", f"{capex_net:,.0f} €".replace(",", " "), delta=f"-{subv_capex_pct}% subventionné (PCAE)")
    m2.metric("Pertes climatiques évitées", f"+{perte_evitee_an:,.0f} €/an".replace(",", " "), delta=f"{reduction_choc_pct*100:.0f}% du choc absorbé")
    m3.metric("Aides & Revenus (PSE/CSRD/CO2)", f"+{revenu_energie_an + revenu_coalition_an:,.0f} €/an".replace(",", " "), delta=f"{tonnes_co2_an:.0f} tCO2e/an")
    m4.metric("EBE sécurisé après adaptation", f"{ebe_adapte_complet:,.0f} €/an".replace(",", " "), delta=f"Retour sur invest. : {temps_retour_ans:.1f} ans")

    annees = np.arange(0, 16)
    flux_inaction = [0.0]
    flux_adapt_seule = [-capex_brut]
    flux_adapt_coalition = [-capex_net]

    for an in annees[1:]:
        deg_progressive = (ebe_projete_moy - ebe_initial) * min(1.0, an / 10.0)
        flux_inaction.append(flux_inaction[-1] + deg_progressive)

        facteur_maturite = 0.4 if an == 1 else (0.75 if an == 2 else 1.0)
        benef_agro_energie = deg_progressive + (perte_evitee_an + revenu_energie_an) * facteur_maturite
        flux_adapt_seule.append(flux_adapt_seule[-1] + benef_agro_energie)
        flux_adapt_coalition.append(flux_adapt_coalition[-1] + benef_agro_energie + revenu_coalition_an)

    df_cashflow = pd.DataFrame({
        "Année": np.tile(annees, 3),
        "Trésorerie cumulée différentielle (€)": np.concatenate([flux_inaction, flux_adapt_seule, flux_adapt_coalition]),
        "Trajectoire": (
            ["1. Statu quo (sans adaptation)"] * len(annees)
            + ["2. Adaptation autofinancée seule (creux de trésorerie initial)"] * len(annees)
            + ["3. Adaptation co-financée (PCAE + PSE + Prime Filière + Label Bas Carbone)"] * len(annees)
        )
    })

    fig_cf = px.line(
        df_cashflow,
        x="Année",
        y="Trésorerie cumulée différentielle (€)",
        color="Trajectoire",
        markers=True,
        color_discrete_map={
            "1. Statu quo (sans adaptation)": "#dc2626",
            "2. Adaptation autofinancée seule (creux de trésorerie initial)": "#f59e0b",
            "3. Adaptation co-financée (PCAE + PSE + Prime Filière + Label Bas Carbone)": "#0068c9"
        },
        title="Trésorerie cumulée différentielle sur 15 ans : sécurisation de la transition (Années 1 à 3)",
        template="plotly_white"
    )
    fig_cf.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_cf.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig_cf, use_container_width=True)

    with st.expander("ℹ️ Détail des dispositifs d'aides publiques et privées mobilisés"):
        st.markdown("""
        - **PCAE (Plan de Compétitivité et d'Adaptation des Exploitations) & FranceAgriMer :** Subventionne de **30 % à 40 %** les investissements matériels d'adaptation (semoirs de semis direct, sondes et pilotage de l'irrigation, équipements de protection climatique) et jusqu'à **60-80 %** pour l'agroforesterie et les haies.
        - **MAEC & PSE (Paiements pour Services Environnementaux) :** Rémunération contractuelle sur 5 ans (**60 € à 150 €/ha/an**) pilotée par les Agences de l'Eau et la PAC pour la couverture des sols, la réduction d'intrants et l'infiltration de l'eau.
        - **Label Bas Carbone (Méthode Grandes Cultures & Haies) :** Génère **0,8 à 1,5 tCO₂e/ha/an** valorisées entre **35 € et 55 €/tCO₂e** auprès d'entreprises partenaires.
        - **Primes Filières Aval (Scope 3 CSRD) :** Bonus contractuel (**40 € à 100 €/ha/an**) versé par les coopératives et industriels agroalimentaires pour sécuriser leurs volumes d'approvisionnement et réduire leur empreinte Scope 3 FLAG.
        """)

# --------------------------------------------
# ONGLET 6 : SYNTHÈSE ENRICHIE & EXPORT
# --------------------------------------------
with tab6:
    st.subheader(f"📄 Synthèse Stratégique & Feuille de Route — {adr_full}")

    # Cartes visuelles de synthèse en 3 colonnes
    c_syn1, c_syn2, c_syn3 = st.columns(3)
    with c_syn1:
        st.markdown(
            f"<div class='synth-card'>"
            f"<h4>🌡️ 1. Exposition Climatique Locale</h4>"
            f"<p><b>Commune :</b> {adr_full}<br>"
            f"<b>Scénario :</b> {choix_scenario}<br>"
            f"<b>Horizon :</b> {choix_horizon}</p>"
            f"<hr style='margin:8px 0;'>"
            f"<p>• <b>{nettoyer_nom_variable(format_func_clean(var_defaut))} :</b> {val_prec:.1f} {extraire_unite(descriptions.get(var_defaut, ''))} "
            f"<i>(Moy. nat. : {national_means[choix_scenario].get(var_defaut, 0):.1f})</i><br>"
            f"• <b>{nettoyer_nom_variable(format_func_clean(var_secheresse_defaut))} :</b> {val_sech:.1f} {extraire_unite(descriptions.get(var_secheresse_defaut, ''))} "
            f"<i>(Moy. nat. : {national_means[choix_scenario].get(var_secheresse_defaut, 0):.1f})</i><br>"
            f"• <b>Choc de rendement estimé :</b> <span style='color:#dc2626;font-weight:bold;'>-{perte_rendement_moy_pct:.1f}%</span> en moyenne "
            f"(-{perte_rendement_crise_pct:.1f}% en année extrême)</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    with c_syn2:
        st.markdown(
            f"<div class='synth-card'>"
            f"<h4>💶 2. Vulnérabilité Financière (Statu Quo)</h4>"
            f"<p><b>Orientation :</b> {filiere[:38]}...<br>"
            f"<b>Surface (SAU) :</b> {surface_ha} ha<br>"
            f"<b>Produit Brut initial :</b> {ca_initial:,.0f} €/an</p>"
            f"<hr style='margin:8px 0;'>"
            f"<p>• <b>EBE historique :</b> {ebe_initial:,.0f} €/an ({marge_ebe_pct}% du CA)<br>"
            f"• <b>EBE projeté sans adaptation :</b> <span style='color:#dc2626;font-weight:bold;'>{ebe_projete_moy:,.0f} €/an</span> "
            f"({ebe_projete_moy - ebe_initial:+,.0f} €/an)<br>"
            f"• <b>Reste après annuités ({annuite_dette:,.0f} €) :</b> <b>{solde_treso_proj:,.0f} €/an</b> ({statut_dette})</p>"
            f"</div>".replace(",", " "),
            unsafe_allow_html=True
        )
    with c_syn3:
        st.markdown(
            f"<div class='synth-card'>"
            f"<h4>🚀 3. Bilan Après Adaptation & Aides</h4>"
            f"<p><b>Surface engagée :</b> {surf_engagee:.0f} ha ({int(part_surface_adaptee*100)}% SAU)<br>"
            f"<b>CAPEX Brut :</b> {capex_brut:,.0f} €<br>"
            f"<b>CAPEX Net (après PCAE {subv_capex_pct}%) :</b> <b>{capex_net:,.0f} €</b></p>"
            f"<hr style='margin:8px 0;'>"
            f"<p>• <b>Pertes évitées :</b> <span style='color:#0068c9;font-weight:bold;'>+{perte_evitee_an:,.0f} €/an</span><br>"
            f"• <b>Co-financements (PSE/Filière/CO2) :</b> <span style='color:#0068c9;font-weight:bold;'>+{revenu_energie_an + revenu_coalition_an:,.0f} €/an</span><br>"
            f"• <b>Nouvel EBE sécurisé :</b> <b>{ebe_adapte_complet:,.0f} €/an</b> (ROI : {temps_retour_ans:.1f} ans)</p>"
            f"</div>".replace(",", " "),
            unsafe_allow_html=True
        )

    st.markdown("#### 📊 Tableau Comparatif : Trajectoire de l'Exploitation")
    df_comp_synthese = pd.DataFrame([
        {
            "Indicateur Financier & Agronomique": "Produit Brut / Chiffre d'Affaires (€/an)",
            "1. Référence Historique": f"{ca_initial:,.0f} €".replace(",", " "),
            f"2. Sans Adaptation ({choix_horizon[:9]})": f"{ca_initial - perte_ca_moy:,.0f} €".replace(",", " "),
            f"3. Avec Adaptation & Co-financements ({choix_horizon[:9]})": f"{ca_initial - perte_ca_moy + perte_evitee_an + revenu_energie_an + revenu_coalition_an:,.0f} €".replace(",", " ")
        },
        {
            "Indicateur Financier & Agronomique": "Excédent Brut d'Exploitation - EBE (€/an)",
            "1. Référence Historique": f"{ebe_initial:,.0f} €".replace(",", " "),
            f"2. Sans Adaptation ({choix_horizon[:9]})": f"{ebe_projete_moy:,.0f} €".replace(",", " "),
            f"3. Avec Adaptation & Co-financements ({choix_horizon[:9]})": f"{ebe_adapte_complet:,.0f} €".replace(",", " ")
        },
        {
            "Indicateur Financier & Agronomique": "Annuités Bancaires (€/an)",
            "1. Référence Historique": f"{annuite_dette:,.0f} €".replace(",", " "),
            f"2. Sans Adaptation ({choix_horizon[:9]})": f"{annuite_dette:,.0f} €".replace(",", " "),
            f"3. Avec Adaptation & Co-financements ({choix_horizon[:9]})": f"{annuite_dette:,.0f} €".replace(",", " ")
        },
        {
            "Indicateur Financier & Agronomique": "Solde disponible (Rémunération exploitant + Autofinancement)",
            "1. Référence Historique": f"{solde_treso_init:,.0f} €".replace(",", " "),
            f"2. Sans Adaptation ({choix_horizon[:9]})": f"{solde_treso_proj:,.0f} €".replace(",", " "),
            f"3. Avec Adaptation & Co-financements ({choix_horizon[:9]})": f"{solde_adapte_apres_dette:,.0f} €".replace(",", " ")
        },
        {
            "Indicateur Financier & Agronomique": "Ratio de couverture de la dette bancaire (DSCR = EBE / Annuités)",
            "1. Référence Historique": f"{ebe_initial/max(annuite_dette,1):.2f}x",
            f"2. Sans Adaptation ({choix_horizon[:9]})": f"{ebe_projete_moy/max(annuite_dette,1):.2f}x",
            f"3. Avec Adaptation & Co-financements ({choix_horizon[:9]})": f"{ebe_adapte_complet/max(annuite_dette,1):.2f}x"
        }
    ])
    st.dataframe(df_comp_synthese, use_container_width=True, hide_index=True)

    st.markdown("#### 🗓️ Feuille de Route Opérationnelle Recommandée")
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.markdown(
            "<div class='info-box'><b>Phase 1 — Court terme (Année 1)</b><br>"
            "• Audit pédologique (Réserve Utile) et montage des dossiers <b>PCAE / MAEC / Label Bas Carbone</b>.<br>"
            "• Ajustement variétal (esquive précoce) et implantation systématique des couverts végétaux d'interculture.</div>",
            unsafe_allow_html=True
        )
    with col_r2:
        st.markdown(
            "<div class='info-box'><b>Phase 2 — Transition (Années 2 à 3)</b><br>"
            "• Réduction du travail du sol, introduction de légumineuses/cultures sobres dans la rotation.<br>"
            "• Perception des premières annuités <b>PSE Eau et Primes Filières</b> pour compenser la phase d'apprentissage.</div>",
            unsafe_allow_html=True
        )
    with col_r3:
        st.markdown(
            "<div class='info-box'><b>Phase 3 — Pleine maturité (Années 4 à 15)</b><br>"
            "• Hausse structurelle du taux de Matière Organique (+160 m³ d'eau/ha par % de MO gagné).<br>"
            "• Stabilisation de l'EBE, audit de vérification et versement des <b>Crédits Carbone</b>.</div>",
            unsafe_allow_html=True
        )

    solutions_liste_str = "\n".join([f"- {s}" for s in solutions_choisies]) if solutions_choisies else "- Aucun levier sélectionné"
    rapport_md = f"""# DIAGNOSTIC DE RÉSILIENCE AGRO-CLIMATIQUE & PLAN DE FINANCEMENT
**Localisation :** {adr_full} ({u_lat:.3f}°N, {u_lon:.3f}°E)
**Scénario & Horizon temporel :** {choix_scenario} — {choix_horizon}
**Orientation technico-économique :** {filiere} ({surface_ha} ha)

---

### 1. Exposition Climatique Locale (Interpolation sur les 5 mailles les plus proches)
- **{format_func_clean(var_defaut)} :** {val_prec:.2f} {extraire_unite(descriptions.get(var_defaut, ""))} (Moyenne nationale : {national_means[choix_scenario].get(var_defaut, 0):.2f})
- **{format_func_clean(var_secheresse_defaut)} :** {val_sech:.2f} {extraire_unite(descriptions.get(var_secheresse_defaut, ""))} (Moyenne nationale : {national_means[choix_scenario].get(var_secheresse_defaut, 0):.2f})
- **Taux de Matière Organique considéré :** {taux_mo_sol:.1f}% (Effet sur la Réserve Utile : {gain_eau_mm:+.0f} mm soit {gain_eau_mm*10:+.0f} m³ d'eau/ha)
- **Impact rendement agronomique estimé :** -{perte_rendement_moy_pct:.1f}% en moyenne (-{perte_rendement_crise_pct:.1f}% en année décennale extrême)

### 2. Stress-Test Financier (Coût de l'Inaction)
- **Produit Brut (Chiffre d'affaires) initial :** {ca_initial:,.0f} € / an ({ca_par_ha:,.0f} €/ha)
- **Excédent Brut d'Exploitation (EBE) historique :** {ebe_initial:,.0f} € / an ({marge_ebe_pct}% du produit brut)
- **EBE projeté sans adaptation :** {ebe_projete_moy:,.0f} € / an (soit une érosion de {ebe_projete_moy - ebe_initial:+,.0f} € / an)
- **Annuités bancaires :** {annuite_dette:,.0f} € / an (Ratio DSCR passant de {ebe_initial/max(annuite_dette,1):.2f}x à {ebe_projete_moy/max(annuite_dette,1):.2f}x)
- **Solde disponible après dette (avant adaptation) :** {solde_treso_proj:,.0f} € / an ({statut_dette})

### 3. Leviers d'Adaptation Déployés ({surf_engagee:.0f} ha engagés, soit {int(part_surface_adaptee*100)}% de la SAU)
{solutions_liste_str}

### 4. Plan de Financement & Retour sur Investissement
- **Investissement initial (CAPEX Brut) :** {capex_brut:,.0f} €
- **Subventions publiques mobilisées (PCAE / FranceAgriMer / Régions - {subv_capex_pct}%) :** -{capex_brut - capex_net:,.0f} €
- **CAPEX Net à la charge de l'exploitation :** {capex_net:,.0f} €
- **Pertes agricoles évitées :** +{perte_evitee_an:,.0f} € / an
- **Co-financements annuels (MAEC/PSE : {pse_eau_ha} €/ha + Prime Filière CSRD : {prime_filiere_ha} €/ha + Label Bas Carbone : {tonnes_co2_an:.0f} tCO2e à {prix_tco2} €/t) :** +{revenu_energie_an + revenu_coalition_an:,.0f} € / an
- **Nouvel EBE sécurisé après adaptation :** {ebe_adapte_complet:,.0f} € / an (Solde après dette : {solde_adapte_apres_dette:,.0f} € / an)
- **Temps de retour sur investissement (ROI) :** {temps_retour_ans:.1f} ans
""".replace(",", " ")

    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Télécharger le rapport de synthèse complet (.md)",
            data=rapport_md.encode("utf-8"),
            file_name=f"Synthese_AgroClimat_{adr_input.replace(' ', '_')}_{choix_horizon[:9]}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_dl2:
        rows_export = [
            {
                "Code_Variable": v,
                "Description": descriptions.get(v, v),
                "Theme": cat_dict.get(v, {}).get("theme", ""),
                "Type": cat_dict.get(v, {}).get("type", ""),
                "Valeur_Locale": round(float(valeurs_locales_actuelles.get(v, 0.0)), 3),
                "Moyenne_Nationale": round(national_means[choix_scenario].get(v, 0), 3),
                "Scenario": choix_scenario,
                "Horizon_Temporel": choix_horizon,
                "Adresse": adr_full
            }
            for v in vars_disponibles
        ]
        csv_bytes = pd.DataFrame(rows_export).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📊 Exporter tous les indicateurs climatiques locaux (.csv)",
            data=csv_bytes,
            file_name=f"Indicateurs_Climat_{adr_input.replace(' ', '_')}_{choix_horizon[:9]}.csv",
            mime="text/csv",
            use_container_width=True
        )
