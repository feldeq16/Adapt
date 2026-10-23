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

# Décorateur de fragment compatible toutes versions de Streamlit (isole le rechargement d'un composant)
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
def trouver_variable_defaut(cat_dict, descriptions):
    """Recherche la variable correspondant à 'Écart de cumul de précipitation'."""
    meilleure_var = None
    meilleur_score = -1
    for var, info in cat_dict.items():
        desc_norm = normaliser_texte(descriptions.get(var, var))
        theme_norm = normaliser_texte(info.get("theme", ""))
        type_norm = normaliser_texte(info.get("type", ""))
        texte_complet = f"{desc_norm} {theme_norm} {type_norm} {normaliser_texte(var)}"
        
        score = 0
        if "ecart" in texte_complet:
            score += 4
        if "cumul" in texte_complet:
            score += 3
        if "precip" in texte_complet or "pluie" in texte_complet:
            score += 4
        if "annuel" in desc_norm:
            score += 1
            
        if score > meilleur_score:
            meilleur_score = score
            meilleure_var = var
            
    return meilleure_var if meilleure_var else next(iter(cat_dict.keys()), None)

@st.cache_data(show_spinner=False)
def geocoder_adresse(adr_query):
    """Interroge l'API Adresse une seule fois par texte saisi et garde le résultat en cache."""
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
    agg_dict = {c: "first" for c in combined.columns if c not in id_cols}
    final_df = combined.groupby(id_cols, as_index=False).agg(agg_dict)
    
    numeric_vars = [c for c in final_df.columns if c not in id_cols + ["Latitude", "Longitude"]]
    global_scales = {v: (float(final_df[v].min()), float(final_df[v].max())) for v in numeric_vars}
    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")
    
    # Extraction unique de la grille spatiale pour vectoriser les calculs de distance
    grille_points = final_df[["Point", "Latitude", "Longitude"]].drop_duplicates(subset=["Point"]).copy()
    return final_df, global_scales, national_means, grille_points

def haversine_vectorise(lat1, lon1, lats, lons):
    """Calcule la distance en km sur des milliers de points simultanément en NumPy (< 0.2 ms)."""
    r = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lats)
    dphi = np.radians(lats - lat1)
    dlambda = np.radians(lons - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

@st.cache_data(show_spinner=False)
def interpoler_toutes_variables_adresse(u_lat, u_lon, _data, _grille_points, numeric_vars):
    """
    Identifie les 5 points les plus proches en 0.2 ms et pré-calcule toutes les variables
    interpolées pour tous les scénarios et toutes les périodes en une seule opération matricielle.
    """
    dists = haversine_vectorise(u_lat, u_lon, _grille_points["Latitude"].to_numpy(), _grille_points["Longitude"].to_numpy())
    idx_top5 = np.argsort(dists)[:5]
    pts_top5 = _grille_points.iloc[idx_top5]["Point"].to_numpy()
    w_top5 = 1.0 / (dists[idx_top5] + 0.1) ** 2
    w_norm = w_top5 / np.sum(w_top5)
    poids_map = dict(zip(pts_top5, w_norm))

    # Filtrage ultra-rapide sur les 5 points uniquement (~30 à 50 lignes au lieu de 80 000)
    df_sub = _data[_data["Point"].isin(pts_top5)].copy()
    df_sub["_w"] = df_sub["Point"].map(poids_map)

    # Moyenne pondérée vectorisée par (Contexte, Période)
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

var_defaut = trouver_variable_defaut(cat_dict, descriptions)
theme_defaut = cat_dict[var_defaut]["theme"] if var_defaut in cat_dict else None
type_defaut = cat_dict[var_defaut]["type"] if var_defaut in cat_dict else None

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
    choix_horizon = st.selectbox("Période", periodes_dispo)
    df_map = df_step[df_step["Période"] == choix_horizon]

# Pré-calcul instantané (en cache) de toutes les valeurs locales pour l'adresse
df_interp_adresse = interpoler_toutes_variables_adresse(u_lat, u_lon, data, grille_points, vars_disponibles)
valeurs_locales_actuelles = df_interp_adresse.loc[(choix_scenario, choix_horizon)].to_dict()

# ============================================
# 4. ONGLETS ET COMPOSANTS ISOLÉS (@st.fragment)
# ============================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Carte",
    "📊 Dashboard",
    "📈 Graphique",
    "🌾 Impact Exploitation",
    "⚡ Adaptation & Financement",
    "📄 Synthèse"
])

# --------------------------------------------
# ONGLET 1 : CARTE (Isolé pour changement instantané)
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
        
        # Barre de couleur CSS native (affichage en 0 ms au lieu de Matplotlib)
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
    # Lecture directe depuis la table pré-calculée en cache (0 ms)
    df_traj = df_interp_adresse[[var_traj]].reset_index()
    df_traj.columns = ["Scénario", "Période", "Valeur"]
    df_traj["Période"] = df_traj["Période"].astype(str)
    
    fig_traj = px.line(
        df_traj, x="Période", y="Valeur", color="Scénario", markers=True,
        title=f"{format_func_clean(var_traj)} — {adr_full}",
        template="plotly_white"
    )
    fig_traj.update_traces(line=dict(width=2.5), marker=dict(size=8))
    st.plotly_chart(fig_traj, use_container_width=True)

with tab2:
    st.subheader(f"📍 Analyse locale : {adr_full}")

    with st.expander("Évolution multi-périodes et scénarios sur ce point", expanded=True):
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
# ONGLET 3 : GRAPHIQUE (Isolé avec @st.fragment)
# --------------------------------------------
@fragment_rapide
def afficher_onglet_graphique():
    st.subheader("📈 Comparateur de variables")
    col_x, col_y = st.columns(2)
    
    themes_list = sorted(list(set(v["theme"] for v in cat_dict.values())))
    idx_tx_def = themes_list.index(theme_defaut) if theme_defaut in themes_list else 0
    
    with col_x:
        st.markdown("**Axe X**")
        tx = st.selectbox("Thème X", themes_list, index=idx_tx_def, key="tx")
        types_x = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == tx)))
        idx_tyx_def = types_x.index(type_defaut) if type_defaut in types_x else 0
        tyx = st.selectbox("Type X", types_x, index=idx_tyx_def, key="tyx")
        vars_x = [k for k, v in cat_dict.items() if v["theme"] == tx and v["type"] == tyx]
        idx_vx_def = vars_x.index(var_defaut) if var_defaut in vars_x else 0
        var_x = st.selectbox("Variable X", vars_x, index=idx_vx_def, format_func=format_func_clean, key="vx")

    with col_y:
        st.markdown("**Axe Y**")
        idx_ty_def = 0 if idx_tx_def != 0 else min(1, len(themes_list) - 1)
        ty = st.selectbox("Thème Y", themes_list, index=idx_ty_def, key="ty")
        types_y = sorted(list(set(v["type"] for k, v in cat_dict.items() if v["theme"] == ty)))
        tyy = st.selectbox("Type Y", types_y, key="tyy")
        vars_y = [k for k, v in cat_dict.items() if v["theme"] == ty and v["type"] == tyy]
        idx_vy_def = min(1, len(vars_y) - 1) if (tx == ty and tyx == tyy and len(vars_y) > 1) else 0
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
        "Légende": ["📍 Votre Adresse"]
    })
    df_scat_plot = pd.concat([df_scat_plot, user_pt], ignore_index=True)

    fig = px.scatter(
        df_scat_plot,
        x="axe_x",
        y="axe_y",
        color="Légende",
        color_discrete_map={
            "Points du territoire": "rgba(100,100,100,0.3)",
            "📍 Votre Adresse": "#e74c3c"
        },
        labels={"axe_x": format_func_clean(var_x), "axe_y": format_func_clean(var_y)},
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")), selector=dict(mode="markers"))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    afficher_onglet_graphique()

# --------------------------------------------
# ONGLET 4 : IMPACT EXPLOITATION
# --------------------------------------------
with tab4:
    st.subheader(f"🌾 Simulation technico-économique — {adr_full}")

    profils_filieres = {
        "Grandes Cultures (Blé, Maïs, Colza, Orge, Betterave)": {"surf": 140, "ca_ha": 1650, "ebe_pct": 32, "dette_an": 42000, "sensibilite": 1.0},
        "Polyculture - Élevage": {"surf": 115, "ca_ha": 2100, "ebe_pct": 28, "dette_an": 38000, "sensibilite": 1.15},
        "Viticulture": {"surf": 20, "ca_ha": 18000, "ebe_pct": 36, "dette_an": 55000, "sensibilite": 1.35},
        "Arboriculture": {"surf": 35, "ca_ha": 9500, "ebe_pct": 30, "dette_an": 48000, "sensibilite": 1.30},
        "Maraîchage & Légumes plein champ": {"surf": 50, "ca_ha": 5800, "ebe_pct": 30, "dette_an": 36000, "sensibilite": 1.20}
    }

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        filiere = st.selectbox("Orientation (OTEX)", list(profils_filieres.keys()))
        p_def = profils_filieres[filiere]
        surface_ha = st.number_input("Surface Agricole Utile - SAU (ha)", min_value=1, max_value=2000, value=p_def["surf"], step=5)
    with col_p2:
        ca_par_ha = st.number_input("Produit brut moyen (€ / ha)", min_value=200, max_value=80000, value=p_def["ca_ha"], step=100)
        marge_ebe_pct = st.slider("Taux d'EBE historique (% du CA)", min_value=10, max_value=65, value=p_def["ebe_pct"])
    with col_p3:
        var_cible_stress = st.session_state.get("var_active", var_defaut)
        idx_stress = vars_disponibles.index(var_cible_stress) if var_cible_stress in vars_disponibles else 0
        var_alea_ref = st.selectbox(
            "Variable climatique de référence",
            vars_disponibles,
            index=idx_stress,
            format_func=format_func_clean
        )
        annuite_dette = st.number_input("Annuités d'emprunt (€ / an)", min_value=0, max_value=500000, value=p_def["dette_an"], step=2000)

    val_locale_stress = float(valeurs_locales_actuelles.get(var_alea_ref, 0.0))
    vmin_s, vmax_s = echelles_globales.get(var_alea_ref, (0.0, 1.0))
    span_s = max(abs(vmax_s - vmin_s), 1e-4)
    indice_exposition_local = np.clip(abs(val_locale_stress - vmin_s) / span_s, 0.08, 0.95)
    
    perte_rendement_moy_pct = round(indice_exposition_local * 24.0 * p_def["sensibilite"], 1)
    perte_rendement_crise_pct = round(min(70.0, perte_rendement_moy_pct * 2.1), 1)

    ca_initial = surface_ha * ca_par_ha
    ebe_initial = ca_initial * (marge_ebe_pct / 100.0)

    perte_ca_moy = ca_initial * (perte_rendement_moy_pct / 100.0)
    ebe_projete_moy = ebe_initial - (perte_ca_moy * 0.78)
    perte_ca_crise = ca_initial * (perte_rendement_crise_pct / 100.0)
    ebe_projete_crise = ebe_initial - (perte_ca_crise * 0.85)

    solde_treso_proj = ebe_projete_moy - annuite_dette
    solde_treso_crise = ebe_projete_crise - annuite_dette

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Chiffre d'affaires initial", f"{ca_initial:,.0f} €/an".replace(",", " "))
    k2.metric(
        f"Variation de rendement ({choix_horizon})",
        f"-{perte_rendement_moy_pct:.1f} %",
        delta=f"-{perte_rendement_crise_pct:.1f}% en année extrême",
        delta_color="inverse"
    )
    k3.metric(
        "EBE projeté (sans adaptation)",
        f"{ebe_projete_moy:,.0f} €/an".replace(",", " "),
        delta=f"{ebe_projete_moy - ebe_initial:+,.0f} €/an vs réf.".replace(",", " ")
    )
    statut_dette = "Couvert" if solde_treso_proj > 15000 else ("Sous tension" if solde_treso_proj > 0 else "Déficitaire")
    k4.metric(
        "Solde après annuités",
        f"{solde_treso_proj:,.0f} €/an".replace(",", " "),
        delta=statut_dette,
        delta_color="normal" if solde_treso_proj > 0 else "inverse"
    )

    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        df_stress_bar = pd.DataFrame({
            "Situation": ["Référence historique", f"Moyenne ({choix_horizon})", f"Aléa décennal ({choix_horizon})"],
            "EBE (€)": [ebe_initial, ebe_projete_moy, ebe_projete_crise],
            "Annuités (€)": [annuite_dette, annuite_dette, annuite_dette]
        })
        fig_stress = go.Figure()
        fig_stress.add_trace(go.Bar(
            x=df_stress_bar["Situation"],
            y=df_stress_bar["EBE (€)"],
            name="EBE (€)",
            marker_color=["#0068c9", "#f59e0b", "#dc2626"],
            text=[f"{v:,.0f} €".replace(",", " ") for v in df_stress_bar["EBE (€)"]],
            textposition="auto"
        ))
        fig_stress.add_trace(go.Scatter(
            x=df_stress_bar["Situation"],
            y=df_stress_bar["Annuités (€)"],
            name="Annuités d'emprunt",
            mode="lines+markers",
            line=dict(color="#1f2937", width=2.5, dash="dash")
        ))
        fig_stress.update_layout(
            title="Évolution de l'EBE face aux annuités bancaires",
            template="plotly_white",
            yaxis_title="€ / an"
        )
        st.plotly_chart(fig_stress, use_container_width=True)

    with col_g2:
        st.markdown("#### Lecture financière")
        if solde_treso_proj <= 0:
            st.markdown(
                f"<div class='alert-box'><b>Risque de trésorerie :</b><br>"
                f"Pour le scénario <b>{choix_scenario} ({choix_horizon})</b>, la baisse d'EBE "
                f"(<b>{ebe_initial - ebe_projete_moy:,.0f} €/an</b>) ne permet plus d'assurer les annuités "
                f"et les prélèvements privés sans adaptation.</div>".replace(",", " "),
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='info-box'><b>Impact sur la marge :</b><br>"
                f"L'exploitation conserve un solde positif en année moyenne mais perd "
                f"<b>{(1 - ebe_projete_moy/max(ebe_initial,1))*100:.0f}% de son EBE</b>. "
                f"En année extrême, le solde après dette passe à <b>{solde_treso_crise:,.0f} €</b>.</div>".replace(",", " "),
                unsafe_allow_html=True
            )
        st.markdown(
            f"- **Ratio de couverture de la dette (DSCR) :** passe de **{ebe_initial/max(annuite_dette,1):.2f}x** à **{ebe_projete_moy/max(annuite_dette,1):.2f}x**.\n"
            f"- **Impact volumes amont/aval :** baisse moyenne estimée à **-{perte_rendement_moy_pct:.1f}%**."
        )

# --------------------------------------------
# ONGLET 5 : ADAPTATION & FINANCEMENT
# --------------------------------------------
with tab5:
    st.subheader("⚡ Leviers d'adaptation et plan de financement")

    catalogue_solutions = {
        "Agriculture de conservation des sols (couverts végétaux, semis direct, allongement des rotations)": {
            "capex_ha": 320, "gain_resilience_pct": 35, "revenu_comp_ha": 0, "tco2_ha_an": 1.4
        },
        "Hydrologie régénérative & agroforesterie (haies brise-vent, fascines, rétention d'eau)": {
            "capex_ha": 650, "gain_resilience_pct": 25, "revenu_comp_ha": 40, "tco2_ha_an": 1.8
        },
        "Irrigation de précision & pilotage par sondes tensiométriques": {
            "capex_ha": 1400, "gain_resilience_pct": 45, "revenu_comp_ha": -35, "tco2_ha_an": 0.1
        },
        "Agrivoltaïsme (protection climatique des cultures & partage de la rente énergétique)": {
            "capex_ha": 1800, "gain_resilience_pct": 40, "revenu_comp_ha": 420, "tco2_ha_an": 0.8
        },
        "Valorisation biomasse : pyrogazéification & amendement biochar (rétention hydrique)": {
            "capex_ha": 950, "gain_resilience_pct": 38, "revenu_comp_ha": 260, "tco2_ha_an": 2.5
        }
    }

    col_sol, col_fin = st.columns([3, 2])
    with col_sol:
        st.markdown("#### 1. Leviers agronomiques et énergétiques")
        solutions_choisies = st.multiselect(
            "Pratiques et équipements déployés :",
            list(catalogue_solutions.keys()),
            default=[
                "Agriculture de conservation des sols (couverts végétaux, semis direct, allongement des rotations)",
                "Hydrologie régénérative & agroforesterie (haies brise-vent, fascines, rétention d'eau)"
            ]
        )
        part_surface_adaptee = st.slider(
            "Part de la SAU engagée (%)",
            min_value=10, max_value=100, value=60, step=10
        ) / 100.0

    with col_fin:
        st.markdown("#### 2. Co-financements mobilisables")
        subv_capex_pct = st.slider("Subventions publiques à l'investissement (% CAPEX)", 0, 70, 35, 5)
        pse_eau_ha = st.slider("Paiements pour Services Environnementaux - PSE (€/ha/an)", 0, 200, 90, 5)
        prime_filiere_ha = st.slider("Prime filière bas-carbone / Scope 3 (€/ha/an)", 0, 250, 70, 10)
        prix_tco2 = st.slider("Valorisation Crédits Carbone (€/tCO2e)", 0, 150, 50, 5)

    surf_engagee = surface_ha * part_surface_adaptee
    capex_brut = sum(catalogue_solutions[s]["capex_ha"] for s in solutions_choisies) * surf_engagee
    capex_net = capex_brut * (1.0 - subv_capex_pct / 100.0)

    reduction_choc_pct = min(0.88, sum(catalogue_solutions[s]["gain_resilience_pct"] for s in solutions_choisies) / 100.0) * (0.4 + 0.6 * part_surface_adaptee)
    perte_evitee_an = (ebe_initial - ebe_projete_moy) * reduction_choc_pct

    revenu_energie_an = sum(catalogue_solutions[s]["revenu_comp_ha"] for s in solutions_choisies) * surf_engagee
    tonnes_co2_an = sum(catalogue_solutions[s]["tco2_ha_an"] for s in solutions_choisies) * surf_engagee
    revenu_coalition_an = surf_engagee * (pse_eau_ha + prime_filiere_ha) + tonnes_co2_an * prix_tco2

    gain_total_annuel = perte_evitee_an + revenu_energie_an + revenu_coalition_an
    ebe_adapte_complet = ebe_projete_moy + gain_total_annuel
    temps_retour_ans = capex_net / max(gain_total_annuel, 1.0)

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CAPEX net exploitant", f"{capex_net:,.0f} €".replace(",", " "), delta=f"-{subv_capex_pct}% d'aides")
    m2.metric("Pertes climatiques évitées", f"+{perte_evitee_an:,.0f} €/an".replace(",", " "), delta=f"{reduction_choc_pct*100:.0f}% du choc absorbé")
    m3.metric("Revenus complémentaires", f"+{revenu_energie_an + revenu_coalition_an:,.0f} €/an".replace(",", " "), delta=f"{tonnes_co2_an:.0f} tCO2e/an")
    m4.metric("EBE après adaptation", f"{ebe_adapte_complet:,.0f} €/an".replace(",", " "), delta=f"Retour en {temps_retour_ans:.1f} ans")

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
        "Scénario": (
            ["Statu quo (sans adaptation)"] * len(annees)
            + ["Adaptation autofinancée"] * len(annees)
            + ["Adaptation co-financée (Subventions + PSE + Filière + Carbone)"] * len(annees)
        )
    })

    fig_cf = px.line(
        df_cashflow,
        x="Année",
        y="Trésorerie cumulée différentielle (€)",
        color="Scénario",
        markers=True,
        color_discrete_map={
            "Statu quo (sans adaptation)": "#dc2626",
            "Adaptation autofinancée": "#f59e0b",
            "Adaptation co-financée (Subventions + PSE + Filière + Carbone)": "#0068c9"
        },
        title="Trésorerie cumulée différentielle sur 15 ans",
        template="plotly_white"
    )
    fig_cf.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_cf.update_traces(line=dict(width=2.5))
    st.plotly_chart(fig_cf, use_container_width=True)

# --------------------------------------------
# ONGLET 6 : SYNTHÈSE & EXPORT
# --------------------------------------------
with tab6:
    st.subheader("📄 Synthèse et export des données")

    solutions_liste_str = "\n".join([f"- {s}" for s in solutions_choisies]) if solutions_choisies else "- Aucun levier sélectionné"
    rapport_md = f"""# SYNTHÈSE D'ANALYSE AGRO-CLIMATIQUE
**Localisation :** {adr_full} ({u_lat:.3f}°N, {u_lon:.3f}°E)
**Scénario & Horizon :** {choix_scenario} — {choix_horizon}
**Orientation :** {filiere} ({surface_ha} ha)

---

### 1. Indicateur climatique de référence
- **Variable :** {format_func_clean(var_alea_ref)}
- **Valeur locale interpolée :** {val_locale_stress:.2f} {extraire_unite(descriptions.get(var_alea_ref, ""))} (Moyenne nationale : {national_means[choix_scenario].get(var_alea_ref, 0):.2f})
- **Impact rendement estimé :** -{perte_rendement_moy_pct:.1f}% en moyenne (-{perte_rendement_crise_pct:.1f}% en année extrême)

### 2. Situation économique de l'exploitation
- **Chiffre d'affaires initial :** {ca_initial:,.0f} € / an
- **EBE initial :** {ebe_initial:,.0f} € / an ({marge_ebe_pct}% du CA)
- **EBE projeté sans adaptation :** {ebe_projete_moy:,.0f} € / an
- **Solde après annuités ({annuite_dette:,.0f} €/an) :** {solde_treso_proj:,.0f} € / an ({statut_dette})

### 3. Plan d'adaptation ({int(part_surface_adaptee*100)}% de la SAU)
{solutions_liste_str}

### 4. Bilan financier après adaptation
- **CAPEX brut :** {capex_brut:,.0f} €
- **CAPEX net après aides ({subv_capex_pct}%) :** {capex_net:,.0f} €
- **Pertes évitées :** +{perte_evitee_an:,.0f} € / an
- **Revenus complémentaires (Énergie, PSE, Filière, Carbone) :** +{revenu_energie_an + revenu_coalition_an:,.0f} € / an
- **EBE après adaptation :** {ebe_adapte_complet:,.0f} € / an (Temps de retour : {temps_retour_ans:.1f} ans)
""".replace(",", " ")

    st.markdown(f"<div class='info-box'>{rapport_md}</div>", unsafe_allow_html=True)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Télécharger la synthèse (.md)",
            data=rapport_md.encode("utf-8"),
            file_name=f"Synthese_{adr_input.replace(' ', '_')}_{choix_horizon}.md",
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
                "Periode": choix_horizon,
                "Adresse": adr_full
            }
            for v in vars_disponibles
        ]
        csv_bytes = pd.DataFrame(rows_export).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📊 Exporter les indicateurs locaux (.csv)",
            data=csv_bytes,
            file_name=f"Indicateurs_Climat_{adr_input.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
