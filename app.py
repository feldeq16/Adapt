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
# 1. CONFIGURATION
# ============================================
st.set_page_config(layout="wide", page_title="Observatoire Climatique", page_icon="🌍")

st.markdown("""
<style>
    /* Petit hack CSS pour espacer les catégories du dashboard */
    .stMetric {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #eee;
    }
    h4 {
        margin-top: 30px;
        color: #0068c9;
        border-bottom: 2px solid #f0f2f6;
        padding-bottom: 10px;
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
    """Extrait le texte entre la dernière parenthèse"""
    if not description: return ""
    match = re.search(r"\((.*?)\)$", description.strip())
    return match.group(1) if match else ""

def nettoyer_nom_variable(description):
    """Enlève l'unité à la fin pour l'affichage propre"""
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

    # Identification des variables numériques
    numeric_vars = [c for c in final_df.columns if c not in id_cols + latlon_cols and pd.api.types.is_numeric_dtype(final_df[c])]
    
    # Calcul des échelles globales (Min/Max)
    global_scales = {}
    for v in numeric_vars:
        vmin = final_df[v].min()
        vmax = final_df[v].max()
        global_scales[v] = (vmin, vmax)

    # Calcul des moyennes nationales par Scénario (Contexte) pour comparaison
    # On groupe par Contexte pour avoir la moyenne de chaque scénario
    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")

    return final_df, global_scales, national_means

# ============================================
# 3. CHARGEMENT
# ============================================

data, echelles_globales, national_means = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
categories = lire_dict_fichier(FICHIER_CATEGORIES)

if data is None:
    st.error("❌ Aucune donnée trouvée.")
    st.stop()

def format_func_var(option):
    desc = descriptions.get(option, "")
    if desc: return f"{option} - {desc[:40]}..."
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
# 5. CARTE & COULEURS
# ============================================

df_map = df_step1[df_step1["Période"] == choix_horizon].copy()

if choix_var not in df_map.columns or df_map[choix_var].isna().all():
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
    adr = st.text_input("📍 Rechercher une adresse", placeholder="Ex: Place du Capitole, Toulouse")
    u_lat, u_lon = None, None
    if adr:
        u_lat, u_lon = geocode_safe(adr)
        if not u_lat: st.warning("Adresse introuvable.")

# Récupération de la moyenne nationale pour ce scénario spécifique
moyenne_nat = national_means[choix_scenario].get(choix_var, 0)
desc_brute = descriptions.get(choix_var, choix_var)
unit_var = extraire_unite(desc_brute)

with col_kpi:
    st.metric(f"Moyenne France ({choix_scenario})", f"{moyenne_nat:.2f} {unit_var}")

# --- COULEURS INTELLIGENTES (Adaptative 0) ---
data_min, data_max = echelles_globales[choix_var]

# Logique : On étend l'échelle jusqu'à 0 pour garantir que Blanc = 0
# Si data = [10, 30] -> Echelle [0, 30] (Blanc -> Rouge)
# Si data = [-30, -10] -> Echelle [-30, 0] (Bleu -> Blanc)
# Si data = [-10, 10] -> Echelle [-10, 10] (Bleu -> Blanc -> Rouge)

vmin = min(data_min, 0)
vmax = max(data_max, 0)

# Edge case : Si tout est parfaitement à 0
if vmin == 0 and vmax == 0:
    vmin, vmax = -1, 1

norm_fixe = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
cmap = plt.get_cmap("coolwarm")

rgb = (cmap(norm_fixe(df_map[choix_var].values))[:, :3] * 255).astype(int)
df_map["r"], df_map["g"], df_map["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

# --- AFFICHAGE CARTE ---
layers = []
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
    tooltip={"html": f"<b>{choix_var}:</b> {{{choix_var}}} {unit_var}<br><i>{{Point}}</i>"}
))

# --- LÉGENDE ADAPTATIVE ---
col_leg1, col_leg2, col_leg3 = st.columns([1, 6, 1])
with col_leg2:
    st.caption(f"Légende : {desc_brute}")
    
    fig, ax = plt.subplots(figsize=(10, 0.4))
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_fixe, cmap=cmap), cax=ax, orientation='horizontal')
    cb.outline.set_visible(False)
    ax.set_axis_off()
    st.pyplot(fig)
    
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div style='text-align: left'><b>{vmin:.1f}</b></div>", unsafe_allow_html=True)
    
    # On n'affiche le 0 au milieu que s'il est pertinent (pas aux bords)
    # vmin < -0.1 et vmax > 0.1 sert à éviter d'afficher 0 si l'échelle est [0, 30] ou [-30, 0]
    if vmin < -0.1 and vmax > 0.1:
        c2.markdown(f"<div style='text-align: center'><b>0</b></div>", unsafe_allow_html=True)
    else:
        c2.write("")
        
    c3.markdown(f"<div style='text-align: right'><b>{vmax:.1f}</b></div>", unsafe_allow_html=True)


# ============================================
# 6. DASHBOARD ANALYTIQUE
# ============================================

if u_lat:
    st.divider()
    st.subheader(f"📍 Tableau de Bord Local : {adr}")
    
    # 1. Calculs
    df_map["dist_km"] = df_map.apply(
        lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
    )
    voisins = df_map.nsmallest(5, "dist_km")
    
    # Interpolation IDW pour la variable principale (FOCUS)
    weights = 1 / (voisins["dist_km"] + 0.01)**2
    val_focus = np.sum(voisins[choix_var] * weights) / np.sum(weights)

    # 2. Section FOCUS
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 30px;">
        <h3 style="margin:0; color:#31333F;">Focus : {nettoyer_nom_variable(desc_brute)}</h3>
        <h1 style="margin:0; font-size: 3em; color:#0068c9;">{val_focus:.2f} <span style="font-size: 0.5em; color:#666;">{unit_var}</span></h1>
        <p style="margin:0; color:#666;">Moyenne Nationale : {moyenne_nat:.2f} {unit_var} | Ecart : {val_focus - moyenne_nat:+.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. DASHBOARD GLOBAL (Cartes Métriques)
    st.subheader("📊 Indicateurs Complets")
    st.caption("Comparaison de la valeur locale (pixel le plus proche) avec la moyenne nationale du scénario.")

    # On utilise le pixel le plus proche pour afficher toutes les autres variables rapidement
    pixel_ref = voisins.iloc[0]
    
    # Préparation des groupes
    # On récupère toutes les variables dispos
    all_vars = [c for c in pixel_ref.index if c in echelles_globales]
    
    # Groupement par catégorie
    grouped_vars = {}
    for var in all_vars:
        cat = categories.get(var, "Autres Indicateurs")
        if cat not in grouped_vars: grouped_vars[cat] = []
        grouped_vars[cat].append(var)
    
    # Ordre des catégories (Logique vs Alphabétique)
    sorted_cats = sorted(grouped_vars.keys())
    if "Autres Indicateurs" in sorted_cats: # Mettre "Autres" à la fin
        sorted_cats.remove("Autres Indicateurs")
        sorted_cats.append("Autres Indicateurs")

    # Affichage du Dashboard
    for cat in sorted_cats:
        st.markdown(f"#### {cat}")
        
        vars_in_cat = grouped_vars[cat]
        
        # On crée une grille de 3 colonnes
        cols = st.columns(3)
        
        for i, var_code in enumerate(vars_in_cat):
            # Récupération des infos
            val_locale = pixel_ref[var_code]
            # Moyenne nationale pour ce scénario spécifique
            mean_nat = national_means[choix_scenario].get(var_code, 0)
            
            # Textes
            full_desc = descriptions.get(var_code, var_code)
            clean_name = nettoyer_nom_variable(full_desc)
            if len(clean_name) > 40: clean_name = clean_name[:37] + "..."
            
            unit = extraire_unite(full_desc)
            
            # Calcul Delta
            delta = val_locale - mean_nat
            
            # Affichage dans la colonne correspondante (modulo 3)
            with cols[i % 3]:
                st.metric(
                    label=clean_name,
                    value=f"{val_locale:.2f} {unit}",
                    delta=f"{delta:+.2f} vs Moy. Nat."
                )
