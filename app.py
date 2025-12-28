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

st.title("🌍 Observatoire Climatique Multi-Scénarios")
st.markdown("---")

DOSSIER = "Données"
FICHIER_DEFINITIONS = "name.txt"
FICHIER_CATEGORIES = "category.txt"

# ============================================
# 2. FONCTIONS UTILES
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
    match = re.search(r"\((.*?)\)$", description.strip())
    return match.group(1) if match else ""

def lire_fichier_data(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
    except:
        return None

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier): return None, None

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

    if not all_dfs: return None, None

    combined = pd.concat(all_dfs, ignore_index=True)
    agg_dict = {c: "first" for c in combined.columns if c not in id_cols}
    final_df = combined.groupby(id_cols, as_index=False).agg(agg_dict)

    numeric_vars = [c for c in final_df.columns if c not in id_cols + latlon_cols and pd.api.types.is_numeric_dtype(final_df[c])]
    
    global_scales = {}
    for v in numeric_vars:
        vmin = final_df[v].min()
        vmax = final_df[v].max()
        global_scales[v] = (vmin, vmax)

    return final_df, global_scales

# ============================================
# 3. CHARGEMENT
# ============================================

data, echelles_globales = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
categories = lire_dict_fichier(FICHIER_CATEGORIES)

if data is None:
    st.error("❌ Aucune donnée trouvée.")
    st.stop()

def format_func_var(option):
    desc = descriptions.get(option, "")
    if desc: return f"{option} - {desc[:50]}..."
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
    
    if choix_var in descriptions:
        st.info(f"**Définition :** {descriptions[choix_var]}")
    
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
# 5. CARTE & LÉGENDE
# ============================================

df_map = df_step1[df_step1["Période"] == choix_horizon].copy()

if choix_var not in df_map.columns or df_map[choix_var].isna().all():
    st.warning("Variable indisponible pour cette sélection.")
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
    adr = st.text_input("📍 Rechercher une adresse", placeholder="Ex: Toulouse, France")
    u_lat, u_lon = None, None
    if adr:
        u_lat, u_lon = geocode_safe(adr)
        if not u_lat: st.warning("Adresse introuvable.")

with col_kpi:
    avg_val = df_map[choix_var].mean()
    desc_courte = descriptions.get(choix_var, choix_var)
    unite = extraire_unite(desc_courte)
    st.metric(f"Moyenne Nationale", f"{avg_val:.2f} {unite}")

# --- COULEURS (Blanc = 0) ---
vmin_glob, vmax_glob = echelles_globales[choix_var]

# CORRECTION DU BUG : On force l'échelle à inclure 0 pour que TwoSlopeNorm fonctionne
# Si toutes les données sont positives (ex: 10 à 30), on force le min à un tout petit nombre négatif
if vmin_glob >= 0: 
    vmin_glob = -0.00001 # Juste un epsilon pour que 0 soit inclus
    # Optionnel : Si vous préférez une échelle symétrique (ex: -30 à +30), mettez : vmin_glob = -vmax_glob

# Si toutes les données sont négatives (ex: -10 à -2), on force le max à un tout petit nombre positif
if vmax_glob <= 0: 
    vmax_glob = 0.00001 
    # Optionnel : Si vous préférez une échelle symétrique, mettez : vmax_glob = -vmin_glob

# Sécurité si tout vaut exactement 0
if vmin_glob == 0 and vmax_glob == 0:
    vmin_glob, vmax_glob = -1, 1

# Maintenant on est sûr que vmin < 0 < vmax
norm_fixe = mcolors.TwoSlopeNorm(vmin=vmin_glob, vcenter=0, vmax=vmax_glob)
cmap = plt.get_cmap("coolwarm")

# Application des couleurs
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
    tooltip={"html": f"<b>{choix_var}:</b> {{{choix_var}}}<br><i>{{Point}}</i>"}
))

# --- LÉGENDE ---
col_leg1, col_leg2, col_leg3 = st.columns([1, 6, 1])
with col_leg2:
    # ... (code titre légende) ...
    
    fig, ax = plt.subplots(figsize=(10, 0.4))
    # On utilise bien norm_fixe qui contient les bornes corrigées
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm_fixe, cmap=cmap), cax=ax, orientation='horizontal')
    cb.outline.set_visible(False)
    ax.set_axis_off()
    st.pyplot(fig)
    
    c1, c2, c3 = st.columns(3)
    # Affichage propre des bornes
    c1.markdown(f"<div style='text-align: left'><b>{vmin_glob:.1f}</b></div>", unsafe_allow_html=True)
    c2.markdown(f"<div style='text-align: center'><b>0</b></div>", unsafe_allow_html=True)
    c3.markdown(f"<div style='text-align: right'><b>{vmax_glob:.1f}</b></div>", unsafe_allow_html=True)
    
# ============================================
# 6. ANALYSE DÉTAILLÉE
# ============================================

if u_lat:
    st.divider()
    st.subheader(f"📍 Analyse Locale : {adr}")
    
    # 1. Calculs
    df_map["dist_km"] = df_map.apply(
        lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
    )
    voisins = df_map.nsmallest(5, "dist_km")
    
    # Interpolation IDW de la variable sélectionnée
    weights = 1 / (voisins["dist_km"] + 0.01)**2
    val_est_var = np.sum(voisins[choix_var] * weights) / np.sum(weights)

    # 2. Mise en avant de la variable sélectionnée
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin:0; color:#31333F;">Focus : {descriptions.get(choix_var, choix_var)}</h3>
        <h1 style="margin:0; font-size: 3em; color:#0068c9;">{val_est_var:.2f} <span style="font-size: 0.5em; color:#666;">{unite}</span></h1>
        <p style="margin:0; color:#666;">Valeur interpolée basée sur les 5 stations les plus proches.</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Détail des voisins
    with st.expander("Voir le détail des 5 stations utilisées pour ce calcul"):
        cols_to_show = ["Point", choix_var, "dist_km"]
        st.dataframe(voisins[cols_to_show].style.format({choix_var: "{:.2f}", "dist_km": "{:.2f} km"}))

    # 4. TABLEAU COMPLET PAR CATÉGORIE
    st.divider()
    st.subheader("📊 Tableau de Bord Complet (Toutes variables)")
    
    # On récupère le pixel le plus proche pour afficher TOUTES les données de ce point
    pixel_ref = voisins.iloc[0] # Le plus proche
    
    # On prépare les données
    all_vars_data = []
    for col in pixel_ref.index:
        if col in echelles_globales: # Si c'est une variable climatique
            val = pixel_ref[col]
            cat = categories.get(col, "Non classé")
            desc = descriptions.get(col, col)
            unit = extraire_unite(desc)
            # On nettoie la description pour enlever l'unité si elle est à la fin
            desc_clean = desc.replace(f"({unit})", "").strip() if unit else desc
            
            all_vars_data.append({
                "Catégorie": cat,
                "Variable": col,
                "Description": desc_clean,
                "Valeur": val,
                "Unité": unit
            })
    
    df_all = pd.DataFrame(all_vars_data)
    
    # Affichage par sections catégorisées
    # On récupère la liste des catégories triées
    cats_uniques = sorted(df_all["Catégorie"].unique())
    # On met "Non classé" à la fin
    if "Non classé" in cats_uniques:
        cats_uniques.remove("Non classé")
        cats_uniques.append("Non classé")

    for cat in cats_uniques:
        st.markdown(f"#### {cat}")
        df_cat = df_all[df_all["Catégorie"] == cat].copy()
        
        # Petit tableau propre pour chaque catégorie
        # On cache l'index
        st.dataframe(
            df_cat[["Description", "Valeur", "Unité"]].style.format({"Valeur": "{:.2f}"}),
            use_container_width=True,
            hide_index=True
        )
