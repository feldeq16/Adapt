import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import uuid
import re
import requests # Nécessaire pour la nouvelle barre de recherche API
from geopy.distance import geodesic

# ============================================
# 1. CONFIGURATION ET STYLE
# ============================================
st.set_page_config(layout="wide", page_title="Observatoire Climatique", page_icon="🌍")

# --- FIX CSS POUR LES ONGLETS ET STYLE GÉNÉRAL ---
st.markdown("""
<style>
    /* Style général */
    .main .block-container {padding-top: 2rem;}
    
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
    
    /* --- CORRECTION DES TABS (ONGLETS) --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding-bottom: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: #eef0f4; /* Gris clair par défaut */
        color: #555; /* Texte gris par défaut */
        border-radius: 6px 6px 0px 0px;
        padding: 10px 15px;
        border: none;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    /* Style de l'onglet sélectionné */
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important; /* Fond blanc forcé */
        color: #0068c9 !important; /* Texte Bleu forcé pour contraste */
        border-top: 3px solid #0068c9 !important;
        border-bottom: none;
        font-weight: bold;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.05);
    }
    /* Hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e6e9ef;
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
    return re.sub(r"\s*\(.*?\)$", "", description).strip()

def lire_fichier_data(path):
    try:
        return pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True)
    except:
        return None

def format_func_var(option):
    desc = descriptions.get(option, "")
    if desc: return f"{option} - {desc[:45]}..."
    return option

# --- NOUVELLE FONCTION DE RECHERCHE D'ADRESSE (API GOUV.FR) ---
# Cette fonction est beaucoup plus rapide et fiable pour la France
# et permet de récupérer des suggestions pour l'autocomplétion.
@st.cache_data(show_spinner=False, ttl=3600) # Cache les résultats pour 1h
def search_address_gouv(query):
    """Recherche des adresses via l'API Base Adresse Nationale (BAN)."""
    if not query or len(query) < 3:
        return []
    try:
        # URL de l'API gouv.fr (limit=5 résultats)
        url = f"https://api-adresse.data.gouv.fr/search/?q={query}&limit=5&autocomplete=1"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            # Retourne une liste de tuples : (Label lisible, Lat, Lon)
            results = []
            for feature in data.get('features', []):
                label = feature['properties'].get('label')
                coords = feature['geometry']['coordinates'] # [lon, lat]
                if label and coords:
                     # On stocke (Label pour l'affichage, Lat, Lon)
                    results.append((label, coords[1], coords[0]))
            return results
    except Exception as e:
        # En production, préférez les logs plutôt que les print
        print(f"Erreur API BAN : {e}")
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
        vmin = final_df[v].min()
        vmax = final_df[v].max()
        global_scales[v] = (vmin, vmax)
    national_means = final_df.groupby("Contexte")[numeric_vars].mean().to_dict(orient="index")
    return final_df, global_scales, national_means

data, echelles_globales, national_means = charger_donnees_globales(DOSSIER)
descriptions = lire_dict_fichier(FICHIER_DEFINITIONS)
categories = lire_dict_fichier(FICHIER_CATEGORIES)

if data is None:
    st.error("❌ Aucune donnée trouvée. Vérifiez le dossier 'Données'.")
    st.stop()

# ============================================
# 4. SIDEBAR (FILTRES & RECHERCHE ROBUSTE)
# ============================================
with st.sidebar:
    st.header("🎛️ Paramètres Globaux")
    
    # --- 1. Barre de recherche "Autocomplete" ---
    st.subheader("📍 Localisation")
    st.caption("Recherchez une adresse en France.")

    # Initialisation du session state pour la recherche
    if 'search_query' not in st.session_state: st.session_state.search_query = ''
    if 'selected_coords' not in st.session_state: st.session_state.selected_coords = (None, None)
    if 'final_address_label' not in st.session_state: st.session_state.final_address_label = None

    # 1.1 Zone de texte pour taper la requête
    query = st.text_input("Saisissez une adresse :", 
                          value=st.session_state.search_query, 
                          placeholder="Ex: 10 rue de la Paix, Paris",
                          key="input_address_query")

    # 1.2 Logique de suggestion (déclenchée si la requête change et > 3 caractères)
    suggestions = []
    if query and len(query) > 3:
        with st.spinner("Recherche..."):
            # Appel à la nouvelle fonction API
            suggestions = search_address_gouv(query)

    u_lat, u_lon = st.session_state.selected_coords
    adr_label = st.session_state.final_address_label

    # 1.3 Affichage des suggestions sous forme de Selectbox si disponibles
    if suggestions:
        # On crée un dictionnaire pour mapper le label aux coordonnées {Label: (Lat, Lon)}
        options_map = {item[0]: (item[1], item[2]) for item in suggestions}
        
        # Selectbox pour choisir parmi les résultats API
        selected_option = st.selectbox("Suggestions (cliquez pour valider) :", 
                                       options=list(options_map.keys()),
                                       key="selectbox_address_suggestions")
        
        if selected_option:
             # Mise à jour des variables finales si une sélection est faite
            u_lat, u_lon = options_map[selected_option]
            adr_label = selected_option
            # On stocke dans le session state pour la persistance entre les rafraîchissements
            st.session_state.selected_coords = (u_lat, u_lon)
            st.session_state.final_address_label = adr_label
            st.session_state.search_query = selected_option # Met à jour le champ texte principal
            
    # 1.4 Feedback visuel
    if u_lat and u_lon and adr_label:
        st.success(f"✅ Adresse validée :\n**{adr_label}**", icon="📍")
    elif query and not suggestions and len(query)>3:
         st.warning("Aucune adresse trouvée.")

    st.divider()

    # --- 2. Scénarios ---
    st.subheader("📅 Contexte Climatique")
    scenarios = sorted(data["Contexte"].unique())
    choix_scenario = st.selectbox("Scénario (RCP)", scenarios, help="Sélectionnez le scénario d'émission.")
    
    df_context = data[data["Contexte"] == choix_scenario]
    horizons = sorted(df_context["Période"].unique())
    choix_horizon = st.selectbox("Période / Horizon", horizons)

    # Préparation des données filtrées pour tous les onglets
    df_map = df_context[df_context["Période"] == choix_horizon].copy()
    
    # --- 3. Calcul d'Interpolation (Voisins) ---
    # Calculé une seule fois ici si une adresse est validée
    voisins = None
    weights = None
    if u_lat and not df_map.empty:
        df_map = df_map.dropna(subset=["Latitude", "Longitude"])
        # Calcul distance géodésique
        df_map["dist_km"] = df_map.apply(
            lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
        )
        # Prendre les 5 points de grille les plus proches
        voisins = df_map.nsmallest(5, "dist_km")
        # Pondération inverse du carré de la distance pour l'interpolation
        weights = 1 / (voisins["dist_km"] + 0.1)**2

# ============================================
# 5. ONGLETS PRINCIPAUX
# ============================================

tab_map, tab_dash, tab_scatter = st.tabs(["🗺️ Visualisateur Carte", "📊 Dashboard Adresse", "📈 Comparateur Variable"])

# --------------------------------------------
# ONGLET 1 : VISUALISATEUR DE CARTE
# --------------------------------------------
with tab_map:
    col_ctrl, col_map = st.columns([1, 3], gap="medium")
    
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
            st.warning("Aucune variable disponible pour ce filtre.")
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
            # Nettoyage des NaN pour l'affichage
            df_display = df_map.dropna(subset=["Latitude", "Longitude", choix_var]).copy()

            # --- COULEURS ---
            vmin_glob, vmax_glob = echelles_globales[choix_var]
            # Choix de la colormap en fonction de la plage de valeurs
            if vmin_glob >= 0:
                norm = mcolors.Normalize(vmin=0, vmax=vmax_glob)
                cmap = plt.get_cmap("Reds") # Valeurs positives
            elif vmax_glob <= 0:
                norm = mcolors.Normalize(vmin=vmin_glob, vmax=0)
                cmap = plt.get_cmap("Blues_r") # Valeurs négatives
            else:
                norm = mcolors.TwoSlopeNorm(vmin=vmin_glob, vcenter=0, vmax=vmax_glob)
                cmap = plt.get_cmap("coolwarm") # Valeurs mixtes (divergent)

            # Application des couleurs aux données
            rgb = (cmap(norm(df_display[choix_var].values))[:, :3] * 255).astype(int)
            df_display["r"], df_display["g"], df_display["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

            # --- COUCHES PYDECK ---
            layers = [
                pdk.Layer(
                    "GridCellLayer",
                    data=df_display,
                    get_position="[Longitude, Latitude]",
                    get_color="[r, g, b, 170]", # Transparence fixée à 170/255
                    cell_size=8000, # Taille des cellules en mètres
                    extruded=False,
                    pickable=True,
                    auto_highlight=True
                )
            ]

            # Ajout d'un marqueur pour l'adresse si définie
            if u_lat:
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame({"lat": [u_lat], "lon": [u_lon]}),
                    get_position="[lon, lat]",
                    get_color="[0, 255, 0]", # Vert vif
                    get_radius=4000,
                    stroked=True,
                    get_line_color=[0,0,0],
                    line_width_min_pixels=3,
                    pickable=False
                ))
                # Centrer la vue sur l'adresse
                view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=8, pitch=0)
            else:
                # Vue par défaut sur la France
                view = pdk.ViewState(latitude=46.6, longitude=2.0, zoom=5.5, pitch=0)
            
            desc_brute = descriptions.get(choix_var, choix_var)
            unit_var = extraire_unite(desc_brute)

            # Affichage de la carte
            st.pydeck_chart(pdk.Deck(
                map_style=styles_map[style_choisi],
                initial_view_state=view,
                layers=layers,
                tooltip={"html": f"<b>{choix_var}</b>: {{{choix_var}}} {unit_var}<br><i>Point de grille : {{Point}}</i>"}
            ))

            # --- LÉGENDE ---
            st.caption(f"Légende : {nettoyer_nom_variable(desc_brute)}")
            fig, ax = plt.subplots(figsize=(8, 0.3))
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
            cb.outline.set_visible(False)
            ax.set_axis_off()
            st.pyplot(fig, use_container_width=True)

# --------------------------------------------
# ONGLET 2 : DASHBOARD DONNÉES À L'ADRESSE
# --------------------------------------------
with tab_dash:
    if not u_lat or voisins is None:
        st.info("👋 Veuillez rechercher et valider une adresse dans la barre latérale (à gauche) pour afficher le tableau de bord local.")
    else:
        st.markdown(f"### 📍 Analyse locale interpolée")
        st.caption(f"Adresse : **{adr_label}**")
        st.caption(f"ℹ️ Données calculées par interpolation spatiale des 5 points de grille climatique les plus proches (Distance min: {voisins.iloc[0]['dist_km']:.1f} km).")
        st.divider()
        
        # Récupération des variables disponibles dans les données voisines
        pixel_ref = voisins.iloc[0]
        all_vars = [c for c in pixel_ref.index if c in echelles_globales]
        
        # Groupement des variables par catégorie
        grouped_vars = {}
        for var in all_vars:
            cat = categories.get(var, "Autres Indicateurs")
            if cat not in grouped_vars: grouped_vars[cat] = []
            grouped_vars[cat].append(var)
        
        # Tri des catégories pour l'affichage
        sorted_cats = sorted(grouped_vars.keys())
        if "Autres Indicateurs" in sorted_cats:
            sorted_cats.remove("Autres Indicateurs")
            sorted_cats.append("Autres Indicateurs")

        # Génération des sections par catégorie
        for cat in sorted_cats:
            st.markdown(f"#### {cat}")
            vars_in_cat = grouped_vars[cat]
            # Grille de 3 colonnes pour les métriques
            cols = st.columns(3, gap="large")
            
            for i, var_code in enumerate(vars_in_cat):
                # --- CALCUL DE L'INTERPOLATION PONDÉRÉE ---
                val_interp = np.sum(voisins[var_code] * weights) / np.sum(weights)
                mean_nat = national_means[choix_scenario].get(var_code, 0)
                
                # Préparation des textes
                full_desc = descriptions.get(var_code, var_code)
                clean_name = nettoyer_nom_variable(full_desc)
                unit = extraire_unite(full_desc)
                delta = val_interp - mean_nat
                
                # Affichage de la métrique
                with cols[i % 3]:
                    st.metric(
                        label=clean_name[:40] + ("..." if len(clean_name)>40 else ""),
                        value=f"{val_interp:.2f} {unit}",
                        delta=f"{delta:+.2f} par rapport à la Moy. Nat.",
                        delta_color="inverse", # Inverse les couleurs (vert si plus bas, rouge si plus haut)
                        help=full_desc
                    )
            st.markdown("<br>", unsafe_allow_html=True) # Espace entre catégories

# --------------------------------------------
# ONGLET 3 : COMPARATEUR (NUAGE DE POINTS)
# --------------------------------------------
with tab_scatter:
    st.markdown("### 📈 Analyse Croisée (Nuage de points)")
    st.markdown("Comparez deux variables climatiques sur l'ensemble du territoire et situez votre adresse (étoile rouge).")

    c1, c2 = st.columns(2)
    vars_scatter = sorted(list(echelles_globales.keys()))
    
    with c1:
        var_x = st.selectbox("Variable Axe X", vars_scatter, index=0, format_func=format_func_var, key="scat_x")
    with c2:
        # Sélection d'une variable différente par défaut pour l'axe Y si possible
        idx_y = 1 if len(vars_scatter) > 1 else 0
        var_y = st.selectbox("Variable Axe Y", vars_scatter, index=idx_y, format_func=format_func_var, key="scat_y")

    if var_x and var_y and not df_map.empty:
        # Nettoyage des données pour le graphique
        df_scat = df_map.dropna(subset=[var_x, var_y])

        fig_scat, ax_scat = plt.subplots(figsize=(10, 6))
        
        # 1. Affichage de TOUS les points de la grille (Gris/Bleu léger)
        ax_scat.scatter(
            df_scat[var_x], 
            df_scat[var_y], 
            alpha=0.3, 
            c="#4A90E2", # Bleu clair
            edgecolors='none', 
            s=35, 
            label="Points de grille France"
        )
        
        val_x_interp, val_y_interp = None, None

        # 2. Calcul et affichage du point interpolé pour l'ADRESSE
        if u_lat and voisins is not None:
            # Calcul des valeurs interpolées pour les 2 variables sélectionnées
            val_x_interp = np.sum(voisins[var_x] * weights) / np.sum(weights)
            val_y_interp = np.sum(voisins[var_y] * weights) / np.sum(weights)
            
            # Affichage du point spécifique (Étoile Rouge)
            ax_scat.scatter(
                [val_x_interp], 
                [val_y_interp], 
                c="#D32F2F", # Rouge vif
                s=250, # Grande taille
                marker="*", 
                edgecolors="black",
                linewidth=1.5,
                label="📍 Votre adresse (interpolée",
                zorder=10 # Au premier plan
            )
            # Lignes pointillées de repérage
            ax_scat.axvline(x=val_x_interp, color='#D32F2F', linestyle=':', alpha=0.6)
            ax_scat.axhline(y=val_y_interp, color='#D32F2F', linestyle=':', alpha=0.6)

        # Finalisation du style du graphique
        unit_x = extraire_unite(descriptions.get(var_x, ''))
        unit_y = extraire_unite(descriptions.get(var_y, ''))
        ax_scat.set_xlabel(f"{var_x} {(unit_x)}")
        ax_scat.set_ylabel(f"{var_y} {(unit_y)}")
        ax_scat.set_title(f"Corrélation : {var_x} vs {var_y}", fontsize=12, pad=15)
        ax_scat.grid(True, linestyle='--', alpha=0.4, color='#ccc')
        ax_scat.legend(frameon=True, fancybox=True, framealpha=0.9)
        
        # Affichage dans Streamlit
        st.pyplot(fig_scat, use_container_width=True)
        
        if u_lat and val_x_interp is not None:
             st.caption(f"Position de l'adresse : **X = {val_x_interp:.2f} {unit_x}** / **Y = {val_y_interp:.2f} {unit_y}**")
        else:
             st.info("Validez une adresse dans la barre latérale pour voir votre position sur ce graphique.")
    else:
        st.warning("Données insuffisantes pour afficher le graphique.")
