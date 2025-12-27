import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import unicodedata
import uuid
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Analyse Climatique Multi-Variables")
st.title("🌡️ Analyse Climatique Multi-Indicateurs")

DOSSIER_DONNEES = 'Données'

# --- OUTILS & FONCTIONS ---
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def trouver_header_et_lire(chemin):
    """Lecture robuste : Trouve le header et nettoie les colonnes."""
    encodages_a_tester = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodages_a_tester:
        try:
            with open(chemin, 'r', encoding=enc) as f:
                lignes = [f.readline() for _ in range(50)]
            header_row = None
            sep = ';'
            for i, ligne in enumerate(lignes):
                clean = remove_accents(ligne).lower()
                if 'latitude' in clean and 'longitude' in clean:
                    header_row = i
                    if ';' in ligne: sep = ';'
                    elif ',' in ligne: sep = ','
                    elif '\t' in ligne: sep = '\t'
                    break
            if header_row is not None:
                df = pd.read_csv(chemin, sep=sep, header=header_row, encoding=enc, engine='python')
                
                # Nettoyage des noms de colonnes
                df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
                
                # Standardisation des colonnes vitales
                map_rename = {}
                for col in df.columns:
                    c_low = remove_accents(col.lower())
                    if 'point' in c_low or 'station' in c_low: map_rename[col] = 'Point'
                    if 'latitude' in c_low: map_rename[col] = 'Latitude'
                    if 'longitude' in c_low: map_rename[col] = 'Longitude'
                
                if map_rename: df = df.rename(columns=map_rename)
                return df, None
        except: continue
    return None, "Erreur lecture"

@st.cache_data(ttl=3600, show_spinner=False)
def charger_donnees(dossier):
    all_data = pd.DataFrame()
    if not os.path.exists(dossier): return pd.DataFrame(), "Dossier introuvable"
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    barre = st.progress(0, text="Chargement et unification des données...")
    
    for i, fichier in enumerate(fichiers):
        barre.progress((i)/len(fichiers))
        path = os.path.join(dossier, fichier)
        df, err = trouver_header_et_lire(path)
        if err: continue
        
        # 1. Gestion Période / Horizon
        col_p = next((c for c in df.columns if 'eriode' in remove_accents(c.lower()) or 'horizon' in remove_accents(c.lower())), None)
        df['Horizon_Filter'] = df[col_p].astype(str).str.strip() if col_p else "Non défini"

        # 2. Gestion Scénario (RCP)
        n = fichier.lower()
        if "rcp2.6" in n or "rcp26" in n: df['Scenario'] = "RCP 2.6"
        elif "rcp4.5" in n or "rcp45" in n: df['Scenario'] = "RCP 4.5"
        elif "rcp8.5" in n or "rcp85" in n: df['Scenario'] = "RCP 8.5"
        else: df['Scenario'] = "Autre"

        # 3. Conversion Numérique Intelligente (Toutes les variables)
        # On identifie les colonnes qui NE SONT PAS des métadonnées
        cols_meta = ['Latitude', 'Longitude', 'Point', 'Horizon_Filter', 'Scenario']
        
        # On s'assure que Lat/Lon sont float
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            df = df.dropna(subset=['Latitude', 'Longitude'])
            
            # Pour toutes les autres colonnes, on essaie de convertir en nombres
            for col in df.columns:
                if col not in cols_meta:
                    # Si c'est du texte, on tente de remplacer les virgules
                    if df[col].dtype == object:
                        try:
                            # On tente la conversion
                            converted = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
                            # Si la conversion donne trop de NaN (>50%), ce n'est pas une variable météo (ex: commentaire)
                            if converted.isna().mean() < 0.5:
                                df[col] = converted
                        except:
                            pass # On laisse tel quel
                    else:
                        # Déjà numérique, on garde
                        pass

            all_data = pd.concat([all_data, df], ignore_index=True)
            
    barre.empty()
    return all_data, None

# --- GÉOCODAGE ---
@st.cache_data
def geocode_address(address):
    try:
        unique_agent = f"climate_app_{uuid.uuid4()}"
        geolocator = Nominatim(user_agent=unique_agent)
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        return None, None

def interpoler_toutes_variables(lat, lon, df, n=5):
    """
    Calcule l'interpolation IDW pour TOUTES les colonnes numériques trouvées.
    """
    df = df.copy()
    
    # 1. Trouver les voisins
    df['d_approx'] = (df['Latitude']-lat)**2 + (df['Longitude']-lon)**2
    neighbors = df.nsmallest(n, 'd_approx').copy()
    
    # 2. Distance précise
    neighbors['dist_km'] = neighbors.apply(lambda r: geodesic((lat,lon), (r['Latitude'], r['Longitude'])).km, axis=1)
    
    # 3. Poids
    neighbors['w'] = 1 / (neighbors['dist_km'] + 0.001)**2
    sum_w = neighbors['w'].sum()
    
    # 4. Interpolation pour chaque colonne numérique (sauf meta)
    cols_meta = ['Latitude', 'Longitude', 'Point', 'Horizon_Filter', 'Scenario', 'dist_km', 'd_approx', 'w']
    resultats = {}
    
    for col in neighbors.columns:
        if col not in cols_meta and pd.api.types.is_numeric_dtype(neighbors[col]):
            weighted_val = (neighbors[col] * neighbors['w']).sum()
            resultats[col] = weighted_val / sum_w
            
    return neighbors, resultats

# --- INTERFACE ---
with st.sidebar:
    st.header("🎛️ Données & Filtres")
    df_tot, err = charger_donnees(DOSSIER_DONNEES)
    if err or df_tot.empty: st.error("Pas de données"); st.stop()

    # 1. Scénario
    scenarios = sorted(df_tot['Scenario'].unique())
    rcp = st.radio("1. Scénario :", scenarios)
    df_rcp = df_tot[df_tot['Scenario'] == rcp]
    
    # 2. Horizon
    horizons = sorted(df_rcp['Horizon_Filter'].unique())
    if not horizons: st.stop()
    horizon = st.radio("2. Période :", horizons)
    
    st.divider()
    
    # 3. VARIABLE À AFFICHER (Dynamique)
    # On liste toutes les colonnes numériques disponibles
    cols_exclues = ['Latitude', 'Longitude', 'Point', 'Horizon_Filter', 'Scenario', 'Source']
    variables_dispos = [c for c in df_rcp.columns if c not in cols_exclues and pd.api.types.is_numeric_dtype(df_rcp[c])]
    
    if not variables_dispos:
        st.error("Aucune variable numérique trouvée dans les fichiers.")
        st.stop()
        
    choix_variable = st.selectbox("3. Variable à visualiser sur la carte :", variables_dispos)
    
    # 4. Style Carte
    st.divider()
    st.header("🗺️ Style")
    styles_dispo = {
        "Clair": "mapbox://styles/mapbox/light-v9",
        "Sombre": "mapbox://styles/mapbox/dark-v9",
        "Satellite": "mapbox://styles/mapbox/satellite-v9",
        "Outdoors": "mapbox://styles/mapbox/outdoors-v11"
    }
    choix_style = st.selectbox("Fond de carte :", list(styles_dispo.keys()))
    style_url = styles_dispo[choix_style]

    # Légende Dynamique
    st.divider()
    vmin = df_tot[choix_variable].min()
    vmax = df_tot[choix_variable].max()
    
    cmap = plt.get_cmap("coolwarm")
    grad = np.linspace(0, 1, 256)
    grad = np.vstack((grad, grad))
    fig, ax = plt.subplots(figsize=(4, 0.4))
    ax.imshow(grad, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    st.write(f"**{choix_variable}**")
    st.pyplot(fig)
    st.write(f"Min: {vmin:.2f} | Max: {vmax:.2f}")

# --- PRÉPARATION ---
df_map = df_rcp[df_rcp['Horizon_Filter'] == horizon].copy()
if df_map.empty: st.warning("Sélection vide"); st.stop()

# Calcul des couleurs basé sur la VARIABLE CHOISIE
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
colors = cmap(norm(df_map[choix_variable].values))
df_map['r'] = (colors[:, 0] * 255).astype(int)
df_map['g'] = (colors[:, 1] * 255).astype(int)
df_map['b'] = (colors[:, 2] * 255).astype(int)

# --- RECHERCHE ---
col_s, col_i = st.columns([3, 1])
with col_s:
    adr = st.text_input("🔍 Rechercher une adresse", placeholder="Ex: Lyon, France")

u_lat, u_lon = None, None
if adr:
    u_lat, u_lon = geocode_address(adr)
    if u_lat:
        st.success(f"📍 Localisé : {u_lat:.4f}, {u_lon:.4f}")
    else:
        st.error("Adresse introuvable.")

# --- CARTE ---
layers = []

# Couche Données (Variable choisie)
data_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position=['Longitude', 'Latitude'],
    get_color=['r', 'g', 'b', 160],
    get_radius=3000,
    pickable=True,
    auto_highlight=True
)
layers.append(data_layer)

if u_lat:
    user_data = pd.DataFrame({'lat': [u_lat], 'lon': [u_lon]})
    user_layer = pdk.Layer(
        "ScatterplotLayer",
        data=user_data,
        get_position='[lon, lat]',
        get_color='[0, 255, 0]', 
        get_radius=1000,
        stroked=True,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=3,
        pickable=False
    )
    layers.append(user_layer)
    view_state = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=10, pitch=0)
else:
    view_state = pdk.ViewState(latitude=df_map['Latitude'].mean(), longitude=df_map['Longitude'].mean(), zoom=5.5, pitch=0)

# Tooltip dynamique : Affiche la variable choisie
tooltip_html = f"<b>Station:</b> {{Point}}<br><b>{choix_variable}:</b> {{{choix_variable}}}"

st.pydeck_chart(pdk.Deck(
    map_style=style_url,
    initial_view_state=view_state,
    layers=layers,
    tooltip={"html": tooltip_html}
))

# --- TABLEAUX MULTI-VARIABLES ---
if u_lat:
    st.divider()
    st.subheader("📊 Tableau de Bord Local")
    
    # Calcul pour TOUTES les variables
    voisins, resultats_interpoles = interpoler_toutes_variables(u_lat, u_lon, df_map)
    p_proche = voisins.iloc[0]
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.info(f"Station la plus proche : **{p_proche.get('Point', 'Inconnu')}** ({p_proche['dist_km']:.1f} km)")
        
        # On prépare un petit tableau propre pour l'affichage
        # On transpose pour avoir les variables en lignes (plus lisible)
        df_proche = pd.DataFrame(p_proche).reset_index()
        df_proche.columns = ['Indicateur', 'Valeur']
        # On filtre les métadonnées inutiles à l'affichage
        meta_exclusion = ['Latitude', 'Longitude', 'Point', 'Horizon_Filter', 'Scenario', 'd_approx', 'dist_km', 'w', 'r', 'g', 'b', 'Source']
        df_proche = df_proche[~df_proche['Indicateur'].isin(meta_exclusion)]
        
        st.dataframe(df_proche, use_container_width=True, hide_index=True)

    with c2:
        st.success("🎯 Valeurs Estimées à votre adresse (Interpolation)")
        
        # Création du dataframe d'estimation
        df_est = pd.DataFrame(list(resultats_interpoles.items()), columns=['Indicateur', 'Valeur Estimée'])
        
        # Mise en forme (arrondi)
        # On détecte si c'est des float pour arrondir
        df_est['Valeur Estimée'] = df_est['Valeur Estimée'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        
        st.dataframe(df_est, use_container_width=True, hide_index=True)
    
    st.caption(f"Données calculées sur la base du scénario {rcp} / {horizon}.")
