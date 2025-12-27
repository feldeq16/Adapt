import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import griddata
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Climat Expert")
st.title("🌡️ Carte Climatique : Modélisation Interactive")

DOSSIER_DONNEES = 'Données'

# --- OUTILS ---
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def trouver_header_et_lire(chemin):
    """Lecture robuste."""
    encodages = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodages:
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
                df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
                map_rename = {}
                for col in df.columns:
                    c_low = remove_accents(col.lower())
                    if 'point' in c_low or 'station' in c_low: map_rename[col] = 'Point'
                if map_rename: df = df.rename(columns=map_rename)
                return df, None
        except: continue
    return None, "Erreur lecture"

# --- CHARGEMENT ---
@st.cache_data(ttl=3600, show_spinner=False)
def charger_donnees(dossier):
    all_data = pd.DataFrame()
    if not os.path.exists(dossier): return pd.DataFrame(), "Dossier introuvable"
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    barre = st.progress(0, text="Lecture des fichiers...")
    
    for i, fichier in enumerate(fichiers):
        barre.progress((i)/len(fichiers))
        path = os.path.join(dossier, fichier)
        df, err = trouver_header_et_lire(path)
        if err: continue
        
        col_p = next((c for c in df.columns if 'eriode' in remove_accents(c.lower()) or 'horizon' in remove_accents(c.lower())), None)
        df['Horizon_Filter'] = df[col_p].astype(str).str.strip() if col_p else "Non défini"

        n = fichier.lower()
        if "rcp2.6" in n or "rcp26" in n: df['Scenario'] = "RCP 2.6"
        elif "rcp4.5" in n or "rcp45" in n: df['Scenario'] = "RCP 4.5"
        elif "rcp8.5" in n or "rcp85" in n: df['Scenario'] = "RCP 8.5"
        else: df['Scenario'] = "Autre"

        cols = {'Latitude', 'Longitude', 'ATXHWD'}
        if cols.issubset(df.columns):
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            if df['ATXHWD'].dtype == object:
                df['ATXHWD'] = pd.to_numeric(df['ATXHWD'].str.replace(',', '.'), errors='coerce')
            else:
                df['ATXHWD'] = pd.to_numeric(df['ATXHWD'], errors='coerce')
            df = df.dropna(subset=['Latitude', 'Longitude', 'ATXHWD'])
            if 'Point' not in df.columns: df['Point'] = df.iloc[:, 0].astype(str)
            all_data = pd.concat([all_data, df], ignore_index=True)

    barre.empty()
    return all_data, None

# --- GENERATION GRILLE (FOND DE CARTE) ---
@st.cache_data(show_spinner=False)
def generer_grille_couleur(df, vmin, vmax):
    """Crée une grille de points interpolés pour servir de fond de carte"""
    # 1. Création de la grille (60x60 points = 3600 tuiles)
    # Augmentez 60j si vous voulez plus fin (mais plus lent)
    grid_x, grid_y = np.mgrid[
        df['Longitude'].min():df['Longitude'].max():60j, 
        df['Latitude'].min():df['Latitude'].max():60j
    ]
    
    # 2. Interpolation
    points = df[['Longitude', 'Latitude']].values
    values = df['ATXHWD'].values
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # 3. Transformation en DataFrame pour PyDeck
    # On aplatit les tableaux
    flat_lon = grid_x.flatten()
    flat_lat = grid_y.flatten()
    flat_val = grid_z.flatten()
    
    df_grid = pd.DataFrame({'lon': flat_lon, 'lat': flat_lat, 'val': flat_val})
    df_grid = df_grid.dropna() # On enlève les zones hors de France (NaN)
    
    # 4. Calcul des couleurs RGB pour chaque tuile
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("coolwarm")
    
    # Vectorisation du calcul couleur (très rapide)
    colors = cmap(norm(df_grid['val'].values))
    df_grid['r'] = (colors[:, 0] * 255).astype(int)
    df_grid['g'] = (colors[:, 1] * 255).astype(int)
    df_grid['b'] = (colors[:, 2] * 255).astype(int)
    
    return df_grid

# --- GÉOCODAGE & ANALYSE ---
@st.cache_data
def geocode_address(address):
    try:
        geolocator = Nominatim(user_agent="app_clim_v5_final")
        loc = geolocator.geocode(address)
        return (loc.latitude, loc.longitude) if loc else (None, None)
    except: return None, None

def interpoler_local(lat, lon, df, n=5):
    df = df.copy()
    df['d_approx'] = (df['Latitude']-lat)**2 + (df['Longitude']-lon)**2
    neighbors = df.nsmallest(n, 'd_approx').copy()
    neighbors['dist_km'] = neighbors.apply(lambda r: geodesic((lat,lon), (r['Latitude'], r['Longitude'])).km, axis=1)
    neighbors['w'] = 1 / (neighbors['dist_km'] + 0.001)**2
    est = (neighbors['ATXHWD']*neighbors['w']).sum() / neighbors['w'].sum()
    return neighbors, est

# --- INTERFACE ---
with st.sidebar:
    st.header("🎛️ Contrôles")
    df_tot, err = charger_donnees(DOSSIER_DONNEES)
    if err or df_tot.empty: st.error("Pas de données"); st.stop()
    
    # Filtres
    scenarios = sorted(df_tot['Scenario'].unique())
    rcp = st.radio("Scénario :", scenarios)
    
    df_rcp = df_tot[df_tot['Scenario'] == rcp]
    horizons = sorted(df_rcp['Horizon_Filter'].unique())
    if not horizons: st.stop()
    horizon = st.radio("Période :", horizons)
    
    st.divider()
    
    # Légende
    vmin, vmax = df_tot['ATXHWD'].min(), df_tot['ATXHWD'].max()
    cmap = plt.get_cmap("coolwarm")
    grad = np.linspace(0, 1, 256)
    grad = np.vstack((grad, grad))
    fig, ax = plt.subplots(figsize=(4, 0.4))
    ax.imshow(grad, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    st.write(f"Légende ({vmin:.1f} à {vmax:.1f})")
    st.pyplot(fig)

# --- PRÉPARATION CARTE ---
df_map = df_rcp[df_rcp['Horizon_Filter'] == horizon].copy()
if df_map.empty: st.warning("Sélection vide"); st.stop()

# Barre de recherche
col_s, col_i = st.columns([3, 1])
with col_s:
    adr = st.text_input("🔍 Rechercher une adresse", "", placeholder="Ex: Place du Capitole, Toulouse")

u_lat, u_lon = None, None
if adr:
    u_lat, u_lon = geocode_address(adr)
    if u_lat: 
        st.success("Adresse localisée !")
    else: 
        st.error("Adresse introuvable via OpenStreetMap.")

# --- GÉNÉRATION DU FOND COLORÉ ---
# On génère les tuiles. Plus de Base64, c'est du vectoriel pur.
with st.spinner("Modélisation du fond de carte..."):
    df_grid = generer_grille_couleur(df_map, vmin, vmax)

# --- COUCHES PYDECK ---
layers = []

# 1. Fond de carte interpolé (Carrés)
grid_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_grid,
    get_position=['lon', 'lat'],
    get_color=['r', 'g', 'b', 200], # 200 = Transparence partielle
    get_radius=8000, # Rayon des tuiles (à ajuster selon la densité)
    pickable=False,  # Fond non cliquable
    stroked=False,
    filled=True
)
layers.append(grid_layer)

# 2. Pin Utilisateur (Prioritaire)
if u_lat:
    user_data = pd.DataFrame({'lat': [u_lat], 'lon': [u_lon]})
    pin_layer = pdk.Layer(
        "ScatterplotLayer",
        data=user_data,
        get_position='[lon, lat]',
        get_color='[0, 255, 0]', # Vert Fluo
        get_radius=6000,
        stroked=True,
        get_line_color=[0,0,0],
        line_width_min_pixels=3,
        pickable=False
    )
    layers.append(pin_layer)
    # ZOOM FORCÉ SUR L'ADRESSE
    view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=8.5)
else:
    # VUE GLOBALE
    view = pdk.ViewState(latitude=df_map['Latitude'].mean(), longitude=df_map['Longitude'].mean(), zoom=5.5)

# 3. Stations réelles (Points noirs transparents cliquables)
real_points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position=['Longitude', 'Latitude'],
    get_radius=1000,
    get_color=[0, 0, 0, 100], # Noir semi-transparent
    pickable=True, # C'est le seul calque cliquable
    auto_highlight=True
)
layers.append(real_points_layer)

# RENDU
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=view,
    layers=layers,
    tooltip={"html": "<b>Valeur Réelle:</b> {ATXHWD}<br><b>Station:</b> {Point}"}
))

# --- TABLEAUX ---
if u_lat:
    st.divider()
    voisins, val_est = interpoler_local(u_lat, u_lon, df_map)
    c1, c2 = st.columns(2)
    p_proche = voisins.iloc[0]
    
    with c1:
        st.info("📍 Station la plus proche")
        st.metric(f"Nom : {p_proche.get('Point', 'Inconnu')}", f"{p_proche['ATXHWD']:.2f}", f"à {p_proche['dist_km']:.1f} km")
    with c2:
        st.success("🎯 Estimation à votre adresse")
        st.metric("Indicateur ATXHWD (Interpolé)", f"{val_est:.2f}")
    
    st.caption("Données utilisées pour le calcul :")
    disp = voisins[['Point', 'ATXHWD', 'dist_km']].copy()
    disp['dist_km'] = disp['dist_km'].map('{:.2f} km'.format)
    st.dataframe(disp, use_container_width=True, hide_index=True)
