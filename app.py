import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import unicodedata
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Analyse Climatique")
st.title("🌡️ Analyse Climatique (Points & Recherche)")

DOSSIER_DONNEES = 'Données'

# --- OUTILS & FONCTIONS ---
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def trouver_header_et_lire(chemin):
    """Lecture robuste avec détection d'encodage."""
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

@st.cache_data(ttl=3600, show_spinner=False)
def charger_donnees(dossier):
    all_data = pd.DataFrame()
    if not os.path.exists(dossier): return pd.DataFrame(), "Dossier introuvable"
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    barre = st.progress(0, text="Chargement...")
    
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

# --- FONCTIONS RECHERCHE & INTERPOLATION ---
@st.cache_data
def geocode_address(address):
    """Géocodage via OSM"""
    try:
        # User-agent unique indispensable pour ne pas être bloqué
        geolocator = Nominatim(user_agent="app_climat_points_v1")
        loc = geolocator.geocode(address)
        return (loc.latitude, loc.longitude) if loc else (None, None)
    except: return None, None

def interpoler_valeur(lat, lon, df, n=5):
    """Calcul des voisins proches"""
    df = df.copy()
    df['d_approx'] = (df['Latitude']-lat)**2 + (df['Longitude']-lon)**2
    neighbors = df.nsmallest(n, 'd_approx').copy()
    neighbors['dist_km'] = neighbors.apply(lambda r: geodesic((lat,lon), (r['Latitude'], r['Longitude'])).km, axis=1)
    neighbors['w'] = 1 / (neighbors['dist_km'] + 0.001)**2
    est = (neighbors['ATXHWD']*neighbors['w']).sum() / neighbors['w'].sum()
    return neighbors, est

# --- INTERFACE ---
with st.sidebar:
    st.header("🎛️ Filtres")
    df_tot, err = charger_donnees(DOSSIER_DONNEES)
    if err or df_tot.empty: st.error("Pas de données"); st.stop()

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
    st.write(f"Min: {vmin:.1f} | Max: {vmax:.1f}")
    st.pyplot(fig)

# --- PRÉPARATION DES DONNÉES ---
df_map = df_rcp[df_rcp['Horizon_Filter'] == horizon].copy()
if df_map.empty: st.warning("Sélection vide"); st.stop()

# Gestion des couleurs (Bleu -> Rouge)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
colors = cmap(norm(df_map['ATXHWD'].values))
df_map['r'] = (colors[:, 0] * 255).astype(int)
df_map['g'] = (colors[:, 1] * 255).astype(int)
df_map['b'] = (colors[:, 2] * 255).astype(int)

# --- RECHERCHE ---
col_s, col_i = st.columns([3, 1])
with col_s:
    adr = st.text_input("🔍 Rechercher une adresse", placeholder="Ex: Bordeaux, France")

u_lat, u_lon = None, None
if adr:
    u_lat, u_lon = geocode_address(adr)
    if u_lat:
        st.success(f"📍 Localisé : {u_lat:.4f}, {u_lon:.4f}")
    else:
        st.error("Adresse introuvable.")

# --- COUCHES PYDECK (L'ordre est important) ---
layers = []

# 1. Couche des Données Climatiques (Points)
data_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position=['Longitude', 'Latitude'],
    get_color=['r', 'g', 'b', 160], # 160 = Transparence
    get_radius=2000, # Rayon de 2km par point
    pickable=True,   # Cliquable
    auto_highlight=True
)
layers.append(data_layer)

# 2. Couche Utilisateur (Si recherche active)
if u_lat:
    user_data = pd.DataFrame({'lat': [u_lat], 'lon': [u_lon]})
    user_layer = pdk.Layer(
        "ScatterplotLayer",
        data=user_data,
        get_position='[lon, lat]',
        get_color='[0, 255, 0]', # VERT FLUO
        get_radius=5000, # Plus gros que les points de données
        stroked=True,
        get_line_color=[0, 0, 0],
        line_width_min_pixels=3,
        pickable=False
    )
    layers.append(user_layer)
    
    # *** LE FIX EST ICI : VIEW STATE DYNAMIQUE ***
    # Si on a une adresse, on zoom fort dessus (Zoom 11)
    view_state = pdk.ViewState(
        latitude=u_lat, 
        longitude=u_lon, 
        zoom=11, 
        pitch=0
    )
else:
    # Sinon, vue globale sur la France (Zoom 5.5)
    view_state = pdk.ViewState(
        latitude=df_map['Latitude'].mean(), 
        longitude=df_map['Longitude'].mean(), 
        zoom=5.5, 
        pitch=0
    )

# --- AFFICHAGE CARTE ---
st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9", # Fond de carte clair et propre
    initial_view_state=view_state,
    layers=layers,
    tooltip={"html": "<b>ATXHWD:</b> {ATXHWD}<br><b>Station:</b> {Point}"}
))

# --- TABLEAUX DE RÉSULTATS ---
if u_lat:
    st.divider()
    voisins, val_est = interpoler_local(u_lat, u_lon, df_map)
    c1, c2 = st.columns(2)
    p_proche = voisins.iloc[0]
    
    with c1:
        st.info("Station la plus proche")
        st.metric(f"{p_proche.get('Point', 'Inconnu')}", f"{p_proche['ATXHWD']:.2f}", f"à {p_proche['dist_km']:.1f} km")
    with c2:
        st.success("Estimation à votre adresse")
        st.metric("ATXHWD Interpolé", f"{val_est:.2f}")
    
    st.write("Stations utilisées pour le calcul :")
    disp = voisins[['Point', 'ATXHWD', 'dist_km']].copy()
    disp['dist_km'] = disp['dist_km'].map('{:.2f} km'.format)
    st.dataframe(disp, use_container_width=True, hide_index=True)
