import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import unicodedata
import matplotlib.pyplot as plt # Pour la légende seulement
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Climat Expert")
st.title("🌡️ Carte Climatique Interactive")

DOSSIER_DONNEES = 'Données'

# --- FONCTIONS UTILITAIRES ---
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def trouver_header_et_lire(chemin):
    """Lecture robuste avec détection d'encodage."""
    encodages = ['utf-8', 'latin-1', 'cp1252']
    logs = []
    
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
                # Nettoyage colonnes
                df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
                # Renommage Point/Station
                map_rename = {}
                for col in df.columns:
                    c_low = remove_accents(col.lower())
                    if 'point' in c_low or 'station' in c_low: map_rename[col] = 'Point'
                if map_rename: df = df.rename(columns=map_rename)
                
                return df, None, f"Lecture OK ({enc}, sep='{sep}', ligne {header_row})"
        except Exception as e:
            logs.append(f"Echec {enc}: {str(e)}")
            continue
            
    return None, f"Erreur fatale lecture: {logs}", "Echec"

# --- CHARGEMENT ---
@st.cache_data(ttl=3600, show_spinner=False)
def charger_donnees(dossier):
    all_data = pd.DataFrame()
    debug_log = []
    
    if not os.path.exists(dossier): return pd.DataFrame(), ["Dossier introuvable"], None
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    barre = st.progress(0, text="Chargement...")
    
    for i, fichier in enumerate(fichiers):
        barre.progress((i)/len(fichiers))
        path = os.path.join(dossier, fichier)
        df, err, msg = trouver_header_et_lire(path)
        
        debug_log.append(f"📄 {fichier} : {msg}")
        if err: 
            debug_log.append(f"❌ Erreur: {err}")
            continue
        
        # Période
        col_p = next((c for c in df.columns if 'eriode' in remove_accents(c.lower()) or 'horizon' in remove_accents(c.lower())), None)
        if col_p:
            df['Horizon_Filter'] = df[col_p].astype(str).str.strip()
            debug_log.append(f"  -> Période trouvée: '{col_p}'")
        else:
            df['Horizon_Filter'] = "Non défini"
            debug_log.append(f"  -> ⚠️ Période NON trouvée. Colonnes: {list(df.columns)}")

        # Scénario
        n = fichier.lower()
        if "rcp2.6" in n or "rcp26" in n: df['Scenario'] = "RCP 2.6"
        elif "rcp4.5" in n or "rcp45" in n: df['Scenario'] = "RCP 4.5"
        elif "rcp8.5" in n or "rcp85" in n: df['Scenario'] = "RCP 8.5"
        else: df['Scenario'] = "Autre"

        # Conversion
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
        else:
            debug_log.append(f"  -> ❌ Colonnes manquantes. Besoin de Lat/Lon/ATXHWD. Vu: {list(df.columns)}")

    barre.empty()
    return all_data, debug_log, None

# --- GÉOCODAGE & INTERPOLATION (Tableau seulement) ---
@st.cache_data
def geocode_address(address):
    try:
        geolocator = Nominatim(user_agent="app_clim_final")
        loc = geolocator.geocode(address)
        return (loc.latitude, loc.longitude) if loc else (None, None)
    except: return None, None

def interpoler_local(lat, lon, df, n=5):
    """Calcul IDW (Inverse Distance Weighting) pour le tableau"""
    df = df.copy()
    # Pré-filtre grossier pour la vitesse
    df['d_approx'] = (df['Latitude']-lat)**2 + (df['Longitude']-lon)**2
    neighbors = df.nsmallest(n, 'd_approx').copy()
    
    # Calcul précis
    neighbors['dist_km'] = neighbors.apply(lambda r: geodesic((lat,lon), (r['Latitude'], r['Longitude'])).km, axis=1)
    
    # Pondération
    neighbors['w'] = 1 / (neighbors['dist_km'] + 0.001)**2
    est = (neighbors['ATXHWD']*neighbors['w']).sum() / neighbors['w'].sum()
    return neighbors, est

# --- INTERFACE ---
with st.sidebar:
    st.header("🎛️ Contrôles")
    
    if st.button("♻️ Recharger tout"):
        st.cache_data.clear()
        st.rerun()

    df_tot, logs, err = charger_donnees(DOSSIER_DONNEES)
    
    # BOITE DIAGNOSTIC
    with st.expander("🛠️ Diagnostic Technique"):
        for l in logs:
            if "❌" in l: st.error(l)
            elif "⚠️" in l: st.warning(l)
            else: st.caption(l)

    if err or df_tot.empty: st.error("Données vides"); st.stop()
    
    # Filtres
    scenarios = sorted(df_tot['Scenario'].unique())
    rcp = st.radio("Scénario :", scenarios)
    
    df_rcp = df_tot[df_tot['Scenario'] == rcp]
    horizons = sorted(df_rcp['Horizon_Filter'].unique())
    if not horizons: st.stop()
    horizon = st.radio("Période :", horizons)
    
    st.divider()
    
    # Légende visuelle
    vmin, vmax = df_tot['ATXHWD'].min(), df_tot['ATXHWD'].max()
    cmap = plt.get_cmap("coolwarm") # YlOrRd est aussi très bien pour la chaleur
    grad = np.linspace(0, 1, 256)
    grad = np.vstack((grad, grad))
    fig, ax = plt.subplots(figsize=(4, 0.4))
    ax.imshow(grad, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    st.write(f"Échelle ({vmin:.1f}° à {vmax:.1f}°)")
    st.pyplot(fig)

# --- CARTE PRINCIPALE ---
df_map = df_rcp[df_rcp['Horizon_Filter'] == horizon].copy()
if df_map.empty: st.warning("Sélection vide"); st.stop()

col_s, col_i = st.columns([3, 1])
with col_s:
    adr = st.text_input("🔍 Rechercher une adresse", "")

u_lat, u_lon = None, None
if adr:
    u_lat, u_lon = geocode_address(adr)
    if u_lat: st.success("Adresse localisée !")
    else: st.error("Introuvable")

# --- CONSTRUCTION DES COUCHES PYDECK ---
layers = []

# 1. HEATMAP LAYER (Le fameux gradient, géré par le GPU)
# C'est la solution la plus robuste.
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=df_map,
    opacity=0.8,
    get_position=['Longitude', 'Latitude'],
    get_weight='ATXHWD', # Le poids détermine la "chaleur"
    radiusPixels=50,     # Rayon de diffusion pour lisser
    intensity=1,
    threshold=0.05,      # Ignore les valeurs trop faibles
    # On définit les couleurs du gradient manuellement pour être sûr
    colorRange=[
        [69, 117, 180], [145, 191, 219], [224, 243, 248], 
        [254, 224, 144], [252, 141, 89], [215, 48, 39]
    ]
)
layers.append(heatmap_layer)

# 2. PIN UTILISATEUR
if u_lat:
    user_data = pd.DataFrame({'lat': [u_lat], 'lon': [u_lon]})
    pin_layer = pdk.Layer(
        "ScatterplotLayer",
        data=user_data,
        get_position='[lon, lat]',
        get_color='[0, 255, 0]', # Vert Fluo
        get_radius=5000,
        stroked=True,
        get_line_color=[0,0,0],
        line_width_min_pixels=3,
        pickable=False
    )
    layers.append(pin_layer)
    view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=8)
else:
    view = pdk.ViewState(latitude=df_map['Latitude'].mean(), longitude=df_map['Longitude'].mean(), zoom=5.5)

# 3. POINTS INVISIBLES POUR L'INFOBULLE
# Le Heatmap ne permet pas de cliquer sur un point précis, 
# donc on ajoute des points transparents par dessus.
point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position='[Longitude, Latitude]',
    get_radius=1000,
    get_color=[0,0,0,0], # Transparent
    pickable=True,
    auto_highlight=True
)
layers.append(point_layer)

# RENDU DE LA CARTE
st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=view,
    layers=layers,
    # Le tooltip ne marchera que sur le point_layer car c'est le seul "pickable=True"
    tooltip={"html": "<b>ATXHWD:</b> {ATXHWD}<br><b>Station:</b> {Point}"}
))

# --- TABLEAUX ANALYSE ---
if u_lat:
    st.divider()
    voisins, val_est = interpoler_local(u_lat, u_lon, df_map)
    c1, c2 = st.columns(2)
    p_proche = voisins.iloc[0]
    
    with c1:
        st.info("📍 Station de mesure la plus proche")
        st.metric(f"Nom : {p_proche.get('Point', 'Inconnu')}", f"{p_proche['ATXHWD']:.2f}", f"à {p_proche['dist_km']:.1f} km")
    with c2:
        st.success("🎯 Valeur Extrapolée (Votre Adresse)")
        st.metric("Indicateur ATXHWD estimé", f"{val_est:.2f}")
    
    st.write("Détail du calcul (Interpolation des 5 voisins) :")
    disp = voisins[['Point', 'ATXHWD', 'dist_km']].copy()
    disp['dist_km'] = disp['dist_km'].map('{:.2f} km'.format)
    st.dataframe(disp, use_container_width=True, hide_index=True)
