import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import unicodedata
import base64
import io
from scipy.interpolate import griddata
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Climat Interpolé")
st.title("🌡️ Carte Climatique Interpolée (Gradient)")

DOSSIER_DONNEES = 'Données'

# --- OUTILS DE NETTOYAGE ---
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def trouver_header_et_lire(chemin):
    """Lecture robuste avec détection d'encodage et nettoyage colonnes."""
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
                # Nettoyage BOM et espaces
                df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
                # Renommage Point/Station
                map_rename = {}
                for col in df.columns:
                    c_low = remove_accents(col.lower())
                    if 'point' in c_low or 'station' in c_low: map_rename[col] = 'Point'
                if map_rename: df = df.rename(columns=map_rename)
                return df, None
        except:
            continue
    return None, "Erreur lecture"

# --- CHARGEMENT ---
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
        
        # Période
        col_p = next((c for c in df.columns if 'eriode' in remove_accents(c.lower()) or 'horizon' in remove_accents(c.lower())), None)
        df['Horizon_Filter'] = df[col_p].astype(str).str.strip() if col_p else "Non défini"

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

    barre.empty()
    return all_data, None

# --- FONCTION MAGIQUE : GÉNÉRATION IMAGE INTERPOLÉE ---
@st.cache_data(show_spinner=False)
def generer_image_interpolated(df, val_min, val_max):
    """
    Crée une image PNG transparente interpolée (carte de chaleur)
    à partir des points et renvoie l'image en base64 + les coordonnées.
    """
    # 1. Définition de la grille (Résolution 100x100 points pour la rapidité)
    grid_x, grid_y = np.mgrid[
        df['Longitude'].min():df['Longitude'].max():150j, 
        df['Latitude'].min():df['Latitude'].max():150j
    ]
    
    points = df[['Longitude', 'Latitude']].values
    values = df['ATXHWD'].values
    
    # 2. Interpolation (Linear = rapide et triangle, Cubic = plus courbe mais lent)
    # 'linear' remplit bien les trous entre les points
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    
    # 3. Création de l'image avec Matplotlib
    # On transpose (.T) car griddata sort en (x, y) et imshow attend (row, col)
    grid_z = grid_z.T 
    
    # Normalisation des couleurs
    norm = mcolors.Normalize(vmin=val_min, vmax=val_max)
    cmap = plt.get_cmap("coolwarm")
    
    # Application de la colormap -> Donne un tableau (H, W, 4) RGBA
    image_data = cmap(norm(grid_z))
    
    # Rendre transparents les pixels NaN (là où il n'y a pas de données, ex: mer)
    image_data[np.isnan(grid_z), 3] = 0.0  # Alpha channel à 0
    # Rendre le reste un peu transparent (0.7) pour voir le fond de carte
    image_data[~np.isnan(grid_z), 3] = 0.75 

    # 4. Sauvegarde en mémoire
    image_data = (image_data * 255).astype(np.uint8) # Conversion 0-255
    img =  pd.DataFrame() # Dummy
    
    # Astuce pour passer l'image à PyDeck : Base64
    pil_image = plt.imsave(io.BytesIO(), image_data, format='png')
    
    # On recrée proprement le buffer
    buffer = io.BytesIO()
    plt.imsave(buffer, image_data, format='png')
    buffer.seek(0)
    b64_string = base64.b64encode(buffer.read()).decode()
    
    # Coordonnées pour plaquer l'image sur la carte (Bounding Box)
    bounds = [
        df['Longitude'].min(), # Ouest
        df['Latitude'].min(),  # Sud
        df['Longitude'].max(), # Est
        df['Latitude'].max()   # Nord
    ]
    
    return f"data:image/png;base64,{b64_string}", bounds

# --- GÉOCODAGE ---
@st.cache_data
def geocode_address(address):
    try:
        geolocator = Nominatim(user_agent="app_clim_v3")
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
    st.header("🎛️ Paramètres")
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
    st.write(f"Légende ({vmin:.1f} à {vmax:.1f})")
    st.pyplot(fig)

# --- CARTE PRINCIPALE ---
df_map = df_rcp[df_rcp['Horizon_Filter'] == horizon].copy()
if df_map.empty: st.warning("Vide"); st.stop()

col_s, col_i = st.columns([3, 1])
with col_s:
    adr = st.text_input("🔍 Rechercher une adresse", "")

u_lat, u_lon = None, None
if adr:
    u_lat, u_lon = geocode_address(adr)
    if u_lat: st.success("Adresse trouvée !")
    else: st.error("Introuvable")

# GÉNÉRATION DU FOND DE CARTE (IMAGE)
with st.spinner("Génération du gradient..."):
    # On génère l'image basée uniquement sur les données filtrées
    img_b64, bounds = generer_image_interpolated(df_map, vmin, vmax)

# COUCHES PYDECK
layers = []

# 1. Couche de fond (L'image interpolée)
bitmap_layer = pdk.Layer(
    "BitmapLayer",
    image=img_b64,
    bounds=bounds,
    opacity=0.8 # Transparence globale de l'image
)
layers.append(bitmap_layer)

# 2. Couche Pin Utilisateur (si recherche)
if u_lat:
    user_data = pd.DataFrame({'lat': [u_lat], 'lon': [u_lon]})
    pin_layer = pdk.Layer(
        "ScatterplotLayer",
        data=user_data,
        get_position='[lon, lat]',
        get_color='[0, 255, 0]', # Vert fluo
        get_radius=8000,
        stroked=True,
        get_line_color=[0,0,0],
        line_width_min_pixels=3,
        pickable=True
    )
    layers.append(pin_layer)
    view = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=7.5)
else:
    view = pdk.ViewState(latitude=df_map['Latitude'].mean(), longitude=df_map['Longitude'].mean(), zoom=5.5)

# 3. (Optionnel) Couche de points invisibles pour le Tooltip
# Pour qu'on puisse quand même voir la valeur en survolant (même si on ne voit pas le point)
point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position='[Longitude, Latitude]',
    get_radius=500, # Petit rayon
    get_color=[0,0,0,0], # Totalement transparent
    pickable=True
)
layers.append(point_layer)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=view,
    layers=layers,
    tooltip={"html": "<b>ATXHWD:</b> {ATXHWD}<br><b>Point:</b> {Point}"}
))

# --- ANALYSE ---
if u_lat:
    st.divider()
    voisins, val_est = interpoler_local(u_lat, u_lon, df_map)
    c1, c2 = st.columns(2)
    p_proche = voisins.iloc[0]
    
    with c1:
        st.info("📍 Point réel le plus proche")
        st.metric(f"Station : {p_proche.get('Point', 'Inconnu')}", f"{p_proche['ATXHWD']:.2f}", f"{p_proche['dist_km']:.1f} km")
    with c2:
        st.success("🎯 Valeur estimée (Interpolée)")
        st.metric("Estimation ATXHWD", f"{val_est:.2f}")
    
    st.caption("Points utilisés pour l'interpolation :")
    disp = voisins[['Point', 'ATXHWD', 'dist_km']].copy()
    disp['dist_km'] = disp['dist_km'].map('{:.2f} km'.format)
    st.dataframe(disp, use_container_width=True, hide_index=True)
