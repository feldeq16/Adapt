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
st.title("🌡️ Analyse Climatique & Recherche d'Adresse")

DOSSIER_DONNEES = 'Données'

# --- OUTILS & FONCTIONS ---
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def trouver_header_et_lire(chemin):
    """Lecture robuste avec détection d'encodage et de header."""
    encodages_a_tester = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodages_a_tester:
        try:
            with open(chemin, 'r', encoding=enc) as f:
                lignes = [f.readline() for _ in range(50)]
            
            header_row = None
            sep = ';'
            for i, ligne in enumerate(lignes):
                ligne_clean = remove_accents(ligne).lower()
                if 'latitude' in ligne_clean and 'longitude' in ligne_clean:
                    header_row = i
                    if ';' in ligne: sep = ';'
                    elif ',' in ligne: sep = ','
                    elif '\t' in ligne: sep = '\t'
                    break
            
            if header_row is not None:
                # On lit le fichier
                df = pd.read_csv(chemin, sep=sep, header=header_row, encoding=enc, engine='python')
                
                # --- CORRECTION DE L'ERREUR KEYERROR ---
                # On nettoie les noms de colonnes (enlève BOM \ufeff et espaces)
                df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
                
                # On renomme la colonne identifiant en "Point" de manière standard
                col_rename_map = {}
                for col in df.columns:
                    c_lower = remove_accents(col.lower())
                    if 'point' in c_lower or 'station' in c_lower:
                        col_rename_map[col] = 'Point'
                
                if col_rename_map:
                    df = df.rename(columns=col_rename_map)
                # ----------------------------------------
                
                return df, None
        except:
            continue
    return None, "Lecture impossible"

@st.cache_data(ttl=3600, show_spinner=False)
def charger_donnees(dossier):
    all_data = pd.DataFrame()
    if not os.path.exists(dossier): return pd.DataFrame(), f"Dossier '{dossier}' introuvable."
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    barre = st.progress(0, text="Chargement des données...")
    
    for i, fichier in enumerate(fichiers):
        chemin = os.path.join(dossier, fichier)
        barre.progress((i)/len(fichiers))
        
        df, erreur = trouver_header_et_lire(chemin)
        if erreur: continue
            
        # Recherche colonne Période
        col_periode = next((c for c in df.columns if 'eriode' in remove_accents(c.lower()) or 'horizon' in remove_accents(c.lower())), None)
        df['Horizon_Filter'] = df[col_periode].astype(str).str.strip() if col_periode else "Non défini"

        # Scénario
        nom = fichier.lower()
        if "rcp2.6" in nom or "rcp26" in nom: df['Scenario'] = "RCP 2.6"
        elif "rcp4.5" in nom or "rcp45" in nom: df['Scenario'] = "RCP 4.5"
        elif "rcp8.5" in nom or "rcp85" in nom: df['Scenario'] = "RCP 8.5"
        else: df['Scenario'] = "Autre"

        # Conversion
        # On vérifie si on a bien 'Point' maintenant (grâce au renommage plus haut)
        colonnes_requises = {'Latitude', 'Longitude', 'ATXHWD'}
        if colonnes_requises.issubset(df.columns):
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            if df['ATXHWD'].dtype == object:
                 df['ATXHWD'] = pd.to_numeric(df['ATXHWD'].str.replace(',', '.'), errors='coerce')
            else:
                 df['ATXHWD'] = pd.to_numeric(df['ATXHWD'], errors='coerce')
            
            df = df.dropna(subset=['Latitude', 'Longitude', 'ATXHWD'])
            df['Source'] = fichier
            
            # Si la colonne Point n'existe pas, on prend la 1ère colonne par défaut pour éviter le plantage
            if 'Point' not in df.columns:
                df['Point'] = df.iloc[:, 0].astype(str)

            all_data = pd.concat([all_data, df], ignore_index=True)

    barre.empty()
    return all_data, None

# --- FONCTION GÉOCODAGE & ANALYSE ---
@st.cache_data
def geocode_address(address):
    """Convertit une adresse en lat/lon via OpenStreetMap"""
    try:
        geolocator = Nominatim(user_agent="my_climate_app_v2")
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        return None, None
    except:
        return None, None

def interpoler_valeur(lat_cible, lon_cible, df, n_neighbors=5):
    """Extrapolation IDW"""
    df = df.copy()
    # Approximation rapide pour trier
    df['dist_approx'] = (df['Latitude'] - lat_cible)**2 + (df['Longitude'] - lon_cible)**2
    
    neighbors = df.nsmallest(n_neighbors, 'dist_approx').copy()
    
    # Calcul précis distance géodésique
    neighbors['distance_km'] = neighbors.apply(
        lambda row: geodesic((lat_cible, lon_cible), (row['Latitude'], row['Longitude'])).km, axis=1
    )
    
    neighbors['weight'] = 1 / (neighbors['distance_km'] + 0.001)**2
    weighted_sum = (neighbors['ATXHWD'] * neighbors['weight']).sum()
    sum_of_weights = neighbors['weight'].sum()
    
    estimated_value = weighted_sum / sum_of_weights
    
    return neighbors, estimated_value

# --- 2. INTERFACE & CHARGEMENT ---
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    if st.button("Recharger les données"):
        st.cache_data.clear()
        st.rerun()

    df_total, erreur = charger_donnees(DOSSIER_DONNEES)
    
    if erreur or df_total.empty:
        st.error(erreur or "Aucune donnée.")
        st.stop()

    # Filtres
    scenarios = sorted(df_total['Scenario'].unique())
    choix_rcp = st.radio("Scénario :", scenarios)

    df_rcp = df_total[df_total['Scenario'] == choix_rcp]
    horizons = sorted(df_rcp['Horizon_Filter'].unique())
    
    if not horizons:
        st.warning("Périodes introuvables.")
        st.stop()
    choix_horizon = st.radio("Période :", horizons)
    
    st.divider()
    st.write("Légende (ATXHWD)")
    
    val_min = df_total['ATXHWD'].min()
    val_max = df_total['ATXHWD'].max()
    cmap = plt.get_cmap("coolwarm")
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(4, 0.4))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    st.pyplot(fig)
    st.write(f"Min: {val_min:.1f} | Max: {val_max:.1f}")

# --- 3. LOGIQUE PRINCIPALE ---

df_map = df_rcp[df_rcp['Horizon_Filter'] == choix_horizon].copy()

if df_map.empty:
    st.warning("Pas de données pour cette sélection.")
    st.stop()

# Barre de recherche
col_search, col_info = st.columns([3, 1])
with col_search:
    adresse_recherche = st.text_input("🔍 Rechercher une adresse en France (ex: 15 rue de Rivoli, Paris)", "")

user_lat, user_lon = None, None
search_layer = None

if adresse_recherche:
    user_lat, user_lon = geocode_address(adresse_recherche)
    
    if user_lat:
        st.success(f"Adresse trouvée : Latitude {user_lat:.4f}, Longitude {user_lon:.4f}")
        
        search_data = pd.DataFrame({'lat': [user_lat], 'lon': [user_lon], 'name': ["Votre Adresse"]})
        search_layer = pdk.Layer(
            "ScatterplotLayer",
            data=search_data,
            get_position='[lon, lat]',
            get_color='[0, 255, 0, 255]',
            get_radius=8000,
            pickable=True,
            stroked=True,
            get_line_color=[0, 0, 0],
            line_width_min_pixels=2
        )
    else:
        st.error("Adresse introuvable.")

# Couleurs
norm = mcolors.Normalize(vmin=val_min, vmax=val_max)
colors = cmap(norm(df_map['ATXHWD'].values))
df_map['r'] = (colors[:, 0] * 255).astype(int)
df_map['g'] = (colors[:, 1] * 255).astype(int)
df_map['b'] = (colors[:, 2] * 255).astype(int)

# --- 4. AFFICHAGE CARTE ---
layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[Longitude, Latitude]',
        get_color='[r, g, b, 180]',
        get_radius=4000,
        pickable=True,
        auto_highlight=True
    )
]

if search_layer:
    layers.append(search_layer)
    view_state = pdk.ViewState(latitude=user_lat, longitude=user_lon, zoom=8, pitch=0)
else:
    view_state = pdk.ViewState(latitude=df_map['Latitude'].mean(), longitude=df_map['Longitude'].mean(), zoom=5.5, pitch=0)

tooltip = {"html": "<b>ATXHWD:</b> {ATXHWD}<br><b>Point:</b> {Point}", "style": {"color": "white"}}

st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers, tooltip=tooltip))

# --- 5. TABLEAUX D'ANALYSE ---
if user_lat and not df_map.empty:
    st.divider()
    st.subheader("📊 Analyse Locale")
    
    voisins, estimation = interpoler_valeur(user_lat, user_lon, df_map, n_neighbors=5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("📍 Point le plus proche")
        # --- Sécurité ici pour éviter le KeyError ---
        plus_proche = voisins.iloc[0]
        nom_station = plus_proche.get('Point', 'Inconnu')
        
        st.metric(
            label=f"Station : {nom_station}", 
            value=f"{plus_proche['ATXHWD']:.2f}",
            delta=f"à {plus_proche['distance_km']:.1f} km"
        )
        st.write(f"*Donnée réelle issue du modèle {choix_rcp} / {choix_horizon}*")

    with col2:
        st.success("🎯 Valeur Extrapolée à votre adresse")
        st.metric(
            label="Estimation ATXHWD",
            value=f"{estimation:.2f}"
        )
        st.write("*Calculée par interpolation (pondération par distance) des 5 points les plus proches.*")
    
    st.write("---")
    st.write("**Détail des 5 points utilisés :**")
    
    cols_to_show = ['Point', 'ATXHWD', 'distance_km', 'Latitude', 'Longitude']
    # On filtre pour ne montrer que les colonnes qui existent vraiment
    cols_existantes = [c for c in cols_to_show if c in voisins.columns]
    
    display_voisins = voisins[cols_existantes].copy()
    display_voisins['distance_km'] = display_voisins['distance_km'].map('{:.2f} km'.format)
    display_voisins['ATXHWD'] = display_voisins['ATXHWD'].map('{:.2f}'.format)
    
    st.dataframe(display_voisins, use_container_width=True, hide_index=True)
