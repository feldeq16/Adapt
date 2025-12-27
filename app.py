import streamlit as st
import pandas as pd
import pydeck as pdk
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mon Portail Cartographique")
st.title("🗺️ Visualisation Haute Performance (PyDeck)")

DOSSIER_DONNEES = 'Données'

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data(ttl=3600)
def charger_donnees_rapides(dossier, max_points):
    """Charge les données de manière optimisée pour PyDeck"""
    all_data = []
    
    if not os.path.exists(dossier):
        return pd.DataFrame()
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    
    # Couleurs pour distinguer les fichiers (RGB)
    couleurs = [
        [255, 0, 0],   # Rouge
        [0, 255, 0],   # Vert
        [0, 0, 255],   # Bleu
        [255, 165, 0], # Orange
        [128, 0, 128]  # Violet
    ]
    
    for i, fichier in enumerate(fichiers):
        chemin = os.path.join(dossier, fichier)
        try:
            # On lit le fichier
            df = pd.read_csv(
                chemin, 
                sep=';', 
                comment='#', 
                encoding='latin-1',
                engine='python',
                nrows=max_points # <--- Sécurité : on charge max X points
            )
            
            # Nettoyage colonnes
            df.columns = [c.strip() for c in df.columns]
            
            # Vérification Lat/Lon
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # Conversion numérique
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                df = df.dropna(subset=['Latitude', 'Longitude'])
                
                # On attribue une couleur à ce fichier
                color = couleurs[i % len(couleurs)]
                df['color_r'] = color[0]
                df['color_g'] = color[1]
                df['color_b'] = color[2]
                df['source'] = fichier
                
                all_data.append(df)
                
        except Exception as e:
            print(f"Erreur {fichier}: {e}")
            
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

# --- INTERFACE ---
with st.sidebar:
    st.header("🚀 Contrôle Performance")
    
    # Curseur pour gérer la charge
    # On commence à 2000 pour être sûr que ça s'affiche, vous pourrez monter après
    nb_points = st.slider("Points par fichier (Max)", 100, 8000, 2000)
    
    st.write("Chargement des données...")
    df_map = charger_donnees_rapides(DOSSIER_DONNEES, nb_points)
    
    if not df_map.empty:
        st.success(f"✅ {len(df_map)} points chargés au total.")
        st.info("Rouge/Vert/Bleu selon le fichier.")
    else:
        st.error("Aucune donnée.")

# --- CARTE PYDECK (GPU) ---
# C'est ici que la magie opère : PyDeck gère des milliers de points sans ramer
if not df_map.empty:
    
    # Configuration de la vue initiale (centrée sur la moyenne des points)
    view_state = pdk.ViewState(
        latitude=df_map['Latitude'].mean(),
        longitude=df_map['Longitude'].mean(),
        zoom=5,
        pitch=0,
    )

    # Création de la couche de points (Scatterplot)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[Longitude, Latitude]',
        get_color='[color_r, color_g, color_b, 160]', # Couleur + Transparence
        get_radius=5000, # Rayon en mètres
        pickable=True,   # Permet de cliquer (info-bulle)
    )

    # Tooltip (Info-bulle au survol)
    tooltip = {
        "html": "<b>Source:</b> {source}<br><b>Lat:</b> {Latitude}<br><b>Lon:</b> {Longitude}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }

    # Rendu de la carte
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9', # Style léger
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    ))

else:
    st.write("En attente de données...")
