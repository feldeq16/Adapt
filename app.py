import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import geopandas as gpd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Mon Portail Cartographique",
    layout="wide",  # Utilise toute la largeur de l'écran
    initial_sidebar_state="expanded"
)

# --- TITRE ET INTRODUCTION ---
st.title("🗺️ Mon Portail Cartographique")
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.write("Bienvenue sur votre plateforme de visualisation géographique.")

# --- BARRE LATÉRALE (CONTROLES) ---
with st.sidebar:
    st.header("🛠️ Contrôles")
    
    # Choix du fond de carte
    map_style = st.selectbox(
        "Fond de carte",
        ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark_Matter"]
    )
    
    st.divider()
    
    # Section pour ajouter des données (Fonctionnalité future)
    st.subheader("📂 Ajouter des données")
    uploaded_file = st.file_uploader("Importer un fichier CSV ou GeoJSON", type=["csv", "geojson"])
    
    if uploaded_file is not None:
        st.success("Fichier chargé ! (Logique de traitement à ajouter)")
    
    st.divider()
    st.info("Cette application est évolutive. Vous pourrez ajouter des filtres ici.")

# --- CRÉATION DE LA CARTE ---
# Coordonnées centrées sur la France
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles=map_style)

# Ajout d'un exemple de marqueur (Paris)
folium.Marker(
    [48.8566, 2.3522], 
    popup="<b>Paris</b><br>Capitale", 
    tooltip="Cliquez ici"
).add_to(m)

# --- AFFICHAGE DE LA CARTE ---
# La carte prend 70% de la hauteur de l'écran environ
st_data = st_folium(m, width="100%", height=600)

# --- INTERACTION ---
# Affiche les coordonnées si on clique sur la carte
if st_data['last_clicked']:
    lat = st_data['last_clicked']['lat']
    lng = st_data['last_clicked']['lng']
    st.write(f"📍 **Dernier clic détecté :** Latitude {lat:.4f}, Longitude {lng:.4f}")
