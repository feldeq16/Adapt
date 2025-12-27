import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mon Portail Cartographique")
st.title("🗺️ Mon Portail Cartographique")

DOSSIER_DONNEES = 'Données'

# --- FONCTION DE CHARGEMENT OPTIMISÉE (CACHE) ---
@st.cache_data # <--- C'est ici que la magie opère
def charger_les_donnees(dossier):
    """Lit tous les fichiers une seule fois et les garde en mémoire"""
    donnees_chargees = [] # Liste pour stocker (Nom du fichier, DataFrame)
    
    if not os.path.exists(dossier):
        return []
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    
    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        try:
            # Lecture rapide
            df = pd.read_csv(
                chemin, 
                sep=';', 
                comment='#', 
                encoding='latin-1',
                engine='python' # Nécessaire pour le séparateur auto ou complexe
            )
            df.columns = df.columns.str.strip()
            
            # On ne garde que si Lat/Lon existent
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                donnees_chargees.append((fichier, df))
                
        except Exception:
            continue # On ignore silencieusement les fichiers illisibles pour aller vite
            
    return donnees_chargees

# --- APPLICATION ---

with st.sidebar:
    st.header("🗂️ Données")
    
    # Appel de la fonction cachée (rapide comme l'éclair au 2ème clic)
    with st.spinner('Chargement des données...'):
        liste_donnees = charger_les_donnees(DOSSIER_DONNEES)
    
    st.success(f"{len(liste_donnees)} fichiers chargés en mémoire.")
    
    st.divider()
    map_style = st.selectbox("Fond de carte", ["OpenStreetMap", "CartoDB Positron"])

# --- CONSTRUCTION DE LA CARTE ---
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles=map_style)

# On dessine seulement si on a des données
for nom_fichier, df in liste_donnees:
    fg = folium.FeatureGroup(name=nom_fichier)
    
    # Optimisation de la boucle (itertuples est 10x plus rapide que iterrows)
    for row in df.itertuples():
        # row.Latitude, row.Longitude, etc.
        
        # Astuce : On prépare le texte HTML simple
        # (Attention : itertuples transforme les noms de colonnes avec des espaces en _)
        infos = f"<b>Fichier:</b> {nom_fichier}<br>"
        
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=4,
            color="blue", 
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(infos, max_width=200)
        ).add_to(fg)
    
    fg.add_to(m)

folium.LayerControl().add_to(m)
st_folium(m, width="100%", height=700)
