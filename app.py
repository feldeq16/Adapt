import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster # <--- L'outil magique
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mon Portail Cartographique")
st.title("🗺️ Mon Portail Cartographique")

DOSSIER_DONNEES = 'Données'

# --- FONCTION DE CHARGEMENT (CACHE) ---
@st.cache_data
def charger_les_donnees(dossier):
    donnees_chargees = []
    if not os.path.exists(dossier):
        return []
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    
    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        try:
            # Lecture optimisée
            df = pd.read_csv(
                chemin, 
                sep=';', 
                comment='#', 
                encoding='latin-1', 
                engine='python'
            )
            df.columns = df.columns.str.strip()
            
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # On allège les données : on garde juste ce qui est nécessaire pour la carte
                cols_utiles = ['Latitude', 'Longitude'] + [c for c in df.columns if c not in ['Latitude', 'Longitude']][:5]
                donnees_chargees.append((fichier, df[cols_utiles]))
                
        except Exception:
            continue
            
    return donnees_chargees

# --- SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Données")
    with st.spinner('Chargement...'):
        liste_donnees = charger_les_donnees(DOSSIER_DONNEES)
    st.success(f"{len(liste_donnees)} fichiers prêts.")
    
    st.divider()
    map_style = st.selectbox("Fond de carte", ["OpenStreetMap", "CartoDB Positron"])

# --- CARTE ---
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles=map_style)

# --- DESSIN OPTIMISÉ (CLUSTERING) ---
for nom_fichier, df in liste_donnees:
    # Au lieu d'ajouter au map, on ajoute à un "Cluster"
    # Cela groupe les points automatiquement
    cluster = MarkerCluster(name=nom_fichier).add_to(m)
    
    # On limite à 2000 points par fichier pour éviter le crash navigateur si le fichier est énorme
    # Si vous avez une machine puissante, vous pouvez enlever .head(2000)
    data_to_plot = df.head(2000) 
    
    for row in data_to_plot.itertuples():
        # Construction du texte
        infos = "<br>".join([f"<b>{k}:</b> {v}" for k, v in row._asdict().items() if k not in ['Index', 'Latitude', 'Longitude']])
        
        folium.CircleMarker(
            location=[row.Latitude, row.Longitude],
            radius=5,
            color="blue",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(infos, max_width=200)
        ).add_to(cluster) # <-- On ajoute au cluster, pas à la carte directement

# Ajout du contrôle des couches
folium.LayerControl().add_to(m)

# --- AFFICHAGE FINAL OPTIMISÉ ---
# returned_objects=[] empêche Streamlit de recharger la page à chaque mouvement de souris
# Cela rend la carte beaucoup plus fluide
st_folium(m, width="100%", height=700, returned_objects=[])
