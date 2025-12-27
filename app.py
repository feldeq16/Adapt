import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import FastMarkerCluster
import pandas as pd
import os
import gc # Garbage Collector (le nettoyeur de mémoire)

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mon Portail Cartographique")
st.title("🗺️ Mon Portail Cartographique (Mode Léger)")

DOSSIER_DONNEES = 'Données'

# --- FONCTION DE CHARGEMENT HYPER OPTIMISÉE ---
@st.cache_data(ttl=3600) # Garde en cache 1 heure max
def charger_donnees_legeres(dossier):
    donnees_chargees = []
    
    if not os.path.exists(dossier):
        return []
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    
    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        try:
            # 1. On lit juste la 1ère ligne pour voir les colonnes
            # Cela évite de charger tout le fichier si les colonnes ne sont pas bonnes
            preview = pd.read_csv(chemin, sep=';', comment='#', encoding='latin-1', nrows=1)
            cols = [c.strip() for c in preview.columns]
            
            # On identifie les colonnes vitales
            if 'Latitude' in cols and 'Longitude' in cols:
                # On détermine quelles colonnes garder pour économiser la RAM
                # On garde Lat, Lon, et la 1ère colonne (souvent le nom du point)
                cols_a_garder = ['Latitude', 'Longitude', cols[0]]
                if 'ATXHWD' in cols: cols_a_garder.append('ATXHWD')
                
                # 2. Lecture restreinte
                df = pd.read_csv(
                    chemin, 
                    sep=';', 
                    comment='#', 
                    encoding='latin-1',
                    engine='python',
                    usecols=lambda c: c.strip() in cols_a_garder, # On ne charge que l'utile
                    nrows=2000 # <--- STOP à 2000 lignes par fichier pour sauver la RAM
                )
                
                # Nettoyage des noms de colonnes
                df.columns = df.columns.str.strip()
                
                # Conversion en numérique (pour alléger la mémoire, les nombres prennent moins de place que le texte)
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                df = df.dropna(subset=['Latitude', 'Longitude'])
                
                donnees_chargees.append((fichier, df))
                
                # 3. On vide la mémoire intermédiaire immédiatement
                del df
                gc.collect()
                
        except Exception as e:
            print(f"Erreur lecture {fichier}: {e}")
            continue
            
    return donnees_chargees

# --- APPLICATION ---

with st.sidebar:
    st.header("🗂️ Données (Mode Éco)")
    
    with st.spinner('Chargement optimisé...'):
        liste_donnees = charger_donnees_legeres(DOSSIER_DONNEES)
    
    if not liste_donnees:
        st.warning("Aucune donnée chargée ou dossier vide.")
    else:
        st.success(f"{len(liste_donnees)} fichiers chargés.")
        st.info("⚠️ Affichage limité à 2000 points/fichier pour la stabilité.")

    st.divider()
    map_style = st.selectbox("Fond de carte", ["OpenStreetMap", "CartoDB Positron"])

# --- CARTE AVEC FASTMARKERCLUSTER (Plus léger que MarkerCluster) ---
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles=map_style)

for nom_fichier, df in liste_donnees:
    # On prépare les données pour FastMarkerCluster
    # C'est une méthode qui envoie juste les coordonnées brutes à la carte (très léger)
    # Le callback permet d'afficher le nom quand on clique
    
    callback = ('function (row) {' 
                'var marker = L.marker(new L.LatLng(row[0], row[1]), {color: "red"});'
                'var icon = L.AwesomeMarkers.icon({'
                "icon: 'info-sign',"
                "markerColor: 'blue',"
                "prefix: 'glyphicon'"
                '});'
                'marker.setIcon(icon);'
                'return marker};')

    # FastMarkerCluster est beaucoup plus performant pour la RAM
    cluster = FastMarkerCluster(
        data=list(zip(df['Latitude'], df['Longitude'])),
        name=nom_fichier,
    ).add_to(m)

folium.LayerControl().add_to(m)

# Affichage sans renvoyer d'objets (économie de RAM navigateur)
st_folium(m, width="100%", height=700, returned_objects=[])
