import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mon Portail Cartographique")
st.title("🗺️ Mon Portail Cartographique")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Données chargées")
    st.write("Les fichiers suivants ont été détectés et chargés automatiquement depuis le serveur :")
    
    # Sélecteur de fond de carte
    st.divider()
    map_style = st.selectbox(
        "Style de carte",
        ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark_Matter"]
    )

# --- CRÉATION DE LA CARTE ---
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles=map_style)

# --- CHARGEMENT AUTOMATIQUE DES FICHIERS .TXT ---
# On liste tous les fichiers du dossier actuel qui finissent par .txt
fichiers_txt = [f for f in os.listdir('.') if f.endswith('.txt')]

if not fichiers_txt:
    st.warning("⚠️ Aucun fichier .txt trouvé dans le dossier GitHub.")
else:
    for fichier in fichiers_txt:
        try:
            # L'option sep=None et engine='python' permet à Pandas de 
            # deviner tout seul si c'est des virgules, des points-virgules ou des tabulations.
            df = pd.read_csv(fichier, sep=None, engine='python')
            
            # Recherche des colonnes Lat/Lon
            cols = [c.lower() for c in df.columns]
            possible_lat = [col for col in df.columns if "lat" in col.lower()]
            possible_lon = [col for col in df.columns if "lon" in col.lower() or "lng" in col.lower()]

            if possible_lat and possible_lon:
                lat_col = possible_lat[0]
                lon_col = possible_lon[0]
                
                # Création d'un groupe pour ce fichier (permet de cocher/décocher dans la carte)
                feature_group = folium.FeatureGroup(name=fichier)
                
                for index, row in df.iterrows():
                    # On tente de récupérer le nom du point (souvent la 1ère colonne)
                    tooltip_text = str(row.iloc[0])
                    
                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=5,
                        fill=True,
                        tooltip=f"{fichier}: {tooltip_text}",
                        color="blue",
                        fill_color="cyan"
                    ).add_to(feature_group)
                
                feature_group.add_to(m)
                st.sidebar.success(f"✅ {fichier} : {len(df)} points")
            else:
                st.sidebar.warning(f"⚠️ {fichier} ignoré : Pas de colonne 'lat'/'lon' trouvée.")
                
        except Exception as e:
            st.sidebar.error(f"Erreur sur {fichier}: {e}")

# Ajout du panneau de contrôle des couches (pour masquer/afficher les fichiers)
folium.LayerControl().add_to(m)

# --- AFFICHAGE ---
st_folium(m, width="100%", height=700)
