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
    st.write("Fichiers détectés (format Point;Latitude;Longitude...) :")
    
    st.divider()
    map_style = st.selectbox(
        "Style de carte",
        ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark_Matter"]
    )

# --- CRÉATION DE LA CARTE ---
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles=map_style)

# --- CHARGEMENT DES FICHIERS ---
fichiers_txt = [f for f in os.listdir('.') if f.endswith('.txt')]

if not fichiers_txt:
    st.warning("⚠️ Aucun fichier .txt trouvé dans le dossier GitHub.")
else:
    for fichier in fichiers_txt:
        try:
            # On force le séparateur ";" et on gère l'encodage (utf-8 ou latin-1 pour le français)
            try:
                df = pd.read_csv(fichier, sep=';', engine='python', encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(fichier, sep=';', engine='python', encoding='latin-1')
            
            # Nettoyage des noms de colonnes (enlève les espaces invisibles)
            df.columns = df.columns.str.strip()
            
            # Vérification que les colonnes vitales sont là
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                
                # Création du groupe de calques pour ce fichier
                feature_group = folium.FeatureGroup(name=fichier)
                
                for index, row in df.iterrows():
                    # Construction du texte qui s'affiche au survol
                    # On utilise .get() pour éviter les erreurs si une colonne est vide
                    texte_bulle = f"""
                    <b>Point:</b> {row.get('Point', 'N/A')}<br>
                    <b>Contexte:</b> {row.get('Contexte', '-')}<br>
                    <b>Période:</b> {row.get('Période', '-')}<br>
                    <b>ATXHWD:</b> {row.get('ATXHWD', '-')}
                    """
                    
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=6,
                        color="#3388ff",
                        fill=True,
                        fill_color="#3388ff",
                        fill_opacity=0.7,
                        popup=folium.Popup(texte_bulle, max_width=300),
                        tooltip=f"{row.get('Point', 'Point')}" # Info-bulle rapide
                    ).add_to(feature_group)
                
                feature_group.add_to(m)
                st.sidebar.success(f"✅ {fichier} : {len(df)} points")
            else:
                st.sidebar.error(f"⚠️ {fichier} : Colonnes 'Latitude' ou 'Longitude' introuvables. Colonnes vues : {list(df.columns)}")
                
        except Exception as e:
            st.sidebar.error(f"Erreur lecture {fichier}: {e}")

# Ajout du contrôle des couches
folium.LayerControl().add_to(m)

# --- AFFICHAGE ---
st_folium(m, width="100%", height=700)
