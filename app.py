import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Mon Portail Cartographique")
st.title("🗺️ Mon Portail Cartographique")

# Nom du dossier où vous avez mis vos fichiers
DOSSIER_DONNEES = 'Données'

# --- SIDEBAR ---
with st.sidebar:
    st.header("🗂️ Données chargées")
    
    # Vérification que le dossier existe
    if not os.path.exists(DOSSIER_DONNEES):
        st.error(f"Le dossier '{DOSSIER_DONNEES}' n'existe pas sur GitHub !")
        fichiers_trouves = []
    else:
        # On ne prend que les fichiers .txt du dossier
        fichiers_trouves = [f for f in os.listdir(DOSSIER_DONNEES) if f.endswith('.txt')]
        st.success(f"{len(fichiers_trouves)} fichiers trouvés.")

    st.divider()
    map_style = st.selectbox("Fond de carte", ["OpenStreetMap", "CartoDB Positron"])

# --- CARTE ---
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles=map_style)

# --- LECTURE ET AFFICHAGE ---
for fichier in fichiers_trouves:
    chemin_complet = os.path.join(DOSSIER_DONNEES, fichier)
    
    try:
        # L'option comment='#' dit à Pandas : "Si une ligne commence par #, ignore-la."
        # C'est magique, ça saute tout l'en-tête automatiquement.
        df = pd.read_csv(
            chemin_complet, 
            sep=';', 
            comment='#', 
            encoding='latin-1',
            engine='python'
        )
        
        # Nettoyage des noms de colonnes (enlève les espaces autour)
        df.columns = df.columns.str.strip()
        
        # Vérification basique
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            fg = folium.FeatureGroup(name=fichier)
            
            # Affichage des points
            for idx, row in df.iterrows():
                # Création du contenu de la bulle
                infos = "<br>".join([f"<b>{k}:</b> {v}" for k, v in row.items() if k not in ['Latitude', 'Longitude']])
                
                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=5,
                    color="blue", # Vous pourrez personnaliser la couleur plus tard
                    fill=True,
                    popup=folium.Popup(infos, max_width=300),
                    tooltip=str(row.values[0])
                ).add_to(fg)
            
            fg.add_to(m)
        else:
            st.sidebar.warning(f"⚠️ {fichier} ignoré (Pas de colonnes Latitude/Longitude)")
            
    except Exception as e:
        st.sidebar.error(f"Erreur sur {fichier} : {e}")

# Contrôle des calques
folium.LayerControl().add_to(m)
st_folium(m, width="100%", height=700)
