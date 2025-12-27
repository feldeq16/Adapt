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
    st.write("Chargement des fichiers ALADIN/CNRM...")
    st.divider()
    map_style = st.selectbox(
        "Style de carte",
        ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark_Matter"]
    )

# --- CRÉATION DE LA CARTE ---
m = folium.Map(location=[46.603354, 1.888334], zoom_start=6, tiles=map_style)

# --- FONCTION POUR TROUVER LE DÉBUT DES DONNÉES ---
def trouver_header(fichier):
    """Lit le fichier ligne par ligne pour trouver où commencent les colonnes"""
    with open(fichier, 'r', encoding='latin-1') as f:
        for i, line in enumerate(f):
            # On cherche la ligne qui contient les mots clés des colonnes
            if "Point" in line and "Latitude" in line:
                return i
    return 0 # Si on ne trouve rien, on lit depuis le début

# --- CHARGEMENT ---
fichiers_txt = [f for f in os.listdir('.') if f.endswith('.txt')]

count_files = 0

if not fichiers_txt:
    st.warning("⚠️ Aucun fichier .txt trouvé.")
else:
    for fichier in fichiers_txt:
        # CORRECTION 1 : On ignore le fichier requirements.txt
        if fichier == "requirements.txt":
            continue
            
        try:
            # CORRECTION 2 : On cherche la ligne d'en-tête dynamiquement
            header_row = trouver_header(fichier)
            
            # On lit le CSV en sautant les lignes d'avant (skiprows)
            # On force le séparateur ";"
            df = pd.read_csv(
                fichier, 
                sep=';', 
                skiprows=header_row, 
                encoding='latin-1',
                engine='python' # Plus robuste pour les fichiers complexes
            )
            
            # Nettoyage des noms de colonnes (enlève les espaces et les points-virgules vides)
            df.columns = [c.strip() for c in df.columns]
            
            # Parfois une dernière colonne vide "Unnamed" apparaît à cause du dernier point-virgule
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Vérification des colonnes
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                
                feature_group = folium.FeatureGroup(name=fichier)
                
                # Limitation à 1000 points par fichier pour ne pas faire ramer la carte si les fichiers sont énormes
                # Si vous voulez tout, enlevez [:1000]
                data_to_plot = df
                
                for index, row in data_to_plot.iterrows():
                    texte_bulle = f"""
                    <b>Point:</b> {row.get('Point', '-')}<br>
                    <b>Période:</b> {row.get('Période', '-')}<br>
                    <b>ATXHWD:</b> {row.get('ATXHWD', '-')}
                    """
                    
                    folium.CircleMarker(
                        location=[row['Latitude'], row['Longitude']],
                        radius=5,
                        color="red" if "RCP8.5" in fichier else "green" if "RCP2.6" in fichier else "blue",
                        fill=True,
                        fill_opacity=0.6,
                        popup=folium.Popup(texte_bulle, max_width=300),
                        tooltip=f"{row.get('Point', 'Point')}"
                    ).add_to(feature_group)
                
                feature_group.add_to(m)
                st.sidebar.success(f"✅ {fichier} : {len(df)} points chargés")
                count_files += 1
            else:
                st.sidebar.warning(f"⚠️ {fichier} : Colonnes non trouvées. Colonnes lues : {list(df.columns)}")
                
        except Exception as e:
            st.sidebar.error(f"Erreur sur {fichier}: {e}")

if count_files == 0:
    st.sidebar.error("Aucun fichier de données valide n'a pu être lu.")

# Ajout du contrôle des couches
folium.LayerControl().add_to(m)

# --- AFFICHAGE ---
st_folium(m, width="100%", height=700)
