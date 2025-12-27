import streamlit as st
import pandas as pd
import os

# --- CONFIGURATION STRICTE ---
st.set_page_config(layout="wide", page_title="Mon Portail Cartographique")
st.title("🗺️ Visualisation Haute Performance")

DOSSIER_DONNEES = 'Données'

# --- FONCTION DE CHARGEMENT ---
@st.cache_data(ttl=3600)
def charger_donnees_natives(dossier):
    # On va créer un seul gros tableau avec toutes les données
    # C'est beaucoup plus efficace pour st.map que plein de petits fichiers
    all_data = []
    
    if not os.path.exists(dossier):
        return pd.DataFrame()
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    
    # Barre de progression
    progression = st.progress(0)
    
    for i, fichier in enumerate(fichiers):
        chemin = os.path.join(dossier, fichier)
        try:
            # On lit tout mais uniquement les colonnes GPS
            # st.map a besoin de colonnes nommées 'latitude' et 'longitude' ou 'lat'/'lon'
            df = pd.read_csv(
                chemin, 
                sep=';', 
                comment='#', 
                encoding='latin-1',
                engine='python',
                usecols=lambda c: 'lat' in c.lower() or 'lon' in c.lower() or 'lng' in c.lower()
            )
            
            # Nettoyage
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Renommage standard pour que Streamlit comprenne
            df = df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            
            # On vérifie qu'on a bien lat et lon
            if 'lat' in df.columns and 'lon' in df.columns:
                # On nettoie les erreurs (virgules au lieu de points, etc)
                df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
                df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
                df = df.dropna()
                
                # On ajoute une colonne pour identifier la source (optionnel)
                # df['source'] = fichier 
                
                all_data.append(df)
        
        except Exception as e:
            print(f"Erreur {fichier}: {e}")
            
        # Mise à jour de la barre
        progression.progress((i + 1) / len(fichiers))

    progression.empty()
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

# --- INTERFACE ---
with st.sidebar:
    st.header("Données")
    st.write("Chargement en mode 'Big Data'...")
    
    df_final = charger_donnees_natives(DOSSIER_DONNEES)
    
    if not df_final.empty:
        st.success(f"✅ {len(df_final)} points affichés !")
        st.info("Utilisation de la technologie DeckGL (Native) pour la performance.")
    else:
        st.error("Aucune donnée valide trouvée.")

# --- AFFICHAGE CARTE NATIVE ---
# st.map est incapable de planter, même avec 1 million de points
# Par contre, c'est juste des points (pas de clic pour l'instant)
if not df_final.empty:
    st.map(df_final, size=20, color='#0044ff')
else:
    st.write("En attente de données...")
