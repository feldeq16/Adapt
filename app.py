import streamlit as st
import pandas as pd
import pydeck as pdk
import os

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(
    layout="wide", 
    page_title="Portail Cartographique",
    initial_sidebar_state="expanded"
)

# Titre
st.title("🗺️ Visualisation de Données Climatiques")
st.markdown("""
<style>
    .stApp { margin-top: -50px; }
</style>
""", unsafe_allow_html=True)

# Nom exact du dossier (Attention aux majuscules/accents)
DOSSIER_DONNEES = 'Données'

# --- 2. FONCTION DE CHARGEMENT ROBUSTE (CACHE) ---
@st.cache_data(ttl=3600, show_spinner=False)
def charger_toutes_les_donnees(dossier):
    """
    Lit tous les fichiers TXT du dossier, nettoie les données,
    et assigne une couleur unique par fichier.
    """
    all_data = pd.DataFrame()
    
    if not os.path.exists(dossier):
        return None, f"Le dossier '{dossier}' est introuvable."
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    
    if not fichiers:
        return None, "Aucun fichier .txt trouvé."

    # Palette de couleurs (RGB) pour distinguer les fichiers
    # Rouge, Vert, Bleu, Orange, Violet, Cyan, Jaune
    couleurs = [
        [231, 76, 60],   # Rouge Aladin
        [46, 204, 113],  # Vert
        [52, 152, 219],  # Bleu
        [243, 156, 18],  # Orange
        [155, 89, 182],  # Violet
        [26, 188, 156],  # Turquoise
        [241, 196, 15]   # Jaune
    ]
    
    # Barre de progression dans l'interface principale pour faire patienter
    barre_progression = st.progress(0, text="Démarrage du chargement...")
    
    for i, fichier in enumerate(fichiers):
        chemin = os.path.join(dossier, fichier)
        
        # Mise à jour de la barre
        barre_progression.progress((i)/len(fichiers), text=f"Lecture de {fichier}...")
        
        try:
            # Lecture optimisée : on saute les commentaires '#' et on force l'encodage
            # On lit TOUT le fichier (pas de limite nrows) car PyDeck peut le gérer
            df = pd.read_csv(
                chemin, 
                sep=';', 
                comment='#', 
                encoding='latin-1', 
                engine='python'
            )
            
            # Nettoyage des noms de colonnes (retirer les espaces)
            df.columns = [c.strip() for c in df.columns]
            
            # Vérification des colonnes GPS
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # Conversion en nombres (écarte les erreurs)
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                df = df.dropna(subset=['Latitude', 'Longitude'])
                
                # Ajout des métadonnées pour la carte
                df['source_fichier'] = fichier
                
                # Assigner une couleur fixe à ce fichier
                couleur_attribuee = couleurs[i % len(couleurs)]
                df['r'] = couleur_attribuee[0]
                df['g'] = couleur_attribuee[1]
                df['b'] = couleur_attribuee[2]
                
                # On concatène au gros tableau
                all_data = pd.concat([all_data, df], ignore_index=True)
                
        except Exception as e:
            st.error(f"Erreur sur
