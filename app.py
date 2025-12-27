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

# Nom exact du dossier
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

    # Palette de couleurs (RGB)
    couleurs = [
        [231, 76, 60],   # Rouge
        [46, 204, 113],  # Vert
        [52, 152, 219],  # Bleu
        [243, 156, 18],  # Orange
        [155, 89, 182],  # Violet
        [26, 188, 156],  # Turquoise
        [241, 196, 15]   # Jaune
    ]
    
    barre_progression = st.progress(0, text="Démarrage du chargement...")
    
    for i, fichier in enumerate(fichiers):
        chemin = os.path.join(dossier, fichier)
        barre_progression.progress((i)/len(fichiers), text=f"Lecture de {fichier}...")
        
        try:
            # Lecture optimisée
            df = pd.read_csv(
                chemin, 
                sep=';', 
                comment='#', 
                encoding='latin-1', 
                engine='python'
            )
            
            # Nettoyage
            df.columns = [c.strip() for c in df.columns]
            
            # Vérification des colonnes GPS
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                df = df.dropna(subset=['Latitude', 'Longitude'])
                
                # Métadonnées
                df['source_fichier'] = fichier
                
                # Couleur
                couleur_attribuee = couleurs[i % len(couleurs)]
                df['r'] = couleur_attribuee[0]
                df['g'] = couleur_attribuee[1]
                df['b'] = couleur_attribuee[2]
                
                all_data = pd.concat([all_data, df], ignore_index=True)
                
        except Exception as e:
            # C'est ici que ça plantait : la ligne est maintenant complète
            print(f"Erreur ignorée sur {fichier}: {e}")
            
    barre_progression.empty()
    
    return all_data, None

# --- 3. BARRE LATÉRALE & FILTRES ---
with st.sidebar:
    st.header("🎛️ Contrôles")
    
    if st.button("Recharger les données"):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    with st.spinner("Analyse des fichiers en cours..."):
        df_total, erreur = charger_toutes_les_donnees(DOSSIER_DONNEES)

    if erreur:
        st.error(erreur)
        st.stop()
        
    if not df_total.empty:
        fichiers_dispos = df_total['source_fichier'].unique()
        st.write(f"**{len(df_total)} points** chargés.")
        
        st.subheader("Afficher/Masquer :")
        selection = []
        for f in fichiers_dispos:
            sample = df_total[df_total['source_fichier'] == f].iloc[0]
            color_hex = f"#{int(sample['r']):02x}{int(sample['g']):02x}{int(sample['b']):02x}"
            
            if st.checkbox(f, value=True, key=f):
                selection.append(f)
                st.markdown(f"<span style='color:{color_hex}'>⬤</span> <small>{f}</small>", unsafe_allow_html=True)
        
        df_map = df_total[df_total['source_fichier'].isin(selection)]
        
    else:
        st.warning("Aucune donnée valide trouvée.")
        st.stop()

# --- 4. AFFICHAGE DE LA CARTE PYDECK ---
if df_map.empty:
    st.info("Veuillez cocher au moins un fichier.")
else:
    mid_lat = df_map['Latitude'].mean()
    mid_lon = df_map['Longitude'].mean()

    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=5.5,
        pitch=0,
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[Longitude, Latitude]',
        get_color='[r, g, b, 140]',
        get_radius=3000,
        pickable=True,
        auto_highlight=True
    )

    tooltip = {
        "html": """
            <div style="background-color: #f0f0f0; color: black; padding: 5px; border-radius: 5px;">
                <b>Source:</b> {source_fichier}<br>
                <b>Point:</b> {Point}<br>
                <b>Période:</b> {Période}<br>
                <b>Contexte:</b> {Contexte}<br>
                <b>ATXHWD:</b> {ATXHWD}
            </div>
        """,
        "style": {"color": "white"}
    }

    st.pydeck_chart(pdk.Deck(
        map_style=None, 
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    ))
