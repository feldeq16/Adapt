import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Analyse Climatique", initial_sidebar_state="expanded")

# Titre et Style
st.title("🌡️ Analyse des Indices Climatiques (ATXHWD)")
st.markdown("""
<style>
    .stApp { margin-top: -30px; }
    div[data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

DOSSIER_DONNEES = 'Données'

# --- 2. FONCTION DE CHARGEMENT ---
@st.cache_data(ttl=3600, show_spinner=False)
def charger_donnees(dossier):
    all_data = pd.DataFrame()
    if not os.path.exists(dossier):
        return pd.DataFrame(), f"Dossier '{dossier}' introuvable."
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    
    barre = st.progress(0, text="Chargement des modèles...")
    
    for i, fichier in enumerate(fichiers):
        chemin = os.path.join(dossier, fichier)
        barre.progress((i)/len(fichiers))
        
        try:
            # Lecture
            df = pd.read_csv(chemin, sep=';', comment='#', encoding='latin-1', engine='python')
            
            # Nettoyage Colonnes
            df.columns = [c.strip() for c in df.columns]
            
            # Détection RCP dans le nom du fichier
            if "RCP2.6" in fichier or "rcp26" in fichier:
                df['Scenario'] = "RCP 2.6"
            elif "RCP4.5" in fichier or "rcp45" in fichier:
                df['Scenario'] = "RCP 4.5"
            elif "RCP8.5" in fichier or "rcp85" in fichier:
                df['Scenario'] = "RCP 8.5"
            else:
                df['Scenario'] = "Autre"

            # Vérification colonnes vitales
            if {'Latitude', 'Longitude', 'ATXHWD'}.issubset(df.columns):
                # Conversion numérique
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                # Gestion des virgules potentielles dans les données scientifiques
                if df['ATXHWD'].dtype == object:
                     df['ATXHWD'] = pd.to_numeric(df['ATXHWD'].str.replace(',', '.'), errors='coerce')
                else:
                     df['ATXHWD'] = pd.to_numeric(df['ATXHWD'], errors='coerce')

                df = df.dropna(subset=['Latitude', 'Longitude', 'ATXHWD'])
                df['Source'] = fichier
                
                # Normalisation de la colonne Horizon/Période
                # On cherche une colonne qui contient "Période" ou "Horizon"
                col_horizon = next((c for c in df.columns if 'période' in c.lower() or 'horizon' in c.lower()), None)
                if col_horizon:
                    df['Horizon_Filter'] = df[col_horizon].astype(str).str.strip()
                else:
                    df['Horizon_Filter'] = "Inconnu"

                all_data = pd.concat([all_data, df], ignore_index=True)
                
        except Exception:
            continue
            
    barre.empty()
    return all_data, None

# --- 3. BARRE LATÉRALE (FILTRES) ---
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    # Chargement
    df_total, erreur = charger_donnees(DOSSIER_DONNEES)
    if erreur:
        st.error(erreur)
        st.stop()
        
    if df_total.empty:
        st.warning("Aucune donnée valide.")
        st.stop()

    st.subheader("1. Scénario RCP")
    # Récupération des scénarios dispos
    scenarios_dispos = sorted(df_total['Scenario'].unique())
    choix_rcp = st.radio("Choisir le scénario d'émission :", scenarios_dispos)

    st.subheader("2. Horizon Temporel")
    # Récupération des horizons dispos pour ce scénario
    df_rcp = df_total[df_total['Scenario'] == choix_rcp]
    horizons_dispos = sorted(df_rcp['Horizon_Filter'].unique())
    
    if not horizons_dispos:
        st.error("Colonne Période/Horizon introuvable dans le fichier.")
        choix_horizon = None
    else:
        choix_horizon = st.radio("Choisir la période :", horizons_dispos)

    st.divider()
    
    # Légende du Gradient
    st.subheader("Légende (ATXHWD)")
    st.write("Indicateur de chaleur")
    
    # Création d'une barre de couleur visuelle pour la sidebar
    cmap = plt.get_cmap("coolwarm") # Bleu -> Rouge
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    st.pyplot(fig)
    
    # Calcul des min/max globaux pour que la couleur soit comparable entre scénarios
    val_min = df_total['ATXHWD'].min()
    val_max = df_total['ATXHWD'].max()
    st.write(f"Min: {val_min:.1f} | Max: {val_max:.1f}")

# --- 4. FILTRAGE ET CALCUL COULEUR ---
if choix_horizon:
    df_map = df_rcp[df_rcp['Horizon_Filter'] == choix_horizon].copy()
else:
    df_map = pd.DataFrame()

if not df_map.empty:
    # Fonction pour convertir une valeur en couleur RGB
    norm = mcolors.Normalize(vmin=val_min, vmax=val_max)
    
    # On applique la colormap "coolwarm" (Bleu=Frais, Rouge=Chaud)
    # On peut changer par 'YlOrRd' (Jaune-Orange-Rouge) ou 'viridis'
    def get_color(val):
        rgba = cmap(norm(val))
        return [int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), 180] # 180 = Opacité

    # Application vectorisée (rapide)
    # On pré-calcule les couleurs
    colors = cmap(norm(df_map['ATXHWD'].values))
    df_map['r'] = (colors[:, 0] * 255).astype(int)
    df_map['g'] = (colors[:, 1] * 255).astype(int)
    df_map['b'] = (colors[:, 2] * 255).astype(int)

    # --- 5. AFFICHAGE CARTE ---
    
    # Statistique rapide
    moyenne = df_map['ATXHWD'].mean()
    st.info(f"**{len(df_map)} points** affichés pour **{choix_rcp} / {choix_horizon}**. Moyenne ATXHWD : **{moyenne:.2f}**")

    view_state = pdk.ViewState(
        latitude=df_map['Latitude'].mean(),
        longitude=df_map['Longitude'].mean(),
        zoom=5.5,
        pitch=0
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[Longitude, Latitude]',
        get_color='[r, g, b, 180]',
        get_radius=4000, # Rayon 4km
        pickable=True,
        auto_highlight=True
    )

    tooltip = {
        "html": """
            <div style="background-color: white; color: black; padding: 5px; border: 1px solid #ccc; font-size: 12px;">
                <b>📍 Point:</b> {Point}<br>
                <b>🌡️ ATXHWD:</b> {ATXHWD}<br>
                <b>📅 Période:</b> {Horizon_Filter}<br>
                <b>📝 Scénario:</b> {Scenario}
            </div>
        """
    }

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=view_state,
        layers=[layer],
        tooltip=tooltip
    ))

else:
    st.warning("Aucune donnée pour cette combinaison.")
