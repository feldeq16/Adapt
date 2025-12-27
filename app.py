import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import unicodedata

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Analyse Climatique", initial_sidebar_state="expanded")

st.title("🌡️ Analyse des Indices Climatiques (ATXHWD)")
st.markdown("""
<style>
    .stApp { margin-top: -30px; }
    div[data-testid="stSidebar"] { background-color: #f8f9fa; }
</style>
""", unsafe_allow_html=True)

DOSSIER_DONNEES = 'Données'

# Fonction pour enlever les accents (Période -> periode) pour la comparaison
def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

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
            # Lecture (latin-1 gère bien les accents français Windows)
            df = pd.read_csv(chemin, sep=';', comment='#', encoding='latin-1', engine='python')
            
            # Nettoyage des noms de colonnes
            df.columns = [c.strip() for c in df.columns]
            
            # --- DÉTECTION ROBUSTE DE LA COLONNE PÉRIODE ---
            col_periode_trouvee = None
            for col in df.columns:
                # On nettoie le nom de la colonne (minuscule + sans accent)
                col_clean = remove_accents(col.lower())
                if 'periode' in col_clean or 'horizon' in col_clean:
                    col_periode_trouvee = col
                    break
            
            if col_periode_trouvee:
                # On standardise le contenu
                df['Horizon_Filter'] = df[col_periode_trouvee].astype(str).str.strip()
            else:
                # Si introuvable, on le signale mais on ne plante pas
                df['Horizon_Filter'] = "Non défini"

            # --- DÉTECTION SCÉNARIO RCP ---
            # On cherche dans le nom du fichier
            nom_min = fichier.lower()
            if "rcp2.6" in nom_min or "rcp26" in nom_min:
                df['Scenario'] = "RCP 2.6"
            elif "rcp4.5" in nom_min or "rcp45" in nom_min:
                df['Scenario'] = "RCP 4.5"
            elif "rcp8.5" in nom_min or "rcp85" in nom_min:
                df['Scenario'] = "RCP 8.5"
            else:
                df['Scenario'] = "Autre"

            # --- CONVERSION ET NETTOYAGE ---
            if {'Latitude', 'Longitude', 'ATXHWD'}.issubset(df.columns):
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                
                # Gestion virgule/point pour ATXHWD
                if df['ATXHWD'].dtype == object:
                     df['ATXHWD'] = pd.to_numeric(df['ATXHWD'].str.replace(',', '.'), errors='coerce')
                else:
                     df['ATXHWD'] = pd.to_numeric(df['ATXHWD'], errors='coerce')

                df = df.dropna(subset=['Latitude', 'Longitude', 'ATXHWD'])
                df['Source'] = fichier
                
                all_data = pd.concat([all_data, df], ignore_index=True)
                
        except Exception as e:
            print(f"Erreur {fichier}: {e}")
            continue
            
    barre.empty()
    return all_data, None

# --- 3. INTERFACE ---
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    # Bouton reset cache (utile si bug)
    if st.button("Recharger les données"):
        st.cache_data.clear()
        st.rerun()

    df_total, erreur = charger_donnees(DOSSIER_DONNEES)
    
    if erreur:
        st.error(erreur)
        st.stop()
        
    if df_total.empty:
        st.warning("Aucune donnée valide trouvée.")
        st.stop()

    # Filtre 1 : RCP
    scenarios = sorted(df_total['Scenario'].unique())
    choix_rcp = st.radio("1. Scénario d'émission :", scenarios)

    # Filtre 2 : Horizon (Filtré selon le RCP choisi pour éviter les options vides)
    df_rcp = df_total[df_total['Scenario'] == choix_rcp]
    
    horizons = sorted(df_rcp['Horizon_Filter'].unique())
    if not horizons or (len(horizons) == 1 and horizons[0] == "Non défini"):
        st.error("⚠️ Colonne 'Période' non détectée dans les fichiers.")
        choix_horizon = None
    else:
        choix_horizon = st.radio("2. Horizon temporel :", horizons)

    st.divider()
    
    # Légende
    val_min = df_total['ATXHWD'].min()
    val_max = df_total['ATXHWD'].max()
    
    st.write(f"**Légende (ATXHWD)**")
    st.write(f"Min global: {val_min:.1f} | Max global: {val_max:.1f}")
    
    cmap = plt.get_cmap("coolwarm")
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(4, 0.4))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    st.pyplot(fig)

# --- 4. CARTE ---

# Application des filtres
if choix_horizon:
    df_map = df_rcp[df_rcp['Horizon_Filter'] == choix_horizon].copy()
else:
    df_map = pd.DataFrame()

if not df_map.empty:
    # Calcul des couleurs
    norm = mcolors.Normalize(vmin=val_min, vmax=val_max)
    
    colors = cmap(norm(df_map['ATXHWD'].values))
    df_map['r'] = (colors[:, 0] * 255).astype(int)
    df_map['g'] = (colors[:, 1] * 255).astype(int)
    df_map['b'] = (colors[:, 2] * 255).astype(int)

    # Stats
    moy = df_map['ATXHWD'].mean()
    st.info(f"Affichage : **{choix_rcp}** - **{choix_horizon}** | Moyenne de la zone : **{moy:.2f}**")

    # Vue
    view_state = pdk.ViewState(
        latitude=df_map['Latitude'].mean(),
        longitude=df_map['Longitude'].mean(),
        zoom=5.5,
        pitch=0
    )

    # Couche
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position='[Longitude, Latitude]',
        get_color='[r, g, b, 180]',
        get_radius=4000,
        pickable=True,
        auto_highlight=True
    )

    # Tooltip
    tooltip = {
        "html": """
            <div style="background-color: white; color: black; padding: 5px; border: 1px solid #ccc; font-size: 12px;">
                <b>Point:</b> {Point}<br>
                <b>ATXHWD:</b> {ATXHWD}<br>
                <b>Scenario:</b> {Scenario}<br>
                <b>Horizon:</b> {Horizon_Filter}
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
    st.warning("Aucune donnée correspondante à cette sélection.")
