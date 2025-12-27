import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import unicodedata

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Analyse Climatique")
st.title("🌡️ Analyse Climatique (Mode Debug)")

DOSSIER_DONNEES = 'Données'

# --- OUTILS DE NETTOYAGE ---
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def trouver_header_et_lire(chemin):
    """
    Tente plusieurs encodages (UTF-8 d'abord, puis Latin-1)
    et cherche la ligne de départ des données.
    """
    # Liste des encodages à tester par ordre de priorité
    encodages_a_tester = ['utf-8', 'latin-1', 'cp1252']
    
    for enc in encodages_a_tester:
        try:
            # 1. On essaie de lire les 50 premières lignes avec cet encodage
            with open(chemin, 'r', encoding=enc) as f:
                lignes = [f.readline() for _ in range(50)]
            
            # Si pas d'erreur, on cherche la ligne Latitude/Longitude
            header_row = None
            sep = ';' # Par défaut
            
            for i, ligne in enumerate(lignes):
                ligne_clean = remove_accents(ligne).lower()
                
                # On cherche les mots clés vitaux
                if 'latitude' in ligne_clean and 'longitude' in ligne_clean:
                    header_row = i
                    # Détection du séparateur
                    if ';' in ligne: sep = ';'
                    elif ',' in ligne: sep = ','
                    elif '\t' in ligne: sep = '\t'
                    break
            
            if header_row is not None:
                # Si on a trouvé le header, on charge le tout avec cet encodage
                df = pd.read_csv(chemin, sep=sep, header=header_row, encoding=enc, engine='python')
                return df, None, enc # Succès !
                
        except UnicodeDecodeError:
            # Si cet encodage échoue, on passe au suivant dans la boucle
            continue
        except Exception as e:
            return None, str(e), enc

    return None, "Impossible de lire le fichier (Encodage ou Header introuvable)", "Inconnu"

# --- 2. CHARGEMENT ---
@st.cache_data(ttl=3600, show_spinner=False)
def charger_donnees(dossier):
    all_data = pd.DataFrame()
    debug_infos = [] 
    
    if not os.path.exists(dossier):
        return pd.DataFrame(), [f"❌ Dossier '{dossier}' introuvable."], None
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    
    barre = st.progress(0, text="Analyse des fichiers...")
    
    for i, fichier in enumerate(fichiers):
        chemin = os.path.join(dossier, fichier)
        barre.progress((i)/len(fichiers))
        
        # Lecture intelligente avec détection d'encodage
        df, erreur, enc_detecte = trouver_header_et_lire(chemin)
        
        if erreur:
            debug_infos.append(f"❌ {fichier} : {erreur}")
            continue
            
        # Nettoyage des colonnes (strip)
        df.columns = [c.strip() for c in df.columns]
        
        # Info debug pour vous
        cols_vues = df.columns.tolist()
        
        # Recherche de la colonne Période (Robustesse maximale)
        col_periode = None
        for col in df.columns:
            # On nettoie tout : "PÃ©riode" devient "periode" si l'encodage est bon
            col_clean = remove_accents(col.lower())
            if 'eriode' in col_clean or 'horizon' in col_clean:
                col_periode = col
                break
        
        if col_periode:
            df['Horizon_Filter'] = df[col_periode].astype(str).str.strip()
            # On ajoute l'info de l'encodage dans le debug
            debug_infos.append(f"✅ {fichier} ({enc_detecte}) : Période trouvée ('{col_periode}')")
        else:
            df['Horizon_Filter'] = "Non défini"
            # Affiche les 3 premières colonnes pour aider au diagnostic
            debug_infos.append(f"⚠️ {fichier} ({enc_detecte}) : Colonne Période introuvable. Colonnes vues : {cols_vues[:5]}...")

        # Scénario
        nom_min = fichier.lower()
        if "rcp2.6" in nom_min or "rcp26" in nom_min: df['Scenario'] = "RCP 2.6"
        elif "rcp4.5" in nom_min or "rcp45" in nom_min: df['Scenario'] = "RCP 4.5"
        elif "rcp8.5" in nom_min or "rcp85" in nom_min: df['Scenario'] = "RCP 8.5"
        else: df['Scenario'] = "Autre"

        # Conversion Données
        if {'Latitude', 'Longitude', 'ATXHWD'}.issubset(df.columns):
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            if df['ATXHWD'].dtype == object:
                 df['ATXHWD'] = pd.to_numeric(df['ATXHWD'].str.replace(',', '.'), errors='coerce')
            else:
                 df['ATXHWD'] = pd.to_numeric(df['ATXHWD'], errors='coerce')
            
            df = df.dropna(subset=['Latitude', 'Longitude', 'ATXHWD'])
            df['Source'] = fichier
            all_data = pd.concat([all_data, df], ignore_index=True)
        else:
            debug_infos.append(f"❌ {fichier} : Colonnes Lat/Lon/ATXHWD manquantes.")

    barre.empty()
    return all_data, debug_infos, None

# --- 3. INTERFACE ---
with st.sidebar:
    st.header("🎛️ Paramètres")
    
    if st.button("Recharger"):
        st.cache_data.clear()
        st.rerun()

    # Appel du chargement
    df_total, debug_logs, erreur_globale = charger_donnees(DOSSIER_DONNEES)
    
    if erreur_globale:
        st.error(erreur_globale)
        st.stop()

    # --- ZONE DE DÉBUG ---
    with st.expander("🕵️ DIAGNOSTIC (Voir détails)"):
        st.write("État de lecture des fichiers :")
        for log in debug_logs:
            if "❌" in log: st.error(log)
            elif "⚠️" in log: st.warning(log)
            else: st.success(log)
    # ---------------------

    if df_total.empty:
        st.warning("Aucune donnée chargée.")
        st.stop()

    # Filtres
    scenarios = sorted(df_total['Scenario'].unique())
    choix_rcp = st.radio("Scénario :", scenarios)

    df_rcp = df_total[df_total['Scenario'] == choix_rcp]
    horizons = sorted(df_rcp['Horizon_Filter'].unique())
    
    if not horizons or (len(horizons) == 1 and horizons[0] == "Non défini"):
        st.warning("⚠️ Impossible de filtrer par Période.")
        choix_horizon = None
    else:
        choix_horizon = st.radio("Période :", horizons)

    # Légende Gradient
    val_min = df_total['ATXHWD'].min()
    val_max = df_total['ATXHWD'].max()
    st.divider()
    st.write(f"Min: {val_min:.1f} | Max: {val_max:.1f}")
    
    cmap = plt.get_cmap("coolwarm")
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(4, 0.4))
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.set_axis_off()
    st.pyplot(fig)

# --- 4. CARTE ---
if choix_horizon:
    df_map = df_rcp[df_rcp['Horizon_Filter'] == choix_horizon].copy()
else:
    df_map = pd.DataFrame()

if not df_map.empty:
    norm = mcolors.Normalize(vmin=val_min, vmax=val_max)
    colors = cmap(norm(df_map['ATXHWD'].values))
    df_map['r'] = (colors[:, 0] * 255).astype(int)
    df_map['g'] = (colors[:, 1] * 255).astype(int)
    df_map['b'] = (colors[:, 2] * 255).astype(int)

    st.info(f"Visualisation : {choix_rcp} - {choix_horizon}")

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
        get_radius=4000,
        pickable=True,
        auto_highlight=True
    )

    tooltip = {
        "html": "<b>ATXHWD:</b> {ATXHWD}<br><b>Période:</b> {Horizon_Filter}",
        "style": {"color": "white"}
    }

    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer], tooltip=tooltip))
else:
    st.write("Sélectionnez un horizon valide.")
