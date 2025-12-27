import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import unicodedata
import uuid
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# --- 1. CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Portail Climatique Agrégé")
st.title("🌡️ Analyse Multi-Scénarios & Multi-Variables")

DOSSIER_DONNEES = 'Données'

# --- OUTILS ---
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def extraire_scenario_du_nom(nom_fichier):
    """Déduit le scénario RCP à partir du nom du fichier."""
    nom = nom_fichier.lower()
    if "2.6" in nom or "26" in nom: return "RCP 2.6"
    if "4.5" in nom or "45" in nom: return "RCP 4.5"
    if "8.5" in nom or "85" in nom: return "RCP 8.5"
    return "Autre"

def trouver_header_et_lire(chemin):
    encodages = ['utf-8', 'latin-1', 'cp1252']
    for enc in encodages:
        try:
            with open(chemin, 'r', encoding=enc) as f:
                lignes = [f.readline() for _ in range(50)]
            header_row = None
            sep = ';'
            for i, ligne in enumerate(lignes):
                clean = remove_accents(ligne).lower()
                if 'latitude' in clean and 'longitude' in clean:
                    header_row = i
                    if ';' in ligne: sep = ';'
                    elif ',' in ligne: sep = ','
                    break
            if header_row is not None:
                df = pd.read_csv(chemin, sep=sep, header=header_row, encoding=enc, engine='python')
                df.columns = [c.replace('\ufeff', '').strip() for c in df.columns]
                # Renommage standard
                for col in df.columns:
                    c_low = remove_accents(col.lower())
                    if 'point' in c_low or 'station' in c_low: df.rename(columns={col: 'Point'}, inplace=True)
                    if 'latitude' in c_low: df.rename(columns={col: 'Latitude'}, inplace=True)
                    if 'longitude' in c_low: df.rename(columns={col: 'Longitude'}, inplace=True)
                return df, None
        except: continue
    return None, "Erreur lecture"

# --- 2. AGRÉGATION TOTALE (CACHE) ---
@st.cache_data(ttl=3600, show_spinner=False)
def aggrerger_donnees_globales(dossier):
    """Lit TOUS les fichiers et fusionne tout dans un seul DataFrame."""
    full_df = pd.DataFrame()
    if not os.path.exists(dossier): return None, "Dossier introuvable"
    
    fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
    barre = st.progress(0, text="Fusion des scénarios...")
    
    for i, fichier in enumerate(fichiers):
        barre.progress((i)/len(fichiers))
        df, err = trouver_header_et_lire(os.path.join(dossier, fichier))
        if err: continue
        
        # Ajout du scénario basé sur le nom
        df['Scenario'] = extraire_scenario_du_nom(fichier)
        
        # Gestion Période
        col_p = next((c for c in df.columns if 'eriode' in remove_accents(c.lower()) or 'horizon' in remove_accents(c.lower())), None)
        df['Période'] = df[col_p].astype(str).str.strip() if col_p else "Inconnue"

        # Nettoyage types numériques
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        
        # Conversion auto des variables météo (ATXHWD, etc.)
        for col in df.columns:
            if col not in ['Latitude', 'Longitude', 'Point', 'Période', 'Scenario']:
                if df[col].dtype == object:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

        full_df = pd.concat([full_df, df], ignore_index=True)
    
    barre.empty()
    return full_df.dropna(subset=['Latitude', 'Longitude']), None

# --- 3. GÉOCODAGE ROBUSTE ---
def geocode_safe(address):
    """Nominatim avec User-Agent unique et gestion d'erreur."""
    try:
        # On change l'agent à chaque appel pour éviter le blocage IP
        agent = f"user_{uuid.uuid4().hex[:8]}"
        geolocator = Nominatim(user_agent=agent, timeout=10)
        location = geolocator.geocode(address)
        return (location.latitude, location.longitude) if location else (None, None)
    except:
        return None, None

# --- 4. INTERFACE ---
with st.sidebar:
    st.header("⚙️ Configuration")
    master_df, err = aggrerger_donnees_globales(DOSSIER_DONNEES)
    
    if err or master_df is None or master_df.empty:
        st.error(err or "Aucune donnée trouvée.")
        st.stop()

    # Filtres
    choix_rcp = st.selectbox("1. Choisir le Scénario", sorted(master_df['Scenario'].unique()))
    df_filtered = master_df[master_df['Scenario'] == choix_rcp]
    
    choix_horizon = st.selectbox("2. Choisir la Période", sorted(df_filtered['Période'].unique()))
    df_final = df_filtered[df_filtered['Période'] == choix_horizon].copy()

    # Liste des variables (on exclut les colonnes de texte/meta)
    exclues = ['Latitude', 'Longitude', 'Point', 'Période', 'Scenario']
    vars_dispos = [c for c in df_final.columns if c not in exclues and pd.api.types.is_numeric_dtype(df_final[c])]
    choix_var = st.selectbox("3. Variable à cartographier", vars_dispos)

    # Légende
    st.divider()
    vmin, vmax = master_df[choix_var].min(), master_df[choix_var].max()
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(4, 0.4))
    ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect='auto', cmap=cmap)
    ax.set_axis_off()
    st.write(f"Échelle : {vmin:.1f} à {vmax:.1f}")
    st.pyplot(fig)

# --- 5. RECHERCHE & CARTE ---
adr = st.text_input("🔍 Rechercher une adresse (ex: Lyon, Marseille...)", "")
u_lat, u_lon = None, None

if adr:
    u_lat, u_lon = geocode_safe(adr)
    if u_lat:
        st.success(f"📍 Localisé : {u_lat:.3f}, {u_lon:.3f}")
    else:
        st.warning("⚠️ Adresse introuvable. Vérifiez l'orthographe ou précisez la ville.")

# Couleurs pour la carte
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
colors = (cmap(norm(df_final[choix_var].values))[:, :3] * 255).astype(int)
df_final['r'], df_final['g'], df_final['b'] = colors[:, 0], colors[:, 1], colors[:, 2]

# PyDeck
layers = [
    pdk.Layer(
        "ScatterplotLayer", data=df_final, get_position='[Longitude, Latitude]',
        get_color='[r, g, b, 160]', get_radius=3000, pickable=True, auto_highlight=True
    )
]

if u_lat:
    layers.append(pdk.Layer(
        "ScatterplotLayer", data=pd.DataFrame({'lat': [u_lat], 'lon': [u_lon]}),
        get_position='[lon, lat]', get_color='[0, 255, 0]', get_radius=6000, stroked=True
    ))
    v_state = pdk.ViewState(latitude=u_lat, longitude=u_lon, zoom=9)
else:
    v_state = pdk.ViewState(latitude=46.6, longitude=1.8, zoom=5.5)

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=v_state, tooltip={"html": f"<b>{choix_var}:</b> {{{choix_var}}}"}))

# --- 6. TABLEAUX ---
if u_lat:
    st.divider()
    # Calcul des 5 voisins
    df_final['dist'] = df_final.apply(lambda r: geodesic((u_lat, u_lon), (r['Latitude'], r['Longitude'])).km, axis=1)
    voisins = df_final.nsmallest(5, 'dist')
    
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"Station la plus proche : {voisins.iloc[0]['Point']}")
        # Affichage de toutes les variables du point
        st.dataframe(voisins.iloc[0:1][vars_dispos].T.rename(columns={voisins.index[0]: 'Valeur Real'}))
    
    with c2:
        st.success("Valeurs Estimées (Interpolation)")
        weights = 1 / (voisins['dist'] + 0.001)**2
        interp = (voisins[vars_dispos].T * weights).sum(axis=1) / weights.sum()
        st.dataframe(pd.DataFrame(interp, columns=['Valeur Estimée']))
