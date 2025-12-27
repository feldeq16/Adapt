import streamlit as st
import pandas as pd
import pydeck as pdk
import os

# 1. Configuration ultra-simple
st.set_page_config(page_title="Test Diagnostic")

# 2. Si ce texte s'affiche, c'est que Streamlit fonctionne
st.title("🛠️ Mode Diagnostic")
st.write("Si vous lisez ceci, l'application a démarré correctement.")

DOSSIER_DONNEES = 'Données'

# 3. Vérification du dossier
if not os.path.exists(DOSSIER_DONNEES):
    st.error(f"Le dossier '{DOSSIER_DONNEES}' n'est pas trouvé sur GitHub.")
    st.stop() # On arrête tout ici si pas de dossier

fichiers = [f for f in os.listdir(DOSSIER_DONNEES) if f.endswith('.txt')]
st.write(f"Fichiers détectés : {len(fichiers)}")

# --- SECTION DE CHARGEMENT MANUEL ---
st.divider()
st.write("Pour éviter le crash, nous allons charger seulement 50 lignes par fichier.")

# Bouton pour lancer le chargement (évite le chargement automatique qui plante)
if st.button("Lancer le chargement test"):
    
    all_data = []
    
    # Barre de progression
    barre = st.progress(0)
    
    for i, fichier in enumerate(fichiers):
        st.write(f"Lecture de {fichier}...")
        chemin = os.path.join(DOSSIER_DONNEES, fichier)
        
        try:
            # ON CHARGE SEULEMENT 50 LIGNES (nrows=50)
            df = pd.read_csv(
                chemin, 
                sep=';', 
                comment='#', 
                encoding='latin-1',
                engine='python',
                nrows=50 
            )
            
            # Nettoyage express
            df.columns = [c.strip() for c in df.columns]
            
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                # Conversion propre
                df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
                df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
                df = df.dropna(subset=['Latitude', 'Longitude'])
                
                # Ajout de couleur rouge par défaut
                df['color_r'] = 255
                df['color_g'] = 0
                df['color_b'] = 0
                
                all_data.append(df)
                st.success(f"✅ {fichier} : OK ({len(df)} lignes)")
            else:
                st.warning(f"⚠️ {fichier} : Colonnes Lat/Lon manquantes.")
                
        except Exception as e:
            st.error(f"❌ Erreur sur {fichier} : {e}")
        
        barre.progress((i + 1) / len(fichiers))
            
    # --- AFFICHAGE CARTE ---
    if all_data:
        df_final = pd.concat(all_data)
        st.write(f"Total points chargés : {len(df_final)}")
        
        # Carte PyDeck simple
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=46.6,
                longitude=1.8,
                zoom=5
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=df_final,
                    get_position='[Longitude, Latitude]',
                    get_color='[color_r, color_g, color_b, 160]',
                    get_radius=10000,
                ),
            ],
        ))
    else:
        st.error("Aucune donnée valide récupérée.")
