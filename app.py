import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import requests
from geopy.distance import geodesic

# ============================================
# CONFIG
# ============================================

st.set_page_config(layout="wide", page_title="Climat Multi-Scénarios")
st.title("🌍 Analyse climatique multi-scénarios")

DOSSIER = "Données"

# ============================================
# LECTURE DES FICHIERS
# ============================================

def lire_fichier(path):
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        comment="#",
        skip_blank_lines=True
    )

@st.cache_data
def charger_donnees_1(dossier):
    all_df = []

    for f in os.listdir(dossier):
        if not f.endswith(".txt"):
            continue

        df = lire_fichier(os.path.join(dossier, f))
        df.columns = [c.strip() for c in df.columns]

        # standardisation des colonnes
        for c in df.columns:
            c2 = c.lower()
            
        # conversion numérique
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

        for c in df.columns:
            if c not in ["Latitude", "Longitude", "Point", "Contexte", "Période"]:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

        all_df.append(df)

    return pd.concat(all_df, ignore_index=True)

@st.cache_data
def charger_donnees(dossier):
    """
    Agrège tous les fichiers dans une table unique.
    L'identifiant unique est ['Point','Contexte','Période'].
    Toutes les colonnes mesures sont fusionnées sans duplication.
    """
    final_df = None  # DataFrame final
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    for f in os.listdir(dossier):
        if not f.endswith(".txt"):
            continue

        df = lire_fichier(os.path.join(dossier, f))
        df.columns = [c.strip() for c in df.columns]

        # Conversion numérique pour les colonnes de mesures
        for c in df.columns:
            if c in latlon_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            elif c not in id_cols:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

        # Si final_df est vide, on initialise avec le premier DataFrame
        if final_df is None:
            final_df = df
        else:
            # On garde uniquement les colonnes nouvelles à fusionner
            new_columns = [col for col in df.columns if col not in final_df.columns and col not in id_cols + latlon_cols]
            
            # Si de nouvelles colonnes existent, on fait la fusion
            if new_columns:
                final_df = pd.merge(final_df, df[id_cols + latlon_cols + new_columns], 
                                     on=id_cols + latlon_cols, 
                                     how="outer")

    # Retourner le DataFrame final avec toutes les variables
    return final_df


# ============================================
# DONNÉES
# ============================================

data = charger_donnees(DOSSIER)
st.dataframe(data)

# ============================================
# FILTRES
# ============================================

with st.sidebar:
    st.header("Filtres")

    scenario = st.selectbox("Scénario", sorted(data["Contexte"].dropna().unique()))
    df1 = data[data["Contexte"] == scenario]

    horizon = st.selectbox("Horizon", sorted(df1["Période"].dropna().unique()))
    df2 = df1[df1["Période"] == horizon]

    meta = ["Latitude", "Longitude", "Point", "Contexte", "Période"]
    variables = [c for c in df2.columns if c not in meta and pd.api.types.is_numeric_dtype(df2[c])]

    var = st.selectbox("Variable", variables)

    df2 = df2.dropna(subset=["Latitude", "Longitude", var])

    vmin = df2[var].quantile(0.02)
    vmax = df2[var].quantile(0.98)

# ============================================
# GÉOCODAGE (OPENCAGE)
# ============================================

@st.cache_data(show_spinner=False)
def geocode(address):
    key = st.secrets["OPENCAGE_KEY"]

    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {
        "q": address,
        "key": key,
        "limit": 1,
        "no_annotations": 1,
        "language": "fr"
    }

    try:
        r = requests.get(url, params=params, timeout=5)
        data = r.json()
        if data["results"]:
            g = data["results"][0]["geometry"]
            return g["lat"], g["lng"]
    except:
        pass

    return None, None

adr = st.text_input("🔍 Adresse", placeholder="10 rue de Rivoli, Paris")
search = st.button("Rechercher")

u_lat, u_lon = None, None
if search and adr:
    with st.spinner("Localisation..."):
        u_lat, u_lon = geocode(adr)

    if u_lat:
        st.success(f"📍 {adr} → {u_lat:.4f}, {u_lon:.4f}")
    else:
        st.warning("Adresse introuvable")

# ============================================
# COULEURS
# ============================================

cmap = plt.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
rgb = (cmap(norm(df2[var]))[:, :3] * 255).astype(int)

df2 = df2.copy()
df2["r"], df2["g"], df2["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

# ============================================
# CARTE
# ============================================

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        df2,
        get_position="[Longitude, Latitude]",
        get_color="[r, g, b, 160]",
        get_radius=3000,
        pickable=True,
        auto_highlight=True,
    )
]

if u_lat:
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            pd.DataFrame({"lat": [u_lat], "lon": [u_lon]}),
            get_position="[lon, lat]",
            get_color="[0, 255, 0]",
            get_radius=6000,
        )
    )

view = pdk.ViewState(
    latitude=u_lat if u_lat else 46.6,
    longitude=u_lon if u_lon else 2.0,
    zoom=9 if u_lat else 5,
)

st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=view,
        tooltip={"html": f"<b>{var}</b>: {{{var}}}"},
    )
)

# ============================================
# TABLEAUX + INTERPOLATION
# ============================================
st.dataframe(df2)
if u_lat:
    df2["dist"] = df2.apply(
        lambda r: geodesic((u_lat, u_lon), (r["Latitude"], r["Longitude"])).km, axis=1
    )

    voisins = df2.nsmallest(5, "dist")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📍 Station la plus proche")
        station = voisins.iloc[0]
        st.dataframe(station[variables].to_frame("Valeur réelle"))

    with c2:
        st.subheader("🧮 Valeurs interpolées")

        W = 1 / (voisins["dist"] + 0.01) ** 2

        interp = {}
        for v in variables:
            vals = voisins[v].values
            mask = ~np.isnan(vals)
            interp[v] = (
                np.sum(vals[mask] * W.values[mask]) / np.sum(W.values[mask])
                if mask.sum() > 0
                else np.nan
            )

        st.dataframe(pd.DataFrame(interp, index=["Estimation"]).T)
