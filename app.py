import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# =====================================
# CONFIG
# =====================================

st.set_page_config(layout="wide", page_title="Climat Multi-Scénarios")
st.title("🌍 Analyse climatique")

DOSSIER = "Données"

# =====================================
# LECTURE DES FICHIERS
# =====================================

def lire_fichier(path):
    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        comment="#",
        skip_blank_lines=True
    )

@st.cache_data
def charger_donnees(dossier):
    all_df = []

    for f in os.listdir(dossier):
        if not f.endswith(".txt"):
            continue

        df = lire_fichier(os.path.join(dossier, f))

        # normalisation des noms
        df.columns = [c.strip() for c in df.columns]

        # renommage standard
        for c in df.columns:
            c2 = c.lower()
            if "lat" in c2: df.rename(columns={c:"Latitude"}, inplace=True)
            if "lon" in c2: df.rename(columns={c:"Longitude"}, inplace=True)
            if "station" in c2 or "point" in c2: df.rename(columns={c:"Point"}, inplace=True)

        # conversion numérique
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

        for c in df.columns:
            if c not in ["Latitude","Longitude","Point","Contexte","Période"]:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")

        all_df.append(df)

    return pd.concat(all_df, ignore_index=True)

# =====================================
# DONNÉES
# =====================================

data = charger_donnees(DOSSIER)

# =====================================
# FILTRES
# =====================================

with st.sidebar:
    st.header("Filtres")

    scenario = st.selectbox("Scénario", sorted(data["Contexte"].dropna().unique()))
    df1 = data[data["Contexte"] == scenario]

    horizon = st.selectbox("Horizon", sorted(df1["Période"].dropna().unique()))
    df2 = df1[df1["Période"] == horizon]

    meta = ["Latitude","Longitude","Point","Contexte","Période"]
    variables = [c for c in df2.columns if c not in meta and pd.api.types.is_numeric_dtype(df2[c])]

    var = st.selectbox("Variable", variables)

    # suppression des lignes sans valeur
    df2 = df2.dropna(subset=["Latitude","Longitude",var])

    vmin = df2[var].quantile(0.02)
    vmax = df2[var].quantile(0.98)

# =====================================
# GEOCODAGE
# =====================================

@st.cache_resource
def get_geocoder():
    return Nominatim(user_agent="climate_app")

geolocator = get_geocoder()

@st.cache_data
def geocode(q):
    loc = geolocator.geocode(q)
    if loc:
        return loc.latitude, loc.longitude
    return None,None

adr = st.text_input("🔍 Ville")
btn = st.button("Rechercher")

u_lat,u_lon = None,None
if btn and adr:
    u_lat,u_lon = geocode(adr)
    if u_lat:
        st.success(f"{adr} → {u_lat:.3f},{u_lon:.3f}")
    else:
        st.warning("Adresse introuvable")

# =====================================
# COULEURS
# =====================================

cmap = plt.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
rgb = (cmap(norm(df2[var]))[:,:3]*255).astype(int)

df2 = df2.copy()
df2["r"],df2["g"],df2["b"] = rgb[:,0],rgb[:,1],rgb[:,2]

# =====================================
# CARTE
# =====================================

layers = [pdk.Layer(
    "ScatterplotLayer",
    df2,
    get_position="[Longitude,Latitude]",
    get_color="[r,g,b,160]",
    get_radius=3000,
    pickable=True
)]

if u_lat:
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        pd.DataFrame({"lat":[u_lat],"lon":[u_lon]}),
        get_position="[lon,lat]",
        get_color="[0,255,0]",
        get_radius=6000
    ))

view = pdk.ViewState(
    latitude=u_lat if u_lat else 46.6,
    longitude=u_lon if u_lon else 2.0,
    zoom=9 if u_lat else 5
)

st.pydeck_chart(pdk.Deck(
    layers=layers,
    initial_view_state=view,
    tooltip={"html":f"<b>{var}</b>: {{{var}}}"}
))

# =====================================
# TABLEAUX CORRIGÉS
# =====================================

if u_lat:
    df2["dist"] = df2.apply(
        lambda r: geodesic((u_lat,u_lon),(r["Latitude"],r["Longitude"])).km,
        axis=1
    )

    voisins = df2.nsmallest(5,"dist")

    c1,c2 = st.columns(2)

    with c1:
        st.subheader("📍 Station la plus proche")
        station = voisins.iloc[0]
        st.dataframe(station[variables].to_frame("Valeur réelle"))

    with c2:
        st.subheader("🧮 Valeurs interpolées")

        W = 1/(voisins["dist"]+0.01)**2

        interp = {}
        for v in variables:
            vals = voisins[v].values
            mask = ~np.isnan(vals)
            interp[v] = np.sum(vals[mask]*W.values[mask]) / np.sum(W.values[mask]) if mask.sum()>0 else np.nan

        st.dataframe(pd.DataFrame(interp,index=["Estimation"]).T)
