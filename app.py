import streamlit as st
import pandas as pd
import pydeck as pdk
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import unicodedata
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# ============================================
# CONFIG
# ============================================

st.set_page_config(layout="wide", page_title="Portail Climatique Agrégé")
st.title("🌍 Analyse Multi-Scénarios Climatiques")

DOSSIER_DONNEES = "Données"

# ============================================
# UTILS
# ============================================

def remove_accents(txt):
    return "".join(c for c in unicodedata.normalize("NFKD", str(txt)) if not unicodedata.combining(c))

def extraire_scenario(f):
    f = f.lower()
    if "2.6" in f or "26" in f: return "RCP 2.6"
    if "4.5" in f or "45" in f: return "RCP 4.5"
    if "8.5" in f or "85" in f: return "RCP 8.5"
    return "Autre"

# ============================================
# FICHIER ROBUSTE
# ============================================

def lire_fichier(path):
    encs = ["utf-8", "latin1", "cp1252"]
    for enc in encs:
        try:
            with open(path, encoding=enc) as f:
                rows = [f.readline() for _ in range(50)]
            header = None
            sep = ";"
            for i, r in enumerate(rows):
                r2 = remove_accents(r.lower())
                if "latitude" in r2 and "longitude" in r2:
                    header = i
                    if "," in r: sep = ","
                    break
            if header is None:
                return None

            df = pd.read_csv(path, sep=sep, header=header, encoding=enc)
            df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

            # normalisation
            for c in df.columns:
                c2 = remove_accents(c.lower())
                if "lat" in c2: df.rename(columns={c:"Latitude"}, inplace=True)
                if "lon" in c2: df.rename(columns={c:"Longitude"}, inplace=True)
                if "station" in c2 or "point" in c2: df.rename(columns={c:"Point"}, inplace=True)

            return df
        except:
            pass
    return None

# ============================================
# AGREGATION
# ============================================

@st.cache_data(ttl=3600)
def charger_donnees(folder):
    all_df = []
    for f in os.listdir(folder):
        if not f.endswith(".txt"): continue
        df = lire_fichier(os.path.join(folder,f))
        if df is None: continue

        df["Scenario"] = extraire_scenario(f)

        # période
        pcol = next((c for c in df.columns if "eriode" in remove_accents(c.lower()) or "horizon" in remove_accents(c.lower())),None)
        df["Période"] = df[pcol].astype(str) if pcol else "Inconnue"

        # conversion lat/lon
        df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
        df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

        # conversion météo
        meta = ["Latitude","Longitude","Point","Scenario","Période"]
        for c in df.columns:
            if c not in meta:
                s = df[c].astype(str).str.replace(",",".")
                if pd.to_numeric(s, errors="coerce").notna().mean() > 0.6:
                    df[c] = pd.to_numeric(s, errors="coerce")
                else:
                    df[c] = np.nan

        all_df.append(df)

    return pd.concat(all_df, ignore_index=True)

# ============================================
# GEOCODER RAPIDE
# ============================================

@st.cache_resource
def get_geocoder():
    return Nominatim(user_agent="climate_streamlit_app", timeout=5)

geolocator = get_geocoder()

@st.cache_data
def geocode(q):
    villes = {
        "paris":(48.8566,2.3522),"lyon":(45.764,4.8357),"marseille":(43.2965,5.3698),
        "toulouse":(43.6045,1.444),"bordeaux":(44.8378,-0.5792),"lille":(50.6292,3.0573),
        "nantes":(47.2184,-1.5536)
    }
    if q in villes:
        return villes[q]
    try:
        loc = geolocator.geocode(q, exactly_one=True, language="fr")
        if loc:
            return loc.latitude, loc.longitude
    except:
        pass
    return None,None

# ============================================
# INTERFACE
# ============================================

master = charger_donnees(DOSSIER_DONNEES)

with st.sidebar:
    rcp = st.selectbox("Scénario", sorted(master["Scenario"].unique()))
    df1 = master[master["Scenario"]==rcp]

    per = st.selectbox("Période", sorted(df1["Période"].unique()))
    df2 = df1[df1["Période"]==per]

    meta = ["Latitude","Longitude","Point","Scenario","Période"]
    vars = [c for c in df2.columns if c not in meta and pd.api.types.is_numeric_dtype(df2[c])]

    var = st.selectbox("Variable", vars)

    df2 = df2.dropna(subset=["Latitude","Longitude",var])

    vmin = df2[var].quantile(0.02)
    vmax = df2[var].quantile(0.98)

# ============================================
# RECHERCHE
# ============================================

adr = st.text_input("🔍 Ville ou adresse")
search = st.button("Rechercher")

u_lat,u_lon = None,None
if search and adr:
    u_lat,u_lon = geocode(adr.lower().strip())
    if u_lat:
        st.success(f"{adr} → {u_lat:.3f},{u_lon:.3f}")
    else:
        st.warning("Adresse introuvable")

# ============================================
# COULEURS
# ============================================

cmap = plt.get_cmap("coolwarm")
norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
rgb = (cmap(norm(df2[var]))[:,:3]*255).astype(int)
df2["r"],df2["g"],df2["b"] = rgb[:,0],rgb[:,1],rgb[:,2]

# ============================================
# CARTE
# ============================================

layers = [pdk.Layer("ScatterplotLayer", df2,
    get_position="[Longitude,Latitude]",
    get_color="[r,g,b,160]",
    get_radius=3000,
    pickable=True)]

if u_lat:
    layers.append(pdk.Layer("ScatterplotLayer",
        pd.DataFrame({"lat":[u_lat],"lon":[u_lon]}),
        get_position="[lon,lat]",
        get_color="[0,255,0]",
        get_radius=6000))

view = pdk.ViewState(latitude=u_lat if u_lat else 46.6, longitude=u_lon if u_lon else 2.0, zoom=9 if u_lat else 5)

st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view,
    tooltip={"html":f"<b>{var}</b>: {{{var}}}"}))

# ============================================
# INTERPOLATION
# ============================================

if u_lat:
    df2["dist"] = df2.apply(lambda r: geodesic((u_lat,u_lon),(r["Latitude"],r["Longitude"])).km,axis=1)
    voisins = df2.nsmallest(5,"dist")

    W = 1/(voisins["dist"]+0.01)**2

    interp = {}
    for v in vars:
        vals = voisins[v].values
        mask = ~np.isnan(vals)
        interp[v] = np.sum(vals[mask]*W.values[mask]) / np.sum(W.values[mask]) if mask.sum()>0 else np.nan

    c1,c2 = st.columns(2)
    with c1:
        st.write("Station la plus proche")
        st.dataframe(voisins.iloc[:1][vars].T)
    with c2:
        st.write("Valeurs interpolées")
        st.dataframe(pd.DataFrame(interp,index=["Estimation"]).T)
