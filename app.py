# ... (Les imports restent les mêmes) ...

# ============================================
# 2. CHARGEMENT ET TRAITEMENT (CORRIGÉ)
# ============================================

def lire_metadonnees_et_data(path):
    description_map = {}
    try:
        # 1. Lecture des commentaires
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    if ":" in line:
                        parts = line.replace("#", "").split(":", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val = parts[1].strip()
                            description_map[key] = val
                else:
                    break
        
        # 2. Lecture des données
        df = pd.read_csv(path, sep=None, engine="python", comment="#", skip_blank_lines=True, encoding='latin-1')
        return df, description_map
    except Exception as e:
        return None, {}

def detecter_scenario(filename):
    """Devine le scénario si la colonne est absente"""
    n = filename.lower()
    if "rcp2.6" in n or "rcp26" in n: return "RCP 2.6"
    if "rcp4.5" in n or "rcp45" in n: return "RCP 4.5"
    if "rcp8.5" in n or "rcp85" in n: return "RCP 8.5"
    return "Scénario Inconnu"

@st.cache_data(show_spinner=False)
def charger_donnees_globales(dossier):
    if not os.path.exists(dossier):
        return None, None, {}

    all_dfs = []
    global_descriptions = {}
    
    # Les colonnes cibles qu'on veut OBLIGATOIREMENT à la fin
    id_cols = ["Point", "Contexte", "Période"]
    latlon_cols = ["Latitude", "Longitude"]

    for f in os.listdir(dossier):
        if not f.endswith(".txt"): continue
        
        df, metas = lire_metadonnees_et_data(os.path.join(dossier, f))
        if df is None: continue
        
        global_descriptions.update(metas)

        # 1. Nettoyage des colonnes (espaces, unnamed)
        df = df.drop(columns=[c for c in df.columns if "Unnamed" in c])
        df.columns = [c.strip() for c in df.columns]

        # 2. RENOMMAGE INTELLIGENT (C'est ici que ça corrige le KeyError)
        rename_map = {}
        for c in df.columns:
            clow = c.lower()
            # On cherche les synonymes
            if 'lat' in clow: rename_map[c] = 'Latitude'
            elif 'lon' in clow: rename_map[c] = 'Longitude'
            elif 'point' in clow or 'station' in clow: rename_map[c] = 'Point'
            elif 'period' in clow or 'horizon' in clow: rename_map[c] = 'Période'
            elif 'context' in clow or 'scenar' in clow or 'rcp' in clow: rename_map[c] = 'Contexte'
        
        df = df.rename(columns=rename_map)

        # 3. Remplissage des manquants (si le renommage n'a pas suffi)
        if "Contexte" not in df.columns:
            df["Contexte"] = detecter_scenario(f)
        
        if "Période" not in df.columns:
            # Si pas de colonne période, on met une valeur par défaut
            df["Période"] = "Horizon Global"

        # 4. Conversion numérique
        for c in df.columns:
            if c in latlon_cols:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
            elif c not in id_cols:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "."), errors="coerce")
        
        # Vérification finale : on ne garde que si on a les clés vitales
        if "Latitude" in df.columns and "Longitude" in df.columns:
            all_dfs.append(df)

    if not all_dfs: return None, None, {}

    # 5. Agrégation
    combined = pd.concat(all_dfs, ignore_index=True)
    
    # On ne fait la moyenne que sur les colonnes qui existent dans combined
    # Et on s'assure que id_cols sont bien présents grâce aux étapes 2 et 3
    agg_dict = {c: "first" for c in combined.columns if c not in id_cols}
    
    final_df = combined.groupby(id_cols, as_index=False).agg(agg_dict)

    # 6. Échelles Globales
    numeric_vars = [c for c in final_df.columns if c not in id_cols + latlon_cols and pd.api.types.is_numeric_dtype(final_df[c])]
    
    global_scales = {}
    for v in numeric_vars:
        vmin = final_df[v].min()
        vmax = final_df[v].max()
        global_scales[v] = (vmin, vmax)

    return final_df, global_scales, global_descriptions
