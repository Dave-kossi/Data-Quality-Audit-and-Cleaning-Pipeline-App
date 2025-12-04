# app.py – DataCleaner Pro++ (LLM édition complète)
import streamlit as st, pandas as pd, numpy as np, io, uuid, shutil, os, platform, tempfile, logging, sys, requests, base64, json, csv
from pathlib import Path


# ---------------- CONFIG ---------------- #
TEMP_DIR    = Path(tempfile.gettempdir()) / "datacleaner"
TEMP_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     handlers=[logging.FileHandler(TEMP_DIR / "app.log"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("datacleaner")
MAX_SIZE    = 500 * 1_000_000  # 500 Mo

# ---------------- LLM – OPENROUTER ---------------- #
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL      = "meta-llama/llama-3.2-3b-instruct"

def ask_llama(prompt: str, max_tokens: 350) -> str | None:
    headers = {"Authorization": f"Bearer {st.secrets['OPENROUTER_KEY']}",
               "HTTP-Referer": "https://datacleaner-pro.streamlit.app",
               "X-Title": "DataCleaner-Pro"}
    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.2}
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("LLM error: %s", e)
        return None

# ---------------- CORE UTILS ---------------- #
def memory_opt(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    before = df.memory_usage(deep=True).sum()
    df = df.copy()
    for c in df.columns:
        col = df[c]
        if col.dtype == "object" and col.nunique() / len(col) < 0.5:
            df[c] = col.astype("category")
        elif pd.api.types.is_integer_dtype(col):
            for dtype in (np.int8, np.int16, np.int32, np.int64):
                if col.min() >= np.iinfo(dtype).min and col.max() <= np.iinfo(dtype).max:
                    df[c] = col.astype(dtype); break
        elif pd.api.types.is_float_dtype(col):
            df[c] = pd.to_numeric(col, downcast="float")
    reduction = (before - df.memory_usage(deep=True).sum()) / before * 100
    return df, reduction

def clean_df(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, list[str]]:
    logs, df = [], df.copy()
    
    # 1. Suppression des doublons
    if params["drop_dup"]:
        prev = len(df)
        df = df.drop_duplicates()
        if (d := prev - len(df)):
            logs.append(f"🗑️ {d} doublons supprimés")
            
    # 2. Normalisation des noms de colonnes
    if params["norm_cols"]:
        old = df.columns.tolist()
        df.columns = df.columns.str.strip().str.lower().str.replace(r"\W+", "_", regex=True)
        if old != df.columns.tolist():
            logs.append("🔠 noms normalisés")
            
    # 3. Nettoyage du texte/catégories (NOUVEAU)
    if params["text_clean"]:
        for c in df.columns:
            if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c]):
                # Conversion en chaîne, suppression des espaces, mise en minuscule (pour uniformité)
                df[c] = df[c].astype(str).str.strip().str.lower()
                # Re-categorisation si le type était catégoriel
                if pd.api.types.is_categorical_dtype(df[c]):
                    df[c] = df[c].astype("category")
        logs.append(f"✍️ Nettoyage du texte (strip, minuscule) appliqué")

    # 4. Conversion automatique des types
    if params["auto_type"]:
        th = params["na_thresh"] / 100
        for c in df.columns:
            if df[c].dtype != "object": continue
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() >= th:
                df[c] = num; logs.append(f"🔢 {c} → num"); continue
            try:
                dat = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                if dat.notna().mean() >= th:
                    df[c] = dat; logs.append(f"📅 {c} → date")
            except: pass
            
    # 5. Gestion des NA
    if params["handle_na"]:
        for c in df.columns:
            if df[c].isna().sum() == 0: continue
            if pd.api.types.is_numeric_dtype(df[c]):
                fill = df[c].median() if params["na_strat"] == "median" else df[c].mode()[0]
                df[c] = df[c].fillna(fill)
                logs.append(f"📈 {c} NA → {params['na_strat']}")
            else:
                df[c] = df[c].fillna(params["na_fill"])
                logs.append(f"📝 {c} NA → '{params['na_fill']}'")
                
    # 6. Plafonnement des Outliers (MISE À JOUR)
    iqr_coeff = params["iqr_coeff"]
    if iqr_coeff > 0: # Le slider est > 0 si l'utilisateur veut appliquer le capping
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                Q1 = df[c].quantile(0.25)
                Q3 = df[c].quantile(0.75)
                IQR = Q3 - Q1
                
                # Utiliser le coefficient choisi par l'utilisateur
                lower_bound = Q1 - iqr_coeff * IQR
                upper_bound = Q3 + iqr_coeff * IQR
                
                # Appliquer le capping
                df[c] = df[c].clip(lower_bound, upper_bound)
                logs.append(f"📈 {c} valeurs aberrantes plafonnées (coeff {iqr_coeff:.1f} IQR)")
                
    # 7. Suppression des colonnes à faible variance
    if params["drop_low_var"]:
        cols_to_drop = []
        for c in df.columns:
            if df[c].nunique() <= 1:
                # Colonnes avec 0 ou 1 valeur unique
                cols_to_drop.append(c)
                logs.append(f"🗑️ {c} supprimé (unique <= 1)")
            elif df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c]):
                # Colonnes textuelles presque constantes (>99% même valeur)
                if not df[c].empty:
                    most_freq_ratio = df[c].value_counts(normalize=True).iloc[0]
                    if most_freq_ratio > 0.99: 
                        cols_to_drop.append(c)
                        logs.append(f"🗑️ {c} supprimé (texte >99% constant)")
            
        df = df.drop(columns=cols_to_drop, errors='ignore')

    # 8. Optimisation de la mémoire (toujours en dernier)
    df, gain = memory_opt(df)
    if gain > 5:
        logs.append(f"💾 mémoire -{gain:.1f}%")
        
    return df, logs

def build_report(df: pd.DataFrame, title: str, minimal: bool) -> Path | None:
    """Tente de générer le rapport ydata-profiling ou retourne None si échec."""
    try:
        # Importation dynamique pour éviter le plantage si la dépendance n'est pas là.
        from ydata_profiling import ProfileReport
        file = TEMP_DIR / f"report_{title.replace(' ','_')}_{uuid.uuid4().hex}.html"
        ProfileReport(df, title=title, minimal=minimal).to_file(str(file))
        return file
    except Exception as e:
        # Log l'erreur (probablement ModuleNotFoundError si ydata-profiling est manquante)
        log.warning("ydata fail: %s. Utilisation du rapport de secours.", e)
        return None

def fallback_report(df: pd.DataFrame, title: str):
    st.subheader(f"📋 Rapport basique – {title}")
    
    # Ajout d'une alerte visible si le rapport complet manque
    st.warning("Rapport complet manquant. Assurez-vous que 'ydata-profiling' est installé (via requirements.txt) et que la génération n'a pas échoué.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Types & uniques**")
        st.dataframe(df.dtypes.astype(str).to_frame("type").assign(uniques=df.nunique()))
    with c2:
        na = df.isna().sum().to_frame("NA").query("NA > 0")
        st.write("**Valeurs manquantes**")
        st.dataframe(na if not na.empty else "Aucune")
    st.write("**Aperçu**")
    st.dataframe(df.head(10))

def show_report(file: Path):
    b64 = base64.b64encode(file.read_bytes()).decode()
    html = f'<iframe src="data:text/html;base64,{b64}" width="100%" height="700" style="border:none;"></iframe>'
    st.components.v1.html(html, height=700)

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(page_title="Axiom", layout="wide")
st.title("✨ Axiom • Data Quality & Audit")
st.markdown("Audit & nettoyage **intelligent** ")
st.divider() # AJOUT 1: Séparateur visuel après le titre pour plus de clarté

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    params = {
        "drop_dup":  st.checkbox("Supprimer doublons", True),
        "norm_cols": st.checkbox("Normaliser colonnes", True),
        "auto_type": st.checkbox("Conversion auto types", True),
        "text_clean": st.checkbox("Nettoyage Texte (strip, minuscule)", True),
        "iqr_coeff": st.slider("Coefficient IQR (Outliers)", 0.0, 3.0, 1.5, 0.1), # Changement de checkbox à slider
        "drop_low_var": st.checkbox("Supprimer colonnes à faible variance", False),
        "handle_na": st.checkbox("Gérer NA", True),
        "na_thresh": st.slider("Taux min valeurs valides (%)", 50, 90, 70),
        "na_strat":  st.selectbox("Stratégie NA num", ["median", "mode"]),
        "na_fill":   st.text_input("Valeur remplacement NA texte", "_MISSING_"),
        "minimal":   st.checkbox("Mode ydata minimal (rapide)", True),
    }
    if st.button("🗑️ vider fichiers temp"):
        shutil.rmtree(TEMP_DIR, ignore_errors=True); TEMP_DIR.mkdir(exist_ok=True); st.success("Temp vidé")
    st.info(f"Système: {platform.system()} | Dossier: {TEMP_DIR}")

# ---------------- CHARGEMENT ---------------- #
uploaded = st.file_uploader("📂 Sélectionnez votre fichier",
                            type=["csv","xlsx","json","parquet","txt"], accept_multiple_files=False)
if not uploaded:
    st.info("⬆️ Chargez un fichier pour commencer l'analyse de qualité des données.", icon="💡") # AJOUT 2: Message d'attente pro
    st.stop()
if uploaded.size > MAX_SIZE:
    st.error("Fichier > 500 Mo refusé"); st.stop()

@st.cache_data(show_spinner=False)
def load(uploaded):
    try:
        ext = Path(uploaded.name).suffix.lower()
        buffer = io.BytesIO(uploaded.getbuffer())

        # --- Gestion des .csv et .txt ---
        if ext in [".csv", ".txt"]:
            try:
                # 1️⃣ Tentative de lecture tabulaire
                df = pd.read_csv(
                    buffer,
                    encoding="utf-8",
                    sep=None,              # détection automatique du séparateur
                    engine="python",
                    quoting=csv.QUOTE_MINIMAL,
                    on_bad_lines="skip"
                )
                return df
            except Exception:
                # 2️⃣ Tentative JSON Lines (si le txt contient du JSON)
                buffer.seek(0)
                try:
                    df = pd.read_json(buffer, lines=True)
                    return df
                except Exception:
                    # 3️⃣ Lecture texte brut → DataFrame à une colonne
                    buffer.seek(0)
                    content = buffer.read().decode("utf-8", errors="ignore")
                    return pd.DataFrame({"texte": [content]})

        # --- Autres formats classiques ---
        elif ext == ".xlsx":
            return pd.read_excel(buffer)
        elif ext == ".json":
            return pd.read_json(buffer)
        elif ext == ".parquet":
            return pd.read_parquet(buffer)

    except Exception as e:
        log.exception("Erreur de chargement du fichier")
        st.error(str(e))
    return None


df_raw = load(uploaded)
if df_raw is None:
    st.stop()

df_opt, gain = memory_opt(df_raw)

# DEBUT AMELIORATION VISUELLE: Aperçu des métriques clés
st.subheader("Aperçu du Dataset Brut")
st.markdown("Les métriques ci-dessous sont calculées sur le fichier brut avant nettoyage.")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Lignes", f"{len(df_raw):,}")
col2.metric("Total Colonnes", len(df_raw.columns))
# Utilisation de Delta pour montrer l'impact des NA
total_cells = len(df_raw) * len(df_raw.columns)
na_count = df_raw.isna().sum().sum()
na_percentage = na_count * 100 / total_cells if total_cells > 0 else 0

col3.metric("NA", na_count, f"{na_percentage:.2f}% des cellules", delta_color="inverse")
col4.metric("Optimisation RAM", f"{gain:.1f}%", "Réduction potentielle", delta_color="normal")
st.divider()
# FIN AMELIORATION VISUELLE

if st.button("🚀 Lancer l’analyse complète", type="primary"):
    bar = st.progress(0)
    bar.progress(10)
    report_before = build_report(df_opt, "Avant", params["minimal"])
    bar.progress(40)
    df_clean, logs = clean_df(df_opt, params)
    bar.progress(70)
    report_after = build_report(df_clean, "Après", params["minimal"])
    bar.progress(100)
    st.session_state["before"] = df_opt
    st.session_state["after"]  = df_clean
    st.session_state["rep_bef"] = report_before
    st.session_state["rep_aft"] = report_after
    st.session_state["logs"]    = logs
    st.rerun()

if "after" not in st.session_state:
    st.stop()

bef, aft = st.session_state["before"], st.session_state["after"]
tab1, tab2, tab3, tab4 = st.tabs(["Avant", "🧹 Nettoyage", "📈 Après + Export", "🤖 LLM"])

with tab1:
    st.subheader("Rapport avant nettoyage")
    if (p := st.session_state["rep_bef"]):
        st.download_button("📥 Télécharger HTML", data=p.read_bytes(), file_name=p.name, mime="text/html")
        show_report(p)
    else:
        fallback_report(bef, "Avant")

with tab2:
    st.success(f"{len(st.session_state['logs'])} transformations")
    for l in st.session_state["logs"]:
        st.write("•", l)
    st.dataframe(aft.head(10))

with tab3:
    st.subheader("Rapport après nettoyage")
    if (p := st.session_state["rep_aft"]):
        st.download_button("📥 Télécharger HTML", data=p.read_bytes(), file_name=p.name, mime="text/html")
        show_report(p)
    else:
        fallback_report(aft, "Après")
    fmt = st.selectbox("Exporter dataset nettoyé", ["csv", "xlsx", "json", "parquet"])
    buf = io.BytesIO()
    match fmt:
        case "csv":      aft.to_csv(buf, index=False)
        case "xlsx":     aft.to_excel(buf, index=False)
        case "json":     buf.write(aft.to_json(orient="records", indent=2).encode())
        case "parquet":  aft.to_parquet(buf)
    buf.seek(0)
    st.download_button("💾 Télécharger dataset", data=buf, file_name=f"clean.{fmt}")

# ---------------- RECOS & CHAT INTELLIGENTS ---------------- #
with tab4:
    st.header("🤖 Assistant IA – Recommandations & Chat")

    # Historique conversation
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Contexte dataset réduit
    def build_context():
        schema = aft.dtypes.astype(str).to_frame("type").assign(uniques=aft.nunique(), NA=aft.isna().sum())
        sample = aft.head(5).to_dict(orient="records")
        
        # --- NOUVEAU: Statistiques Clés pour Colonnes Numériques ---
        numeric_cols = aft.select_dtypes(include=np.number).columns
        key_stats = {}
        if not numeric_cols.empty:
            stats_df = aft[numeric_cols].agg(['min', 'max', 'mean', 'std']).T
            # Arrondir pour un contexte plus propre
            stats_df = stats_df.apply(lambda x: round(x, 2))
            key_stats = stats_df.to_dict(orient='index')
        # --- FIN NOUVEAU ---

        context = f"Schema colonnes :\n{schema.to_string()}\n"
        
        if key_stats:
             context += f"\nStatistiques Clés (Numériques) :\n{json.dumps(key_stats, indent=2, ensure_ascii=False)}\n"
             
        context += f"\n5 premières lignes :\n{json.dumps(sample, ensure_ascii=False, indent=2)}"
        return context

    # --- Boutons rapides ---
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Résumé qualité"):
            prompt = f"{build_context()}\n\nDonne un résumé global (4 phrases) : qualité, anomalies, conseils."
            answer = ask_llama(prompt, 350)
            st.session_state.chat.append(("bot", answer or "Hors-ligne."))
    with col2:
        if st.button("🔍 Règles de validation"):
            col = aft.columns[0]
            prompt = (f"{build_context()}\n"
                      f"En te basant sur le contexte ci-dessus, propose 3 règles de validation métier pour la colonne '{col}' (format : règle + raison).")
            answer = ask_llama(prompt, 350)
            st.session_state.chat.append(("bot", answer or "Hors-ligne."))
    with col3:
        if st.button("💡 Recommandations métier"):
            prompt = f"{build_context()}\n\nImagine 3 actions concrètes (métier) pour améliorer ce dataset."
            answer = ask_llama(prompt, 350)
            st.session_state.chat.append(("bot", answer or "Hors-ligne."))

    st.markdown("---")

    # --- Chat libre ---
    user_msg = st.text_input("💬 Posez une question libre :", placeholder="Ex. : Quelles colonnes ont le plus d'impact sur le target ?")
    if st.button("📤 Envoyer"):
        if not user_msg.strip():
            st.warning("Message vide.")
        else:
            with st.spinner("LLM réfléchit..."):
                context = build_context()
                prompt = (f"{context}\n\nQuestion utilisateur : {user_msg}\nRéponse concise (max 5 phrases) :")
                answer = ask_llama(prompt, 400)
                if not answer:
                    answer = "Désolé, le service LLM est hors-ligne."
                st.session_state.chat.append(("user", user_msg))
                st.session_state.chat.append(("bot", answer))

    # --- Affichage conversation ---
    st.markdown("---")
    for author, msg in st.session_state.chat:
        if author == "user":
            st.markdown(f'<div style="text-align:right; color:#0d47a1;"><b>Moi :</b> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="text-align:left; color:#388e3c;"><b>IA :</b> {msg}</div>', unsafe_allow_html=True)

    # --- Clear chat ---
    if st.button("🗑️ Effacer la conversation"):
        st.session_state.chat = []
        st.rerun()
