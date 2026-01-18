# app.py – DataCleaner Pro++ (LLM édition complète)
import streamlit as st, pandas as pd, numpy as np, io, uuid, shutil, os, platform, tempfile, logging, sys, requests, base64, json, csv, xlrd, tabulate
from pathlib import Path


# ---------------- CONFIG ---------------- #
TEMP_DIR    = Path(tempfile.gettempdir()) / "datacleaner"
TEMP_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                     handlers=[logging.FileHandler(TEMP_DIR / "app.log"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("datacleaner")
MAX_SIZE    = 500 * 1000000  # 500 Mo

# ---------------- LLM – OPENROUTER ---------------- #
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL      = "meta-llama/llama-3.2-3b-instruct"

# Remplacer votre fonction ask_llama existante par :

def ask_llama(prompt: str, max_tokens: int = 350, temperature: float = 0.2, 
              profile: dict = None) -> str | None:
    """Fonction LLM améliorée avec adaptation contextuelle"""
    
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENROUTER_KEY']}",
        "HTTP-Referer": "https://datacleaner-pro.streamlit.app",
        "X-Title": "DataCleaner-Pro"
    }
    
    # System prompt adaptatif basé sur le profil
    if profile:
        if profile["type"] == "numerical":
            system_msg = """Tu es un Data Scientist expert en analyse numérique. 
            Tu réponds de manière technique mais accessible. Utilise des concepts statistiques.
            Sois précis avec les chiffres."""
        elif profile["type"] == "categorical":
            system_msg = """Tu es un expert en analyse catégorielle.
            Concentre-toi sur les fréquences, proportions et relations entre catégories.
            Propose des encodages adaptés."""
        elif profile["type"] == "temporal":
            system_msg = """Tu es un expert en séries temporelles.
            Analyse les tendances, saisonnalités et patterns temporels.
            Suggère des métriques temporelles pertinentes."""
        else:
            system_msg = """Tu es un expert en data quality et nettoyage de données.
            Tu réponds de manière concise, technique et factuelle."""
    else:
        system_msg = """Tu es un assistant expert en data quality et nettoyage de données.
        Réponds de manière concise et précise."""
    
    payload = {
    "model": LLM_MODEL,
    "messages": [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 800,    # Augmenté pour éviter les coupures
    "temperature": 0.1,   # Plus bas pour la précision d'audit
    "top_p": 0.9,         # Augmenté pour la richesse du vocabulaire métier
    "frequency_penalty": 0.1
}
    
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        response = r.json()
        
        if "choices" not in response or len(response["choices"]) == 0:
            log.error("Réponse LLM vide")
            return "Erreur: réponse vide de l'IA."
            
        content = response["choices"][0]["message"]["content"].strip()
        
        if len(content.split()) < 3:
            log.warning("Réponse LLM trop courte")
            return "L'IA n'a pas pu générer une réponse utile."
            
        return content
        
    except requests.exceptions.Timeout:
        log.error("Timeout LLM après 45s")
        return "Délai dépassé. L'IA met trop de temps à répondre."
    except requests.exceptions.RequestException as e:
        log.error("Erreur réseau LLM: %s", e)
        return "Erreur de connexion à l'IA. Vérifiez votre clé API."
    except Exception as e:
        log.error("Erreur LLM inattendue: %s", e)
        return f"Erreur technique: {str(e)[:100]}"

# Analyse intelligente du dataset

def analyze_dataset_profile(df: pd.DataFrame) -> dict:
    """Analyse intelligente avec capture d'exemples sémantiques"""
    profile = {
        "type": "unknown",
        "characteristics": [],
        "suggested_questions": [],
        "domain_hints": [],
        "quality_score": 0,
        "semantic_sample": {} # Nouveau : pour stocker des exemples réels
    }
    
    # 1. Capture d'échantillons (clés pour le contexte métier)
    # On prend 3 valeurs uniques représentatives pour les 10 premières colonnes
    for col in df.columns[:10]:
        sample_vals = df[col].dropna().unique()[:3].tolist()
        profile["semantic_sample"][col] = sample_vals

    # 2. Détection du type de dataset (ton code existant optimisé)
    num_cols = len(df.select_dtypes(include=np.number).columns)
    total_cols = len(df.columns)
    
    if total_cols > 0:
        if num_cols / total_cols > 0.7:
            profile["type"] = "numerical"
        elif len(df.select_dtypes(include=['object', 'category']).columns) / total_cols > 0.7:
            profile["type"] = "categorical"
        elif len(df.select_dtypes(include=['datetime']).columns) > 0:
            profile["type"] = "temporal"

    # 3. Détection de domaine par mots-clés
    column_names = " ".join([col.lower() for col in df.columns])
    indicators = [
        (['price', 'revenue', 'tax', 'amount'], "Finance/Comptabilité"),
        (['customer', 'client', 'crm', 'email'], "Relation Client (CRM)"),
        (['sku', 'stock', 'inventory', 'warehouse'], "Logistique/Stock"),
        (['patient', 'medical', 'diagnosis'], "Santé/Médical"),
        (['employee', 'salary', 'hr', 'hiring'], "Ressources Humaines")
    ]
    
    for indicator_list, domain in indicators:
        if any(ind in column_names for ind in indicator_list):
            profile["domain_hints"].append(domain)
            
    # Score de qualité simplifié
    na_rate = df.isna().mean().mean()
    profile["quality_score"] = int((1 - na_rate) * 100)
    
    return profile

def build_adaptive_prompt(df: pd.DataFrame, user_question: str, profile: dict, user_context: str = "") -> str:
    """Construit un prompt ancré dans le contexte métier"""
    
    # Préparation de la vue sémantique
    semantic_view = ""
    for col, examples in profile.get("semantic_sample", {}).items():
        semantic_view += f"- {col} (exemples: {examples})\n"
    
    # Échantillon réel en Markdown (très bien compris par Llama 3)
    data_preview = df.head(5).to_markdown()
    
    # Prise en compte du guidage utilisateur si présent
    context_guide = f"\nINFO SUPPLÉMENTAIRE : L'utilisateur précise que c'est un dataset de type : {user_context}" if user_context else ""

    prompt = f"""
### RÔLE
Tu es un Expert Senior en Audit et Qualité des Données. Ton objectif est d'analyser le dataset suivant avec une rigueur professionnelle.

### 1. CONTEXTE DU DATASET
- **Dimensions** : {len(df)} lignes x {len(df.columns)} colonnes
- **Domaines détectés** : {', '.join(profile['domain_hints']) or 'À déterminer'}
- **Structure des colonnes** :
{semantic_view}
{context_guide}

### 2. APERÇU DES DONNÉES (5 PREMIÈRES LIGNES)
{data_preview}

### 3. MISSION
D'abord, identifie formellement le **SECTEUR D'ACTIVITÉ** et la **NATURE DES ENREGISTREMENTS** (ex: transactions bancaires, logs serveurs, inventaire retail). 
Ensuite, réponds à la question de l'utilisateur : "{user_question}"

### FORMAT DE RÉPONSE IMPÉRATIF
**[IDENTIFICATION MÉTIER]** : (Ton analyse du secteur et du contexte)
**[ANALYSE TECHNIQUE]** : (Ta réponse précise basée sur les données)
**[RECOMMANDATION AUDIT]** : (Un conseil sur la qualité ou la conformité)
"""
    return prompt
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
                df[c] = num; logs.append(f"{c} → num"); continue
            try:
                dat = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                if dat.notna().mean() >= th:
                    df[c] = dat; logs.append(f"{c} → date")
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
        # FIX: Utilisation de select_dtypes pour garantir que nous traitons uniquement les types numériques purs.
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        for c in numeric_cols:
            # S'assurer qu'il y a plus d'une valeur unique pour le calcul de quantile (pour éviter IQR=0)
            if df[c].nunique() > 1:
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
# CHANGEMENT 1: Nom de la page (Utilisation de Axiom AI)
st.set_page_config(page_title="Axiom", layout="wide")
# CHANGEMENT 2: Titre principal (Utilisation de Axiom AI)
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
    st.info("Chargez un fichier pour commencer l'analyse de qualité des données.", icon="💡") # AJOUT 2: Message d'attente pro
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
                # 1️ Tentative de lecture tabulaire
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
                # 2️Tentative JSON Lines (si le txt contient du JSON)
                buffer.seek(0)
                try:
                    df = pd.read_json(buffer, lines=True)
                    return df
                except Exception:
                    # 3️Lecture texte brut → DataFrame à une colonne
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
st.subheader(" Aperçu du Dataset Brut")
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

if st.button(" Lancer l’analyse complète", type="primary"):
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
tab1, tab2, tab3, tab4 = st.tabs(["Rapport Avant nettoyage", "🧹 Nettoyage", "📈 Rapport Après nettoyage + Export", "🤖 Assistant IA"])

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
# Remplacer TOUT le contenu de `with tab4:` par :

with tab4:
    st.header("🤖 Assistant IA – Intelligence Contextuelle")
    
    # Initialisation
    if "chat" not in st.session_state:
        st.session_state.chat = []
    
    # Analyser le dataset une fois
    if "dataset_profile" not in st.session_state and aft is not None:
        with st.spinner("🔍 Analyse intelligente du dataset en cours..."):
            st.session_state.dataset_profile = analyze_dataset_profile(aft)
    
    profile = st.session_state.get("dataset_profile", {})
    
    # Afficher l'analyse du dataset
    if profile:
        with st.expander("Profil détecté du Dataset", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Type", profile["type"].title())
            with col2:
                st.metric("Score Qualité", f"{profile['quality_score']:.0f}/100")
            with col3:
                domains = ", ".join(profile["domain_hints"][:2]) or "Général"
                st.metric("Domaines", domains)
            
            if profile["characteristics"]:
                st.caption("**Caractéristiques:**")
                for char in profile["characteristics"][:3]:
                    st.write(f"• {char}")
    
    # Section 1: Questions suggérées intelligentes
    st.subheader("💡 Questions suggérées")
    
    if profile and profile.get("suggested_questions"):
        # Organiser en 2 colonnes
        col1, col2 = st.columns(2)
        
        questions = profile["suggested_questions"]
        mid = len(questions) // 2
        
        with col1:
            for idx, question in enumerate(questions[:mid]):
                if st.button(f"{question[:50]}...", 
                           key=f"sugg_{idx}", 
                           use_container_width=True,
                           type="secondary"):
                    with st.spinner("Analyse en cours..."):
                        # Construire le prompt adaptatif
                        prompt = build_adaptive_prompt(aft, question, profile)
                        
                        # Appel LLM adaptatif
                        answer = ask_llama(
                            prompt=prompt,
                            max_tokens=400,
                            temperature=0.2,
                            profile=profile
                        )
                        
                        if answer:
                            st.session_state.chat.append(("user", question))
                            st.session_state.chat.append(("bot", answer))
        
        with col2:
            for idx, question in enumerate(questions[mid:], start=mid):
                if st.button(f"{question[:50]}...", 
                           key=f"sugg_{idx}", 
                           use_container_width=True,
                           type="secondary"):
                    with st.spinner("Analyse en cours..."):
                        prompt = build_adaptive_prompt(aft, question, profile)
                        answer = ask_llama(
                            prompt=prompt,
                            max_tokens=400,
                            temperature=0.2,
                            profile=profile
                        )
                        
                        if answer:
                            st.session_state.chat.append(("user", question))
                            st.session_state.chat.append(("bot", answer))
    else:
        # Questions par défaut si pas de profil
        default_questions = [
            "Résume la qualité de ce dataset",
            "Quels problèmes potentiels vois-tu?",
            "Propose des améliorations de nettoyage"
        ]
        
        cols = st.columns(3)
        for idx, question in enumerate(default_questions):
            with cols[idx]:
                if st.button(question, use_container_width=True, type="secondary"):
                    with st.spinner("Analyse..."):
                        answer = ask_llama(question, 350, 0.2)
                        if answer:
                            st.session_state.chat.append(("user", question))
                            st.session_state.chat.append(("bot", answer))
    
    st.divider()
    
    # Section 2: Chat intelligent principal
    st.subheader("💬 Conversation avec l'IA")
    
    # Input utilisateur
    user_input = st.text_area(
        "Posez votre question spécifique:",
        placeholder="Ex: Comment puis-je améliorer la qualité de ces données pour une analyse ML?",
        height=100,
        key="user_input"
    )
    
    # Boutons d'action
    col_send, col_clear, col_regen = st.columns([2, 1, 1])
    
    with col_send:
        send_disabled = not user_input.strip() or not st.secrets.get("OPENROUTER_KEY")
        if st.button("Analyser avec IA", 
                    type="primary", 
                    use_container_width=True,
                    disabled=send_disabled):
            
            with st.spinner("🔍 Analyse contextuelle en cours..."):
                # Construire le prompt adaptatif si profil disponible
                if profile:
                    prompt = build_adaptive_prompt(aft, user_input, profile)
                    max_tokens = 450 if len(user_input) > 100 else 350
                else:
                    prompt = user_input
                    max_tokens = 350
                
                # Appel LLM
                response = ask_llama(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.2,
                    profile=profile
                )
                
                # Ajouter à l'historique
                st.session_state.chat.append(("user", user_input))
                st.session_state.chat.append(("bot", response or "Désolé, je n'ai pas pu générer de réponse."))
                
                st.rerun()
    
    with col_clear:
        if st.button("🗑️ Effacer chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()
    
    with col_regen:
        if st.button("🔄 Re-analyser dataset", use_container_width=True):
            if aft is not None:
                with st.spinner("Nouvelle analyse du dataset..."):
                    st.session_state.dataset_profile = analyze_dataset_profile(aft)
                st.success("Dataset ré-analysé!")
                st.rerun()
    
    # Section 3: Affichage de l'historique
    st.divider()
    
    if st.session_state.chat:
        st.subheader("📜 Historique")
        
        for i in range(0, len(st.session_state.chat), 2):
            if i + 1 < len(st.session_state.chat):
                user_msg = st.session_state.chat[i][1]
                bot_msg = st.session_state.chat[i + 1][1]
                
                # Message utilisateur
                with st.chat_message("user"):
                    st.write(user_msg)
                
                # Message IA avec contexte
                with st.chat_message("assistant"):
                    # Ajouter un badge contextuel si profil disponible
                    if profile:
                        badge = f"**🤖 Axiom AI • {profile['type'].title()}**"
                        st.markdown(badge)
                        st.markdown("---")
                    
                    st.write(bot_msg)
                    
                    # Boutons d'action pour chaque réponse
                    col_copy, col_expand = st.columns([1, 3])
                    with col_copy:
                        if st.button("📋", key=f"copy_{i//2}", help="Copier"):
                            st.code(bot_msg, language=None)
                    
        # Bouton pour tout effacer en bas
        if st.button("🗑️ Effacer toute la conversation", type="secondary"):
            st.session_state.chat = []
            st.rerun()
    else:
        # État initial - conseils contextuels
        st.info("**Comment utiliser l'assistant IA intelligent:**")
        
        advice_cols = st.columns(2)
        
        with advice_cols[0]:
            st.markdown("""
            **Pour de meilleurs résultats:**
            1. Posez des questions spécifiques
            2. Mentionnez vos objectifs
            3. Demandez des recommandations concrètes
            4. Utilisez les questions suggérées
            """)
        
        with advice_cols[1]:
            if profile:
                st.markdown(f"""
                **Contexte détecté:**
                - Type: **{profile['type'].title()}**
                - Domaine: **{', '.join(profile['domain_hints'][:2]) or 'Général'}**
                - Qualité: **{profile['quality_score']:.0f}/100**
                
                L'IA adaptera ses réponses à ce contexte.
                """)
            else:
                st.markdown("""
                **Analyse en cours...**
                L'IA analysera automatiquement
                votre dataset pour des réponses
                plus pertinentes.
                """)
