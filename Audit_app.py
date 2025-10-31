# app.py – DataCleaner Pro++  (LLM edition – Streamlit-Cloud-ready)
import streamlit as st, pandas as pd, numpy as np, io, uuid, shutil, os, platform, tempfile, logging, sys, requests, base64, ast, time, json
from pathlib import Path

# ---------- CONFIG ---------- #
TEMP_DIR   = Path(tempfile.gettempdir()) / "datacleaner"
TEMP_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.FileHandler(TEMP_DIR / "app.log"), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("datacleaner")
MAX_SIZE   = 500 * 1_000_000  # 500 Mo

# ---------- LLM – OPENROUTER ---------- #
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL      = "meta-llama/llama-3.2-3b-instruct"

def ask_llama(prompt: str, max_tokens: 350) -> str:
    """Call Llama-3.2-3B on OpenRouter (free tier)."""
    headers = {"Authorization": f"Bearer {st.secrets['OPENROUTER_KEY']}",
               "HTTP-Referer": "https://datacleaner-pro.streamlit.app",
               "X-Title": "DataCleaner-Pro"}
    payload = {"model": LLM_MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.2}
    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.warning("LLM error: %s", e)
        return None

# ---------- CORE UTILS ---------- #
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
    if params["drop_dup"]:
        prev = len(df)
        df = df.drop_duplicates()
        if (d := prev - len(df)):
            logs.append(f"🗑️  {d} doublons supprimés")
    if params["norm_cols"]:
        old = df.columns.tolist()
        df.columns = df.columns.str.strip().str.lower().str.replace(r"\W+", "_", regex=True)
        if old != df.columns.tolist():
            logs.append("🔠 noms normalisés")
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
    df, gain = memory_opt(df)
    if gain > 5:
        logs.append(f"💾 mémoire -{gain:.1f}%")
    return df, logs

def build_report(df: pd.DataFrame, title: str, minimal: bool) -> Path | None:
    try:
        from ydata_profiling import ProfileReport
        file = TEMP_DIR / f"report_{title.replace(' ','_')}_{uuid.uuid4().hex}.html"
        ProfileReport(df, title=title, minimal=minimal).to_file(str(file))
        return file
    except Exception as e:
        log.warning("ydata fail: %s", e); return None

def fallback_report(df: pd.DataFrame, title: str):
    st.subheader(f"📋 Rapport basique – {title}")
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

# ---------- STREAMIT UI ---------- #
st.set_page_config(page_title="🧽 DataCleaner Pro++  (LLM)", layout="wide")
st.title("🧽 DataCleaner Pro++  •  LLAma-3.2 édition")
st.markdown("Audit & nettoyage **intelligent** – hébergé sur Streamlit Cloud **gratuit**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    params = {
        "drop_dup":  st.checkbox("Supprimer doublons", True),
        "norm_cols": st.checkbox("Normaliser colonnes", True),
        "auto_type": st.checkbox("Conversion auto types", True),
        "handle_na": st.checkbox("Gérer NA", True),
        "na_thresh": st.slider("Taux min valeurs valides (%)", 50, 90, 70),
        "na_strat":  st.selectbox("Stratégie NA num", ["median", "mode"]),
        "na_fill":   st.text_input("Valeur remplacement NA texte", "_MISSING_"),
        "minimal":   st.checkbox("Mode ydata minimal (rapide)", True),
    }
    if st.button("🗑️  vider fichiers temp"):
        shutil.rmtree(TEMP_DIR, ignore_errors=True); TEMP_DIR.mkdir(exist_ok=True); st.success("Temp vidé")
    st.info(f"Système: {platform.system()} | Dossier: {TEMP_DIR}")

# ---------- CHARGEMENT ---------- #
uploaded = st.file_uploader("📂 Sélectionnez votre fichier",
                            type=["csv","xlsx","json","parquet","txt"], accept_multiple_files=False)
if not uploaded:
    st.stop()
if uploaded.size > MAX_SIZE:
    st.error("Fichier > 500 Mo refusé"); st.stop()

@st.cache_data(show_spinner=False)
def load(uploaded):
    try:
        ext = Path(uploaded.name).suffix.lower()
        buffer = io.BytesIO(uploaded.getbuffer())
        match ext:
            case ".csv"|".txt": return pd.read_csv(buffer)
            case ".xlsx":       return pd.read_excel(buffer)
            case ".json":       return pd.read_json(buffer)
            case ".parquet":    return pd.read_parquet(buffer)
    except Exception as e:
        log.exception("load"); st.error(str(e))
    return None

df_raw = load(uploaded)
if df_raw is None:
    st.stop()

df_opt, gain = memory_opt(df_raw)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lignes", f"{len(df_raw):,}")
c2.metric("Colonnes", len(df_raw.columns))
c3.metric("NA", df_raw.isna().sum().sum())
c4.metric("Mémoire gagnée", f"{gain:.1f}%")

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
    st.session_state["logs"]   = logs
    st.rerun()

if "after" not in st.session_state:
    st.stop()

bef, aft = st.session_state["before"], st.session_state["after"]
tab1, tab2, tab3, tab4 = st.tabs(["📊 Avant", "🧹 Nettoyage", "📈 Après + Export", "🤖 LLM"])
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
        case "csv":     aft.to_csv(buf, index=False)
        case "xlsx":    aft.to_excel(buf, index=False)
        case "json":    buf.write(aft.to_json(orient="records", indent=2).encode())
        case "parquet": aft.to_parquet(buf)
    buf.seek(0)
    st.download_button("💾 Télécharger dataset", data=buf, file_name=f"clean.{fmt}")

# ---------- LLM VALUE-ADD ----------
# ---------- CHAT AVEC L'IA ----------
with tab4:
    st.header("🤖 Discutez avec l’IA à propos de vos données")

    # Initialiser l’historique
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Préparer le contexte dataset (schema + 5 lignes)
    def build_dataset_context():
        # Schema
        schema = aft.dtypes.astype(str).to_frame("type").assign(uniques=aft.nunique())
        # 5 premières lignes
        sample = aft.head(5).to_dict(orient="records")
        return f"Schema des colonnes :\n{schema.to_string()}\n\n5 premières lignes :\n{json.dumps(sample, ensure_ascii=False, indent=2)}"

    # Zone de saisie
    user_msg = st.text_input("Votre question :", placeholder="Ex. : Quelles colonnes ont le plus d'impact ?")
    if st.button("📤 Envoyer"):
        if not user_msg.strip():
            st.warning("Message vide.")
        else:
            with st.spinner("LLM réfléchit..."):
                context = build_dataset_context()
                prompt = f"""Tu es un assistant data-quality. Voici le contexte du dataset nettoyé :
{context}

Question de l'utilisateur : {user_msg}
Réponse concise (3-5 phrases max) :"""
                answer = ask_llama(prompt, max_tokens=350)
                if not answer:
                    answer = "Désolé, le service LLM est hors-ligne pour le moment."
                # Ajouter à l’historique
                st.session_state.chat.append(("user", user_msg))
                st.session_state.chat.append(("bot", answer))

    # Affichage conversation
    for author, msg in st.session_state.chat:
        if author == "user":
            st.markdown(f'<div style="text-align:right; color:#0d47a1;"><b>Moi :</b> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="text-align:left; color:#388e3c;"><b>IA :</b> {msg}</div>', unsafe_allow_html=True)

    # Bouton Clear
    if st.button("🗑️  Effacer la conversation"):
        st.session_state.chat = []
        st.rerun()
