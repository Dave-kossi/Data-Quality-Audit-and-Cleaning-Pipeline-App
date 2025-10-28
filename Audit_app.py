import streamlit as st
import pandas as pd
import io
import tempfile
import os
import webbrowser
from pathlib import Path

# ===============================
# 1. Import ydata-profiling avec fallback
# ===============================
try:
    from ydata_profiling import ProfileReport
    YDATA_AVAILABLE = True
    st.success("✅ ydata-profiling disponible - Rapports interactifs activés")
except Exception as e:
    st.error(f"❌ ydata-profiling non disponible: {e}")
    YDATA_AVAILABLE = False

# ===============================
# 2. Chargement du dataset
# ===============================
def charger_donnees(uploaded_file):
    try:
        extension = uploaded_file.name.split(".")[-1].lower()
        file_bytes = io.BytesIO(uploaded_file.getvalue())
        
        if extension == "csv":
            return pd.read_csv(file_bytes)
        elif extension in ["xls", "xlsx"]:
            return pd.read_excel(file_bytes)
        elif extension == "json":
            uploaded_file.seek(0)
            content = uploaded_file.read().decode('utf-8')
            return pd.read_json(io.StringIO(content))
        elif extension == "parquet":
            return pd.read_parquet(file_bytes)
        elif extension == "txt":
            return pd.read_csv(file_bytes, delimiter="\t")
        else:
            st.error(f"❌ Extension non supportée : {extension}")
            return None
            
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {str(e)}")
        return None

# ===============================
# 3. Nettoyage automatique
# ===============================
def nettoyer_donnees(df: pd.DataFrame):
    df_clean = df.copy()
    transformations = []
    
    # 1. Supprimer les doublons
    avant_doublons = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    apres_doublons = len(df_clean)
    if avant_doublons != apres_doublons:
        transformations.append(f"🗑️ {avant_doublons - apres_doublons} doublons supprimés")
    
    # 2. Nettoyer les noms de colonnes
    anciens_noms = df_clean.columns.tolist()
    df_clean.columns = (
        df_clean.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    nouveaux_noms = df_clean.columns.tolist()
    if anciens_noms != nouveaux_noms:
        transformations.append("🔠 Noms de colonnes normalisés")
    
    # 3. Conversion automatique des types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Essai conversion numérique
            try:
                numeric_test = pd.to_numeric(df_clean[col], errors='coerce')
                if numeric_test.notna().mean() > 0.7:
                    df_clean[col] = numeric_test
                    transformations.append(f"🔢 {col} → numérique")
                    continue
            except:
                pass
            
            # Essai conversion date
            try:
                date_test = pd.to_datetime(df_clean[col], errors='coerce')
                if date_test.notna().mean() > 0.7:
                    df_clean[col] = date_test
                    transformations.append(f"📅 {col} → datetime")
            except:
                pass
    
    # 4. Gestion des valeurs manquantes
    for col in df_clean.columns:
        na_count = df_clean[col].isna().sum()
        if na_count > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                transformations.append(f"📈 {col}: {na_count} NA → médiane")
            else:
                df_clean[col].fillna("inconnu", inplace=True)
                transformations.append(f"📝 {col}: {na_count} NA → 'inconnu'")
    
    return df_clean, transformations

# ===============================
# 4. Génération des rapports ydata-profiling (VERSION CORRIGÉE)
# ===============================
def generer_rapport_ydata(df, titre, dossier_temp):
    """Génère un vrai rapport ydata-profiling avec configuration corrigée"""
    
    if not YDATA_AVAILABLE:
        st.error("ydata-profiling n'est pas disponible")
        return None
    
    try:
        with st.spinner(f"🔄 Création du rapport {titre}..."):
            # Configuration SIMPLIFIÉE et CORRECTE
            profile = ProfileReport(
                df,
                title=titre,
                explorative=True,
                minimal=False,
                # Suppression des paramètres problématiques
            )
            
            # Sauvegarder le rapport HTML
            nom_fichier = f"rapport_{titre.lower().replace(' ', '_')}.html"
            chemin_rapport = os.path.join(dossier_temp, nom_fichier)
            profile.to_file(chemin_rapport)
            
            return chemin_rapport
            
    except Exception as e:
        st.error(f"❌ Erreur avec ydata-profiling: {e}")
        
        # Solution alternative si l'erreur persiste
        st.info("""
        **🔄 Solution alternative :**
        Essayez d'installer une version différente :
        ```bash
        pip uninstall ydata-profiling -y
        pip install ydata-profiling==4.6.1
        ```
        """)
        return None

# ===============================
# 5. Interface Streamlit
# ===============================
st.set_page_config(
    page_title="🧹 DataCleaner Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧹 DataCleaner Pro")
st.markdown("Application de **nettoyage automatique** avec **rapports ydata-profiling interactifs**")

# Initialisation du dossier temporaire
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# Sidebar avec informations ydata-profiling
with st.sidebar:
    st.header("📊 ydata-profiling")
    
    if YDATA_AVAILABLE:
        st.success("✅ **FONCTIONNEL**")
        st.markdown("**Visualisations disponibles :**")
        st.markdown("""
        - 📈 **Histogrammes interactifs**
        - 🔥 **Heatmaps de corrélation**
        - 📊 **Matrices de valeurs manquantes**
        - 🎯 **Analyses de distributions**
        - 📉 **Boxplots et outliers**
        - 🔍 **Scatter plots interactifs**
        """)
        
        # Affichage de la version
        try:
            import ydata_profiling
            st.write(f"Version: {ydata_profiling.__version__}")
        except:
            pass
    else:
        st.error("❌ **INDISPONIBLE**")
        st.markdown("**Solutions :**")
        st.code("""
# Option 1 (Recommandée)
pip uninstall ydata-profiling -y
pip install ydata-profiling==4.6.1

# Option 2  
pip uninstall ydata-profiling -y
pip install ydata-profiling==4.5.1
        """)

# Upload de fichier
uploaded_file = st.file_uploader(
    "📂 **Sélectionnez votre fichier**", 
    type=["csv", "xls", "xlsx", "json", "parquet", "txt"]
)

if uploaded_file is not None:
    with st.spinner("Chargement des données..."):
        df = charger_donnees(uploaded_file)
    
    if df is not None and not df.empty:
        # Métriques rapides
        st.subheader("📊 Aperçu du dataset")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Lignes", f"{len(df):,}")
        with col2:
            st.metric("Colonnes", len(df.columns))
        with col3:
            st.metric("Valeurs manquantes", f"{df.isna().sum().sum():,}")
        with col4:
            st.metric("Doublons", f"{df.duplicated().sum():,}")
        
        # Aperçu rapide
        with st.expander("👀 Aperçu des données originales", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Bouton principal
        if st.button("🚀 **Lancer l'analyse ydata-profiling complète**", type="primary", use_container_width=True):
            
            # Création des onglets
            tab1, tab2, tab3 = st.tabs([
                "📊 Rapport AVANT", 
                "🧹 Nettoyage", 
                "📈 Rapport APRÈS"
            ])
            
            with tab1:
                st.header("📊 Rapport ydata-profiling AVANT nettoyage")
                
                if YDATA_AVAILABLE:
                    rapport_avant = generer_rapport_ydata(df, "Audit AVANT Nettoyage", st.session_state.temp_dir)
                    if rapport_avant:
                        st.success(f"✅ Rapport généré avec succès !")
                        
                        # Boutons pour le rapport AVANT
                        col_btn1, col_btn2 = st.columns(2)
                        
                        with col_btn1:
                            with open(rapport_avant, "rb") as f:
                                st.download_button(
                                    label="📥 Télécharger rapport AVANT",
                                    data=f,
                                    file_name="rapport_avant_nettoyage.html",
                                    mime="text/html",
                                    use_container_width=True
                                )
                        
                        with col_btn2:
                            if st.button("🌐 Ouvrir rapport AVANT", use_container_width=True):
                                webbrowser.open(f"file://{rapport_avant}")
                        
                        # Aperçu du contenu
                        st.info("""
                        **📋 Le rapport HTML contient :**
                        - **Overview** : Résumé général avec alertes
                        - **Variables** : Analyse détaillée par colonne  
                        - **Interactions** : Graphiques interactifs entre variables
                        - **Correlations** : Matrices de corrélation
                        - **Missing values** : Analyse des valeurs manquantes
                        - **Sample** : Extrait des données
                        """)
                    else:
                        st.error("Échec de la génération du rapport")
                else:
                    st.error("ydata-profiling n'est pas disponible")
            
            with tab2:
                st.header("🧹 Processus de Nettoyage")
                
                with st.spinner("Nettoyage automatique en cours..."):
                    df_clean, transformations = nettoyer_donnees(df)
                
                # Résultats du nettoyage
                if transformations:
                    st.success("✅ **Transformations appliquées :**")
                    for transformation in transformations:
                        st.write(f"• {transformation}")
                else:
                    st.info("ℹ️ Aucune transformation nécessaire")
                
                # Comparaison avant/après
                st.subheader("📊 Comparaison avant/après")
                col_av1, col_av2, col_av3, col_av4 = st.columns(4)
                
                with col_av1:
                    st.metric("Lignes AVANT", len(df))
                    st.metric("Lignes APRÈS", len(df_clean))
                
                with col_av2:
                    st.metric("Colonnes AVANT", len(df.columns))
                    st.metric("Colonnes APRÈS", len(df_clean.columns))
                
                with col_av3:
                    na_avant = df.isna().sum().sum()
                    na_apres = df_clean.isna().sum().sum()
                    st.metric("Valeurs manquantes AVANT", na_avant)
                    st.metric("Valeurs manquantes APRÈS", na_apres)
                
                with col_av4:
                    dup_avant = df.duplicated().sum()
                    dup_apres = df_clean.duplicated().sum()
                    st.metric("Doublons AVANT", dup_avant)
                    st.metric("Doublons APRÈS", dup_apres)
                
                # Aperçu des données nettoyées
                with st.expander("👀 Aperçu des données nettoyées", expanded=True):
                    st.dataframe(df_clean.head(15), use_container_width=True)
            
            with tab3:
                st.header("📈 Rapport ydata-profiling APRÈS nettoyage")
                
                if YDATA_AVAILABLE:
                    rapport_apres = generer_rapport_ydata(df_clean, "Audit APRÈS Nettoyage", st.session_state.temp_dir)
                    if rapport_apres:
                        st.success(f"✅ Rapport généré avec succès !")
                        
                        # Boutons pour le rapport APRÈS
                        col_btn3, col_btn4 = st.columns(2)
                        
                        with col_btn3:
                            with open(rapport_apres, "rb") as f:
                                st.download_button(
                                    label="📥 Télécharger rapport APRÈS",
                                    data=f,
                                    file_name="rapport_apres_nettoyage.html",
                                    mime="text/html",
                                    use_container_width=True
                                )
                        
                        with col_btn4:
                            if st.button("🌐 Ouvrir rapport APRÈS", use_container_width=True):
                                webbrowser.open(f"file://{rapport_apres}")
                        
                        # Conseils de comparaison
                        st.info("""
                        **🔍 Comparez les rapports :**
                        1. **Ouvrez les deux rapports** dans votre navigateur
                        2. **Comparez l'onglet Overview** : réduction des alertes
                        3. **Vérifiez Missing Values** : diminution des NA
                        4. **Observez les distributions** : impact du nettoyage
                        """)
                    else:
                        st.error("Échec de la génération du rapport")
                else:
                    st.error("ydata-profiling n'est pas disponible")
            
            # Section Export
            st.markdown("---")
            st.header("💾 Export des Données Nettoyées")
            
            # Options d'export
            format_export = st.selectbox(
                "Choisissez le format d'export :",
                ["CSV", "Excel", "JSON"],
                index=0
            )
            
            try:
                buffer = io.BytesIO()
                
                if format_export == "CSV":
                    df_clean.to_csv(buffer, index=False, encoding='utf-8')
                    mime_type = "text/csv"
                    file_ext = "csv"
                elif format_export == "Excel":
                    df_clean.to_excel(buffer, index=False)
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    file_ext = "xlsx"
                elif format_export == "JSON":
                    json_str = df_clean.to_json(orient="records", indent=2, force_ascii=False)
                    buffer.write(json_str.encode('utf-8'))
                    mime_type = "application/json"
                    file_ext = "json"
                
                buffer.seek(0)
                
                st.download_button(
                    label=f"📥 Télécharger les données nettoyées ({format_export})",
                    data=buffer,
                    file_name=f"donnees_nettoyees.{file_ext}",
                    mime=mime_type,
                    type="primary",
                    use_container_width=True
                )
                
                st.success("✅ Données prêtes au téléchargement !")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de l'export: {e}")
            
            # Message final
            st.balloons()
            st.success("🎉 **Analyse ydata-profiling terminée avec succès !**")

else:
    # Page d'accueil
    st.markdown("""
    ### 🎯 **ydata-profiling - Audit de données professionnel**
    
    **📊 Fonctionnalités des rapports HTML :**
    
    **🔍 Analyse exploratoire complète :**
    - **Overview** : Résumé avec scores de qualité
    - **Variables** : Analyse statistique par colonne
    - **Interactions** : Visualisations entre variables
    - **Corrélations** : Matrices interactives
    - **Missing Values** : Diagrammes des valeurs manquantes
    
    **📈 Visualisations interactives :**
    - Histogrammes avec zoom
    - Boxplots pour outliers
    - Scatter plots avec sélection
    - Heatmaps de corrélation
    - Graphiques de distribution
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🧹 DataCleaner Pro • ydata-profiling • Audit interactif"
    "</div>", 
    unsafe_allow_html=True
)