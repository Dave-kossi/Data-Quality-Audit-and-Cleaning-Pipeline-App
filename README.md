🌟 Axiom — Data Quality & Audit (LLM Powered)
Automated Data Cleaning • Profiling • Optimization • LLM Analysis
<p align="center"> <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/> <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/AI-OpenRouter-blue?style=for-the-badge&logo=openai&logoColor=white"/> <img src="https://img.shields.io/badge/Data_Profiling-ydata_profiling-orange?style=for-the-badge"/> </p>

# A propos
---
Axiom est une application intelligente d’audit et de nettoyage de données construite avec Streamlit, intégrant un LLM via OpenRouter.
Elle permet :
- d’explorer, nettoyer et optimiser un dataset
- de générer un rapport automatique avant/après
- de discuter avec une IA qui analyse les données
- d’exporter le dataset optimisé dans plusieurs formats
Axiom combine Data Engineering, Data Quality, Analyse Assistée par IA et profiling professionnel dans une interface simple et puissante.
--- 

---
##  Fonctionnalités

###  1. Chargement intelligent
- Support : **CSV**, **XLSX**, **JSON**, **Parquet**, **TXT**
- Détection automatique des séparateurs
- Lecture intelligente **JSON / JSON Lines**
- Prévisualisation immédiate

### 2. Nettoyage automatique
- Suppression des doublons
- Normalisation des colonnes
- Conversion des types (dates, numériques…)
- Gestion intelligente des valeurs manquantes
- Nettoyage texte (trim, lower, accents)
- Détection et correction des outliers (**IQR**)
- Suppression des colonnes à variance nulle

###  3. Optimisation mémoire
- Downcast automatique (int/float)
- Conversion en **category**
- Rapport clair avant/après optimisation et nettoyage 

### 4. Profiling complet
- Compatible **ydata-profiling**
- Export des rapports en HTML intégré

###  5. Analyse IA contextuelle
- Résumé automatique du dataset
- Détection d’anomalies, suggestions, règles métier
- Chat IA avec informations dynamiques :
  - cette partie permet a l`utilisateur de mieux comprendre son dataset. 

