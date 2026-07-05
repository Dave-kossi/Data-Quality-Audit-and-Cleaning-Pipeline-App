# ✨ Axiom • Data Quality & Audit 

Axiom (anciennement DataCleaner Pro++) est une application Web moderne et intelligente construite avec **Streamlit**. Elle permet d'importer, auditer, optimiser et nettoyer des jeux de données de manière automatisée, tout en intégrant un assistant IA contextuel propulsé par **Llama 3 (via OpenRouter)** pour guider l'utilisateur dans son analyse de données.

---

##  Fonctionnalités Clés

###  1. Audit & Profiling complet
*   **Indicateurs instantanés** : Visualisation immédiate des métriques clés (lignes, colonnes, taux de valeurs manquantes, gain mémoire potentiel).
*   **Rapports Dynamiques** : Génération de rapports de qualité avant/après nettoyage à l'aide de `ydata-profiling` (avec un mode de secours basique intégré en cas d'absence de la dépendance).

###  2. Pipeline de Nettoyage Automatisé & Paramétrable
*   **Déduplication** : Suppression des lignes doublons en un clic.
*   **Normalisation** : Passage des noms de colonnes en minuscules, suppression des espaces et remplacement des caractères spéciaux par des underscores (`_`).
*   **Nettoyage Textuel** : Application automatique de `strip()` et passage en minuscules pour uniformiser les catégories.
*   **Typage Automatique** : Conversion intelligente des colonnes textuelles en types numériques ou temporels (dates) selon un seuil de validité ajustable.
*   **Gestion des Valeurs Manquantes (NA)** : Remplacement par la médiane ou le mode pour le numérique, et par une valeur personnalisée (`_MISSING_` par défaut) pour le texte.
*   **Traitement des Valeurs Aberrantes** : Plafonnement (*capping*) des outliers basé sur la méthode de l'Écart Interquartile (IQR) avec un coefficient ajustable par slider.
*   **Filtre de Variance** : Suppression optionnelle des colonnes constantes ou quasi-constantes (plus de 99% de valeurs identiques).

###  3. Optimisation des Performances
*   **Memory Optimization** : Réduction drastique de l'empreinte RAM en convertissant les types `object` à faible cardinalité en `category` et en effectuant un *downcasting* des types numériques (`int` et `float`).
*   **Gestion Large Files** : Supporte des fichiers volumineux jusqu'à 500 Mo.

###  4. Assistant IA à Intelligence Contextuelle
*   **Analyse Sémantique** : L'application extrait le profil du dataset (numérique, catégoriel, temporel), détecte le domaine métier (Finance, CRM, RH, Santé, Logistique) et capture des exemples réels.
*   **Prompts Adaptatifs** : L'IA reçoit un prompt ancré dans le contexte métier exact de vos données sous forme de tableau Markdown.
*   **Système Multi-Profils** : Le comportement de l'IA (Llama 3.2 3B) s'ajuste dynamiquement (Data Scientist, Expert en analyse catégorielle, Expert en séries temporelles) selon la nature détectée de vos données.
*   **Questions Suggérées** : Génération automatique de questions pertinentes basées sur l'état du dataset.

---

##  Technologies Utilisées

*   **Frontend / UI** : Streamlit
*   **Traitement de Données** : Pandas, NumPy
*   **Profiling** : Ydata-profiling (Optionnel / Recommandé)
*   **Moteur LLM** : OpenRouter API (Modèle : `meta-llama/llama-3.2-3b-instruct`)

---

##  Prérequis & Installation

### 1. Cloner le projet
```bash
git clone [https://github.com/Dave-kossi/axiom-data-quality.git](https://github.com/Dave-kossi/axiom-data-quality.git)
cd axiom-data-quality
```
---
### Installation des dépendances
pip install -r requirements.txt
