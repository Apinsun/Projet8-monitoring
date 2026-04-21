import streamlit as st
from supabase import create_client, Client
import pandas as pd
import os
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Imports Evidently
from evidently import Report
from evidently.presets import DataDriftPreset

# 1. Configuration initiale
load_dotenv()
st.set_page_config(page_title="Dashboard MLOps", page_icon="🏦", layout="wide")
st.title("🏦 Dashboard de Monitoring - Prêt à Dépenser")

# 2. Connexion à Supabase
@st.cache_resource
def init_connection():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        st.error("❌ Erreur : Variables d'environnement Supabase introuvables.")
        st.stop()
    return create_client(url, key)

supabase: Client = init_connection()

@st.cache_data(ttl=60)
def load_production_data():
    all_data = []
    limit = 1000
    offset = 0
    
    while True:
        # On demande une tranche (range) de 1000 lignes
        response = (
            supabase.table("predictions_logs")
            .select("*")
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        
        batch = response.data
        if not batch:
            break
            
        all_data.extend(batch)
        
        # Si on a récupéré moins que la limite, c'est qu'on a fini
        if len(batch) < limit:
            break
            
        offset += limit
        
        # Sécurité pour ne pas boucler à l'infini si ta DB est immense
        if offset >= 5000: 
            break

    return pd.DataFrame(all_data)

@st.cache_data
def load_reference_data():
    try:
        df_ref = pd.read_csv("data/dataset_raw.csv")
        # On remplace les NaN de Pandas par des None pour éviter les bugs
        return df_ref.replace({pd.NA: None})
    except FileNotFoundError:
        return pd.DataFrame()

# 4. Le moteur d'Analyse de Drift
@st.cache_data(ttl=3600) # On met en cache le rapport pendant 1 heure (calcul lourd)
def generate_drift_report(df_ref, df_prod):
    # On extrait les variables JSON en vraies colonnes
    df_current_features = pd.json_normalize(df_prod['client_features'])
    
    # On trouve les colonnes communes
    common_cols = list(set(df_ref.columns).intersection(set(df_current_features.columns)))
    
    # On retire manuellement les colonnes qui ne sont pas des features ML
    cols_to_ignore = ['SK_ID_CURR', 'TARGET', 'is_test']
    common_cols = [col for col in common_cols if col not in cols_to_ignore]
    
    df_ref_clean = df_ref[common_cols]
    df_cur_clean = df_current_features[common_cols]

    # On force les types pour aider Evidently
    for col in common_cols:
        df_cur_clean[col] = df_cur_clean[col].astype(df_ref_clean[col].dtype, errors='ignore')

    # On crée le plan de construction (le blueprint)
    report_definition = Report(metrics=[DataDriftPreset()])
    
    # On stocke le calcul final dans une nouvelle variable
    mon_rapport_calcule = report_definition.run(reference_data=df_ref_clean, current_data=df_cur_clean)
    
    # On sauvegarde à partir de ce résultat calculé (qui possède bien .save_html) !
    mon_rapport_calcule.save_html("drift_report.html")
    
    with open("drift_report.html", "r", encoding="utf-8") as f:
        return f.read()

# --- CHARGEMENT DE TOUTES LES DONNÉES ---
with st.spinner('Chargement des données...'):
    df_logs = load_production_data()
    df_ref = load_reference_data()

if df_logs.empty:
    st.warning("Aucune donnée de production trouvée.")
    st.stop()


# ==========================================
# 📊 L'INTERFACE UTILISATEUR EN ONGLETS
# ==========================================
tab_kpi, tab_drift = st.tabs(["📈 Monitoring API", "🚨 Analyse du Data Drift"])

# --- ONGLET 1 : KPIs CLASSIQUES ---
with tab_kpi:
    st.success(f"✅ API sous surveillance : {len(df_logs)} requêtes loggées.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total des Requêtes", len(df_logs))
    with col2:
        st.metric("Latence Moyenne", f"{round(df_logs['execution_time_ms'].mean(), 2)} ms")
    with col3:
        st.metric("Taux de Refus", f"{round((df_logs['decision'] == 'Refusé').mean() * 100, 1)} %")

    st.subheader("Dernières prédictions en direct")
    st.dataframe(df_logs[['created_at', 'score_defaut', 'decision', 'execution_time_ms', 'is_test']].sort_values(by="created_at", ascending=False).head(15))


# --- ONGLET 2 : DATA DRIFT (EVIDENTLY) ---
with tab_drift:
    st.header("Détection Automatique de Dérive des Données")
    
    # Affichage du volume de données analysé
    st.metric(label="Volume de données en Production analysé", value=f"{len(df_logs)} requêtes")

    if df_ref.empty:
        st.error("❌ Fichier de référence ('dataset_raw.csv') introuvable. Placez-le dans un dossier 'data/' à la racine.")
    else:
        st.info("Le rapport est généré en comparant le Dataset d'entraînement avec les logs actuels de production.")
        
        # Bouton pour forcer le recalcul si besoin (car on a mis le rapport en cache)
        if st.button("🔄 Rafraîchir l'analyse de Drift"):
            st.cache_data.clear()
            
        with st.spinner("Analyse statistique en cours par Evidently AI... (Cela peut prendre 10-20 secondes)"):
            drift_html = generate_drift_report(df_ref, df_logs)
            
            # MAGIE : On intègre la page web HTML d'Evidently directement dans Streamlit !
            components.html(drift_html, height=1000, scrolling=True)
