# project_dashboard.py: Streamlit dashboard per il progetto IoT Data Analytics
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import requests

# Configurazione
API_URL    = "http://localhost:8000/resolve"
DATA_DIR   = Path('.')
TRAIN_FILE = DATA_DIR / 'train.jsonl'
VAL_FILE   = DATA_DIR / 'val.jsonl'
TEST_FILE  = DATA_DIR / 'test.jsonl'

# Layout pagina
st.set_page_config(
    page_title='DNA Ambiguity Resolver Dashboard',
    layout='wide'
)

# Sidebar per navigazione
df_page = st.sidebar.radio(
    "Seleziona pagina", ['Overview', 'Data Explorer', 'Inference', 'Correzione']
)

# Funzione per caricare JSONL in DataFrame
@st.cache_data
def load_jsonl(path):
    records = []
    with open(path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)

# Overview
if df_page == 'Overview':
    st.title('Progetto IoT Data Analytics')
    st.markdown(
        """
        **Workflow completo:**
        1. Generazione dataset da FASTA (`crea_dataset_split.py`)
        2. Fine-tuning batch (`train_batch.py`)
        3. Fine-tuning online (`train_online.py`)
        4. Confronto modelli (`compare_both.py`)
        5. Servizio REST con FastAPI (`app.py`)
        6. Dashboard e tool di correzione (questo app)

        **File nel progetto:**
        - `train.jsonl`, `val.jsonl`, `test.jsonl`
        - Cartelle modelli: `batch_output/final`, `river_ckpts/final`
        - Script Python per ogni fase
        """
    )

# Data Explorer
elif df_page == 'Data Explorer':
    st.title('Data Explorer')
    dataset = st.sidebar.selectbox('Dataset', ['train', 'validation', 'test'])
    df = load_jsonl(DATA_DIR / f'{dataset}.jsonl')
    st.write(f"Visualizzazione del dataset **{dataset}** - {len(df)} esempi")
    st.dataframe(df)

# Inference
elif df_page == 'Inference':
    st.title('Inference Interattiva')
    seq = st.text_area('Inserisci la sequenza ambigua:', height=100)
    if st.button('De-ambigua ora'):
        if not seq.strip():
            st.warning('Inserisci una sequenza valida')
        else:
            with st.spinner('Chiamata al servizio...'):
                try:
                    resp = requests.post(API_URL, json={'sequence': seq})
                    resp.raise_for_status()
                    resolved = resp.json().get('resolved', '')
                    st.success('Risultato:')
                    st.code(resolved)
                except Exception as e:
                    st.error(f'Errore: {e}')

# Correzione
elif df_page == 'Correzione':
    st.title('Tool di Correzione Manuale')
    df_test = load_jsonl(TEST_FILE)
    # Pre-compute predictions
    if 'predicted' not in df_test.columns:
        df_test['predicted'] = df_test['input'].apply(
            lambda s: requests.post(API_URL, json={'sequence': s}).json().get('resolved', '')
        )
    # Display editable table
    st.markdown('Modifica manualmente le predizioni se necessario:')
    edited = st.data_editor(
        df_test.rename(columns={'input': 'Input', 'target': 'Target', 'predicted': 'Predicted'})[
            ['Input', 'Target', 'Predicted']
        ],
        num_rows='dynamic'
    )
    # Export
    if st.button('Esporta correzioni'):
        save_path = DATA_DIR / 'test_corrected.jsonl'
        with open(save_path, 'w') as fout:
            for _, row in edited.iterrows():
                rec = {'input': row['Input'], 'target': row['Predicted']}
                fout.write(json.dumps(rec) + '\n')
        st.success(f'File corretto salvato in {save_path}')
