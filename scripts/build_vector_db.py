# scripts/build_vector_db.py

import os
import sys
import logging
import pandas as pd
from datetime import timedelta
from typing import List
import mlflow
import mlflow.pyfunc
import argparse # [ENHANCEMENT] To accept arguments

# Setup path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from outage_predictor.utils import config, setup_logging
    from outage_predictor.data.data_loader import load_data_from_db
    from langchain_chroma import Chroma
    from chromadb.config import Settings
    from langchain_core.embeddings import Embeddings
    from langchain_community.embeddings import SentenceTransformerEmbeddings # For base model
    from langchain_core.documents import Document

except ImportError as e:
    print(f"FATAL ERROR: Could not import project modules or RAG libraries: {e}")
    sys.exit(1)

# --- Initialize Logging ---
setup_logging()
logger = logging.getLogger('outage_predictor_rag_builder')

# --- Configuration ---
C_DATA = config['data']
C_MLOPS = config['mlops']
DB_PATH = C_DATA.get('DB_PATH', 'data/log_outage_data.db')
WINDOW_MINUTES = C_DATA.get('ROLLING_WINDOW_MINUTES', 30)

# --- RAG Configuration ---
BASE_EMBEDDING_MODEL = C_MLOPS.get('BASE_EMBEDDING_MODEL_FOR_RAG')
FINETUNED_EMBEDDING_MODEL_URI = f"models:/{C_MLOPS.get('EMBEDDING_MODEL_REGISTRY_NAME')}@Production"
CHROMA_COLLECTION_NAME = "outage_logs"

# --- MLflow Embedding Wrapper ---
class MlflowPyfuncEmbedding(Embeddings):
    def __init__(self, pyfunc_model): self.model = pyfunc_model
    def embed_documents(self, texts: List[str]) -> List[List[float]]: return self.model.predict(texts)
    def embed_query(self, text: str) -> List[float]: return self.model.predict([text])[0]

# --- Helper Functions --- (format_log_window_to_string, create_log_documents are unchanged)
def format_log_window_to_string(window: pd.DataFrame) -> str:
    return "\n".join(
        f"Severity: {row.Severity} Component: {row.Component} Message: {row.Message}"
        for _, row in window.iterrows()
    )

def create_log_documents(df: pd.DataFrame, window_size_minutes: int) -> List[Document]:
    logger.info(f"Creating documents with {window_size_minutes}min sliding window...")
    documents = []
    window_delta = pd.Timedelta(minutes=window_size_minutes)
    total_rows = len(df)
    for i, (index, row) in enumerate(df.iterrows()):
        # ... (progress logging)
        window_start_time = index - window_delta
        current_window_df = df.loc[window_start_time:index]
        if current_window_df.empty or len(current_window_df) < 2: continue
        doc_string = format_log_window_to_string(current_window_df)
        outage_flag = int(row['Outage_Flag'])
        doc = Document(page_content=doc_string, metadata={"end_timestamp": index.isoformat(), "Outage_Flag": outage_flag, "window_size_min": window_size_minutes, "log_count": len(current_window_df)})
        documents.append(doc)
    logger.info(f"Created {len(documents)} total log window documents.")
    return documents


def build_and_persist_vector_db(documents: List[Document], use_finetuned_model: bool, db_suffix: str):
    """Builds the Chroma vector store using the specified embedding model."""

    if not documents: logger.error("No documents..."); return

    # [ENHANCEMENT] Select embedding model based on flag
    if use_finetuned_model:
        logger.info(f"Loading FINE-TUNED embedding model from MLflow: {FINETUNED_EMBEDDING_MODEL_URI}...")
        mlflow.set_tracking_uri(C_MLOPS.get('MLFLOW_TRACKING_URI'))
        try:
            model_pyfunc = mlflow.pyfunc.load_model(model_uri=FINETUNED_EMBEDDING_MODEL_URI)
            embeddings = MlflowPyfuncEmbedding(model_pyfunc)
            model_source = "Fine-Tuned (MLflow)"
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}. Falling back to base model.")
            embeddings = SentenceTransformerEmbeddings(model_name=BASE_EMBEDDING_MODEL)
            model_source = f"Base ({BASE_EMBEDDING_MODEL}) - Fallback"
    else:
        logger.info(f"Loading BASE embedding model: {BASE_EMBEDDING_MODEL}...")
        embeddings = SentenceTransformerEmbeddings(model_name=BASE_EMBEDDING_MODEL)
        model_source = f"Base ({BASE_EMBEDDING_MODEL})"

    logger.info(f"Using embedding model: {model_source}")

    # [ENHANCEMENT] Use suffix to create unique persist directory
    chroma_persist_dir = os.path.join(os.path.dirname(DB_PATH), f"chroma_db{db_suffix}")
    logger.info(f"Building and persisting ChromaDB to: {chroma_persist_dir}...")
    client_settings = Settings(anonymized_telemetry=False)

    db = Chroma.from_documents(
        documents, embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=chroma_persist_dir, # Use unique directory
        client_settings=client_settings
    )

    logger.info(f"Vector database ({model_source}) built and persisted successfully.")


def run_rag_data_prep(use_finetuned: bool, suffix: str): # [ENHANCEMENT] Accept arguments
    """Main function to run the RAG data pipeline."""
    model_type = "Fine-Tuned" if use_finetuned else "Base"
    logger.info("=" * 70)
    logger.info(f"STEP RAG-BUILD ({model_type}): Starting RAG Vector Database Build")
    logger.info("=" * 70)
    try:
        df_raw = load_data_from_db()
        documents = create_log_documents(df_raw, WINDOW_MINUTES)
        build_and_persist_vector_db(documents, use_finetuned_model=use_finetuned, db_suffix=suffix)
        logger.info(f"--- RAG Vector Database Build ({model_type}) Complete ---")
    except Exception as e:
        logger.error(f"FATAL ERROR during RAG build ({model_type}): {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # [ENHANCEMENT] Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Build RAG Vector DB")
    parser.add_argument(
        "--use-finetuned",
        action="store_true", # Makes it a flag, default is False
        help="Use the fine-tuned embedding model from MLflow Registry."
    )
    args = parser.parse_args()

    # Determine suffix based on argument
    db_suffix = C_MLOPS.get('RAG_DB_SUFFIX_FINETUNED') if args.use_finetuned else C_MLOPS.get('RAG_DB_SUFFIX_BASE')

    run_rag_data_prep(use_finetuned=args.use_finetuned, suffix=db_suffix)