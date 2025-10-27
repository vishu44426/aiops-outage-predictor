# src/outage_predictor/deployment/llm_predictor_service.py

import os
import sys
import logging
import uuid
import pandas as pd

from typing import List, Dict, Any, Tuple
from datetime import datetime
import mlflow
import mlflow.pyfunc

try:
    from outage_predictor.utils import config
    from langchain_chroma import Chroma
    from chromadb.config import Settings
    from langchain_core.embeddings import Embeddings

    from langchain_core.documents import Document
    from langchain_community.embeddings import SentenceTransformerEmbeddings

except ImportError as e:
    print(f"FATAL ERROR: Could not import project modules or RAG libraries: {e}")
    sys.exit(1)

logger = logging.getLogger('outage_predictor_rag_service')

# --- Configuration ---
C_DATA = config.get('data', {})
C_MLOPS = config.get('mlops', {})
ALERT_THRESHOLD = C_MLOPS.get('ALERT_THRESHOLD', 0.5)

# --- RAG Configuration ---
DB_PATH = C_DATA.get('DB_PATH', 'data/log_outage_data.db')
BASE_EMBEDDING_MODEL = C_MLOPS.get('BASE_EMBEDDING_MODEL_FOR_RAG')
FINETUNED_EMBEDDING_MODEL_URI = f"models:/{C_MLOPS.get('EMBEDDING_MODEL_REGISTRY_NAME')}@Production"
CHROMA_COLLECTION_NAME = "outage_logs"
# Get DB suffixes
SUFFIX_BASE = C_MLOPS.get('RAG_DB_SUFFIX_BASE')
SUFFIX_FINETUNED = C_MLOPS.get('RAG_DB_SUFFIX_FINETUNED')

# --- MLflow Embedding Wrapper ---
class MlflowPyfuncEmbedding(Embeddings):
    def __init__(self, pyfunc_model): self.model = pyfunc_model
    def embed_documents(self, texts: List[str]) -> List[List[float]]: return self.model.predict(texts)
    def embed_query(self, text: str) -> List[float]: return self.model.predict([text])[0]


vector_store_base = None
vector_store_finetuned = None

def _load_vector_store_internal(use_finetuned: bool, db_suffix: str):
    """Internal function to load a specific vector store (ChromaDB instance)."""
    global vector_store_base, vector_store_finetuned
    current_store = vector_store_finetuned if use_finetuned else vector_store_base
    model_type = "Fine-Tuned" if use_finetuned else "Base"

    if current_store is None:
        try:
            chroma_persist_dir = os.path.join(os.path.dirname(DB_PATH), f"chroma_db{db_suffix}")
            if not os.path.exists(chroma_persist_dir):
                logger.error(f"ChromaDB ({model_type}) not found at {chroma_persist_dir}.")
                return "ERROR"

            # Load appropriate embedding model
            if use_finetuned:
                logger.info(f"Loading FINE-TUNED embedding model from MLflow: {FINETUNED_EMBEDDING_MODEL_URI}...")
                mlflow.set_tracking_uri(C_MLOPS.get('MLFLOW_TRACKING_URI'))
                model_pyfunc = mlflow.pyfunc.load_model(model_uri=FINETUNED_EMBEDDING_MODEL_URI)
                embeddings = MlflowPyfuncEmbedding(model_pyfunc)
            else:
                logger.info(f"Loading BASE embedding model: {BASE_EMBEDDING_MODEL}...")
                embeddings = SentenceTransformerEmbeddings(model_name=BASE_EMBEDDING_MODEL)

            logger.info(f"Loading ChromaDB ({model_type}) vector store...")
            client_settings = Settings(anonymized_telemetry=False)
            db = Chroma( # This is the vector store object
                persist_directory=chroma_persist_dir,
                embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION_NAME,
                client_settings=client_settings
            )
            logger.info(f"Vector Store ({model_type}) loaded successfully.")

            # Store in the correct cache
            if use_finetuned: vector_store_finetuned = db
            else: vector_store_base = db
            return db

        except Exception as e:
            logger.error(f"Error loading Vector Store ({model_type}): {e}", exc_info=True)
            if use_finetuned: vector_store_finetuned = "ERROR"
            else: vector_store_base = "ERROR"
            return "ERROR"
    else:
        return current_store # Return cached version


# Public function to load the BASE vector store
def load_vector_store_base():
    return _load_vector_store_internal(use_finetuned=False, db_suffix=SUFFIX_BASE)

# Public function to load the FINE-TUNED vector store
def load_vector_store_finetuned():
    return _load_vector_store_internal(use_finetuned=True, db_suffix=SUFFIX_FINETUNED)

# --- RAG Model Caching ---
# Separate caches for base and fine-tuned retrievers
rag_retriever_base = None
rag_retriever_finetuned = None

def _load_retriever_internal(use_finetuned: bool, db_suffix: str, k_neighbors: int):
    """Internal function to load a specific retriever."""
    global rag_retriever_base, rag_retriever_finetuned
    current_retriever = rag_retriever_finetuned if use_finetuned else rag_retriever_base
    model_type = "Fine-Tuned" if use_finetuned else "Base"

    if current_retriever is None:
        try:
            chroma_persist_dir = os.path.join(os.path.dirname(DB_PATH), f"chroma_db{db_suffix}")
            if not os.path.exists(chroma_persist_dir):
                logger.error(f"ChromaDB ({model_type}) not found at {chroma_persist_dir}.")
                return "ERROR"

            # Load appropriate embedding model
            if use_finetuned:
                logger.info(f"Loading FINE-TUNED embedding model from MLflow: {FINETUNED_EMBEDDING_MODEL_URI}...")
                mlflow.set_tracking_uri(C_MLOPS.get('MLFLOW_TRACKING_URI'))
                model_pyfunc = mlflow.pyfunc.load_model(model_uri=FINETUNED_EMBEDDING_MODEL_URI)
                embeddings = MlflowPyfuncEmbedding(model_pyfunc)
            else:
                logger.info(f"Loading BASE embedding model: {BASE_EMBEDDING_MODEL}...")
                embeddings = SentenceTransformerEmbeddings(model_name=BASE_EMBEDDING_MODEL)

            logger.info(f"Loading ChromaDB ({model_type}) vector store...")
            client_settings = Settings(anonymized_telemetry=False)
            db = Chroma(
                persist_directory=chroma_persist_dir,
                embedding_function=embeddings,
                collection_name=CHROMA_COLLECTION_NAME,
                client_settings=client_settings
            )

            retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k_neighbors})
            logger.info(f"RAG retriever ({model_type}) loaded successfully (k={k_neighbors}).")

            # Store in the correct cache
            if use_finetuned: rag_retriever_finetuned = retriever
            else: rag_retriever_base = retriever
            return retriever

        except Exception as e:
            logger.error(f"Error loading RAG retriever ({model_type}): {e}", exc_info=True)
            if use_finetuned: rag_retriever_finetuned = "ERROR"
            else: rag_retriever_base = "ERROR"
            return "ERROR"
    else:
        return current_retriever # Return cached version


# Public function to load the BASE retriever
def load_rag_retriever_base(k_neighbors: int = 5):
    return _load_retriever_internal(use_finetuned=False, db_suffix=SUFFIX_BASE, k_neighbors=k_neighbors)

# Public function to load the FINE-TUNED retriever
def load_rag_retriever_finetuned(k_neighbors: int = 5):
    return _load_retriever_internal(use_finetuned=True, db_suffix=SUFFIX_FINETUNED, k_neighbors=k_neighbors)


def format_log_list_to_string(log_data_list: List[Dict[str, Any]]) -> str:
    # ... (Unchanged)
    return "\n".join(
        f"Severity: {log.get('Severity')} Component: {log.get('Component')} Message: {log.get('Message')}"
        for log in log_data_list
    )

def _predict_realtime_rag_internal(vector_store, model_type: str, log_data_list: List[Dict[str, Any]], k_neighbors: int) -> Dict[str, Any]:
    """Internal prediction logic using a specific vector store."""
    correlation_id = str(uuid.uuid4())

    if vector_store == "ERROR" or vector_store is None:
        return {"error": f"RAG Vector Store ({model_type}) is not loaded.", "correlationid": correlation_id}

    try:
        query_string = format_log_list_to_string(log_data_list)

        # Use similarity_search_with_score to get scores
        # Returns a list of (Document, score) tuples
        # Note: Chroma uses cosine similarity *distance* by default (0=identical, higher=less similar)
        # We will convert this to a similarity score (1=identical, lower=less similar)
        similar_docs_with_scores: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(
            query_string,
            k=k_neighbors
        )

        if not similar_docs_with_scores:
            return {"error": "RAG search returned no similar documents.", "correlationid": correlation_id}

        outage_count = 0
        related_events = []
        explanation = (f"Prediction based on the {len(similar_docs_with_scores)} most semantically similar "
                       f"historical log windows found using the {model_type} embedding model:")

        for doc, score in similar_docs_with_scores:
            was_outage = doc.metadata.get("Outage_Flag", 0)
            if was_outage == 1:
                outage_count += 1

            # Convert distance score to similarity (assuming score is cosine distance 0 to 2)
            # Similarity = 1 - (Distance / 2) is a simple conversion, max 1.
            # Or just use 1 - distance if scores are 0-1 range. Chroma's default L2 is 0-inf, Cosine 0-2.
            # Let's use 1 - score for simplicity, assuming score is roughly normalized distance 0-1 range.
            # A more robust approach might analyze score distribution.
            similarity_score = 1.0 - score # Higher is better now

            related_events.append({
                "log_snippet": doc.page_content,
                "end_timestamp": doc.metadata.get("end_timestamp"),
                "was_outage_flag": was_outage,
                "similarity_score": round(similarity_score, 4) # Add the score
            })

        outage_probability = outage_count / len(similar_docs_with_scores)
        alert_status = "ALERT: Outage Imminent" if outage_probability > ALERT_THRESHOLD else "Normal Operations"

        logger.info(f"RAG Prediction ({model_type}): Found {outage_count}/{len(similar_docs_with_scores)} similar outages. Prob: {outage_probability:.4f}")

        return {
            "prediction_probability": float(outage_probability),
            "alert_status": alert_status,
            "correlationid": correlation_id,
            "prediction_timestamp": datetime.now().isoformat(),
            "explanation": explanation, # Add explanation
            "related_events": related_events
        }

    except Exception as e:
        logger.error(f"Prediction inference failed during RAG search ({model_type}): {e}", exc_info=True)
        return {"error": f"Prediction failed during RAG search ({model_type}): {e}", "correlationid": correlation_id}

# Public function for BASE RAG prediction
def predict_realtime_rag_base(log_data_list: List[Dict[str, Any]], k_neighbors: int = 5) -> Dict[str, Any]:
    vector_store = load_vector_store_base() # Load the store
    return _predict_realtime_rag_internal(vector_store, "Base", log_data_list, k_neighbors)

# Public function for FINE-TUNED RAG prediction
def predict_realtime_rag_finetuned(log_data_list: List[Dict[str, Any]], k_neighbors: int = 5) -> Dict[str, Any]:
    vector_store = load_vector_store_finetuned() # Load the store
    return _predict_realtime_rag_internal(vector_store, "Fine-Tuned", log_data_list, k_neighbors)

"""
def _predict_realtime_rag_internal(retriever, model_type: str, log_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    correlation_id = str(uuid.uuid4())

    if retriever == "ERROR" or retriever is None:
        return {"error": f"RAG Model ({model_type}) is not loaded.", "correlationid": correlation_id}

    try:
        query_string = format_log_list_to_string(log_data_list)
        similar_docs = retriever.invoke(query_string)
        # ... (rest of prediction logic is the same)
        if not similar_docs: return {"error": "RAG search returned no similar documents.", "correlationid": correlation_id}
        outage_count = 0
        related_events = []
        for doc in similar_docs:
            was_outage = doc.metadata.get("Outage_Flag", 0)
            if was_outage == 1: outage_count += 1
            related_events.append({"log_snippet": doc.page_content, "end_timestamp": doc.metadata.get("end_timestamp"), "was_outage_flag": was_outage})
        outage_probability = outage_count / len(similar_docs)
        alert_status = "ALERT: Outage Imminent" if outage_probability > ALERT_THRESHOLD else "Normal Operations"
        logger.info(f"RAG Prediction ({model_type}): Found {outage_count}/{len(similar_docs)} similar outages. Prob: {outage_probability:.4f}")
        return {"prediction_probability": float(outage_probability), "alert_status": alert_status, "correlationid": correlation_id, "prediction_timestamp": datetime.now().isoformat(), "related_events": related_events}

    except Exception as e:
        logger.error(f"Prediction inference failed during RAG search ({model_type}): {e}", exc_info=True)
        return {"error": f"Prediction failed during RAG search ({model_type}): {e}", "correlationid": correlation_id}

# Public function for BASE RAG prediction
def predict_realtime_rag_base(log_data_list: List[Dict[str, Any]], k_neighbors: int = 5) -> Dict[str, Any]:
    retriever = load_rag_retriever_base(k_neighbors=k_neighbors)
    return _predict_realtime_rag_internal(retriever, "Base", log_data_list)

# Public function for FINE-TUNED RAG prediction
def predict_realtime_rag_finetuned(log_data_list: List[Dict[str, Any]], k_neighbors: int = 5) -> Dict[str, Any]:
    retriever = load_rag_retriever_finetuned(k_neighbors=k_neighbors)
    return _predict_realtime_rag_internal(retriever, "Fine-Tuned", log_data_list)

"""