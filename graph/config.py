import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "local-vllm")
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen2.5-7b-instruct-q4_k_m")

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag-chroma")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./.chroma")
LOCAL_DOCS_DIR = Path(os.getenv("LOCAL_DOCS_DIR", "documents"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "750"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
MAX_GENERATION_ATTEMPTS = int(os.getenv("MAX_GENERATION_ATTEMPTS", "2"))
MAX_WEB_SEARCH_ATTEMPTS = int(os.getenv("MAX_WEB_SEARCH_ATTEMPTS", "1"))

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than zero.")
if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")
if RETRIEVAL_K <= 0:
    raise ValueError("RETRIEVAL_K must be greater than zero.")
if MAX_GENERATION_ATTEMPTS <= 0:
    raise ValueError("MAX_GENERATION_ATTEMPTS must be greater than zero.")
if MAX_WEB_SEARCH_ATTEMPTS < 0:
    raise ValueError("MAX_WEB_SEARCH_ATTEMPTS must be non-negative.")
