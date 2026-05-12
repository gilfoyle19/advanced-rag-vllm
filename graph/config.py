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
