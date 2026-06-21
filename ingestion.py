import argparse
import hashlib
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LOCAL_DOCS_DIR,
    RETRIEVAL_K,
)

DEFAULT_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

SUPPORTED_LOCAL_EXTENSIONS = {".pdf", ".tex", ".txt"}


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": RETRIEVAL_K})


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def clean_latex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(
        r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]*)\}", r"\2\n", text
    )
    text = re.sub(r"\\(begin|end)\{[^}]*\}", "\n", text)
    text = re.sub(r"\\(cite|ref|label|url)\*?(\[[^]]*\])?\{([^}]*)\}", r"\3", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?(\{[^}]*\})?", " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_plain_text(path: Path) -> Document:
    return Document(
        page_content=path.read_text(encoding="utf-8", errors="ignore"),
        metadata={"source": str(path), "file_type": path.suffix.lower().lstrip(".")},
    )


def load_tex(path: Path) -> Document:
    return Document(
        page_content=clean_latex(path.read_text(encoding="utf-8", errors="ignore")),
        metadata={"source": str(path), "file_type": "tex"},
    )


def load_pdf(path: Path) -> list[Document]:
    docs = PyPDFLoader(str(path)).load()
    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadata["file_type"] = "pdf"
    return docs


def iter_local_paths(docs_dir: Path) -> Iterable[Path]:
    if not docs_dir.exists():
        return []

    return (
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_LOCAL_EXTENSIONS
    )


def load_local_documents(docs_dir: Path = LOCAL_DOCS_DIR) -> list[Document]:
    documents: list[Document] = []

    for path in iter_local_paths(docs_dir):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(load_pdf(path))
        elif suffix == ".tex":
            documents.append(load_tex(path))
        elif suffix == ".txt":
            documents.append(load_plain_text(path))

    return documents


def load_web_documents(urls: Sequence[str]) -> list[Document]:
    from langchain_community.document_loaders import WebBaseLoader

    docs = [WebBaseLoader(url).load() for url in urls]
    return [item for sublist in docs for item in sublist]


def split_documents(documents: Sequence[Document]) -> list[Document]:
    return get_text_splitter().split_documents(list(documents))


def document_id(doc: Document, index: int) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", ""))
    payload = f"{source}:{page}:{index}:{doc.page_content}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ingest_documents(documents: Sequence[Document], rebuild: bool = False) -> int:
    persist_dir = Path(CHROMA_PERSIST_DIRECTORY)
    if rebuild and persist_dir.exists():
        get_retriever.cache_clear()
        shutil.rmtree(persist_dir)

    splits = split_documents(documents)
    if not splits:
        return 0

    vectorstore = get_vectorstore()
    ids = [document_id(doc, index) for index, doc in enumerate(splits)]
    vectorstore.add_documents(splits, ids=ids)
    get_retriever.cache_clear()
    return len(splits)


def build_documents(
    docs_dir: Path = LOCAL_DOCS_DIR,
    include_web: bool = False,
    urls: Sequence[str] = DEFAULT_URLS,
) -> list[Document]:
    documents = load_local_documents(docs_dir)
    if include_web:
        documents.extend(load_web_documents(urls))
    return documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest local and optional web documents into Chroma."
    )
    parser.add_argument("--docs-dir", type=Path, default=LOCAL_DOCS_DIR)
    parser.add_argument(
        "--include-web", action="store_true", help="Also ingest the default seed URLs."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing Chroma store before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = build_documents(docs_dir=args.docs_dir, include_web=args.include_web)
    chunk_count = ingest_documents(documents, rebuild=args.rebuild)
    print(f"Ingested {len(documents)} documents as {chunk_count} chunks.")


if __name__ == "__main__":
    main()
