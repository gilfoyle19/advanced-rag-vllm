import hashlib
import re
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.documents import Document

from graph.config import CHROMA_PERSIST_DIRECTORY, LOCAL_DOCS_DIR
from ingestion import (
    SUPPORTED_LOCAL_EXTENSIONS,
    ingest_documents,
    load_pdf,
    load_plain_text,
    load_tex,
)

UPLOAD_DIRECTORY = LOCAL_DOCS_DIR / "streamlit_uploads"
GRAPH_RECURSION_LIMIT = 12


def safe_upload_path(filename: str, content: bytes) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_LOCAL_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'none'}")

    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).stem).strip("-._")
    stem = stem or "document"
    digest = hashlib.sha256(content).hexdigest()[:12]
    return UPLOAD_DIRECTORY / f"{stem}-{digest}{suffix}"


def save_upload(upload: Any) -> tuple[Path, bool]:
    content = upload.getvalue()
    path = safe_upload_path(upload.name, content)
    path.parent.mkdir(parents=True, exist_ok=True)

    created = not path.exists()
    if created:
        path.write_bytes(content)
    return path, created


def load_uploaded_document(path: Path) -> list[Document]:
    if path.suffix.lower() == ".pdf":
        return load_pdf(path)
    if path.suffix.lower() == ".tex":
        return [load_tex(path)]
    if path.suffix.lower() == ".txt":
        return [load_plain_text(path)]
    raise ValueError(f"Unsupported file type: {path.suffix}")


def index_uploads(uploads: list[Any]) -> tuple[int, int, list[Path]]:
    chunk_count = 0
    new_file_count = 0
    saved_paths: list[Path] = []

    for upload in uploads:
        path, created = save_upload(upload)
        saved_paths.append(path)
        new_file_count += int(created)
        chunk_count += ingest_documents(load_uploaded_document(path))

    return new_file_count, chunk_count, saved_paths


def source_label(document: Document) -> str:
    metadata = document.metadata or {}
    source = str(metadata.get("source", "Unknown source"))
    label = "Web search" if source == "web_search" else Path(source).name
    page = metadata.get("page")
    if isinstance(page, int):
        label += f", page {page + 1}"
    return label


def unique_sources(documents: list[Document]) -> list[Document]:
    seen: set[tuple[str, Any, str]] = set()
    result: list[Document] = []
    for document in documents:
        metadata = document.metadata or {}
        key = (
            str(metadata.get("source", "")),
            metadata.get("page"),
            document.page_content,
        )
        if key not in seen:
            seen.add(key)
            result.append(document)
    return result


def render_sources(documents: list[Document], web_search_used: bool) -> None:
    documents = unique_sources(documents)
    title = f"Sources ({len(documents)})"
    if web_search_used:
        title += " - web search used"

    with st.expander(title):
        if not documents:
            st.caption("No source documents were returned.")
        for document in documents:
            st.markdown(f"**{source_label(document)}**")
            preview = document.page_content.strip()
            if len(preview) > 900:
                preview = f"{preview[:900].rstrip()}..."
            st.text(preview)


def initialize_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("ingestion_report", None)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Document ingestion")
        uploads = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "tex"],
            accept_multiple_files=True,
            help="Files are stored locally and added to the shared Chroma collection.",
        )

        if st.button(
            "Ingest documents",
            type="primary",
            disabled=not uploads,
            use_container_width=True,
        ):
            try:
                with st.spinner("Loading, chunking, embedding, and indexing..."):
                    files, chunks, paths = index_uploads(list(uploads))
                st.session_state.ingestion_report = {
                    "files": len(paths),
                    "new_files": files,
                    "chunks": chunks,
                }
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

        report = st.session_state.ingestion_report
        if report:
            st.success(
                f"Indexed {report['files']} file(s) as {report['chunks']} chunks "
                f"({report['new_files']} newly saved)."
            )

        index_path = Path(CHROMA_PERSIST_DIRECTORY)
        if index_path.exists():
            st.caption(f"Index ready: `{index_path}`")
        else:
            st.caption("No local Chroma index found yet.")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "Demo mode: uploads and the vector index are shared by users of this process. "
            "Each question is retrieved independently; visible chat history is not model memory."
        )


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(
                    message.get("documents", []),
                    message.get("web_search_used", False),
                )


def handle_chat() -> None:
    question = st.chat_input("Ask a question about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            from graph.graph import app as rag_graph

            with st.spinner("Retrieving, grading, and generating..."):
                result = rag_graph.invoke(
                    {"question": question},
                    config={"recursion_limit": GRAPH_RECURSION_LIMIT},
                )
            answer = result.get("generation") or "The pipeline returned no answer."
            documents = list(result.get("documents") or [])
            web_search_used = bool(result.get("web_search", False))
            st.markdown(answer)
            render_sources(documents, web_search_used)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "documents": documents,
                    "web_search_used": web_search_used,
                }
            )
        except Exception as exc:
            message = f"The RAG pipeline failed: {exc}"
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": message, "documents": []}
            )


def main() -> None:
    st.set_page_config(page_title="Advanced RAG Demo", layout="wide")
    initialize_state()
    render_sidebar()

    st.title("Advanced RAG Demo")
    st.write(
        "Upload PDF, TXT, or LaTeX documents, ingest them into Chroma, and chat "
        "through the complete LangGraph and vLLM pipeline."
    )
    render_history()
    handle_chat()


if __name__ == "__main__":
    main()
