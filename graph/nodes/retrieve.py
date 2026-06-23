from typing import Any, Dict

import re

from langchain_core.documents import Document

from graph.config import RETRIEVAL_K
from graph.local_questions import is_local_document_question
from graph.state import GraphState
from ingestion import get_retriever, get_vectorstore


STOPWORDS = {
    "about",
    "according",
    "chapter",
    "document",
    "does",
    "from",
    "into",
    "question",
    "report",
    "section",
    "stated",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def lexical_score(question: str, document: str) -> int:
    query_terms = tokenize(question)
    if not query_terms:
        return 0
    document_lower = document.lower()
    document_terms = tokenize(document)
    score = 2 * len(query_terms & document_terms)
    for phrase in re.findall(r"[a-z0-9]+(?:\s+[a-z0-9]+){1,3}", question.lower()):
        if phrase in document_lower:
            score += 3
    if "research goal" in question.lower() and "goal of this research" in document_lower:
        score += 8
    return score


def lexical_retrieve(question: str, limit: int = 3) -> list[Document]:
    stored = get_vectorstore().get(include=["documents", "metadatas"])
    scored: list[tuple[int, int, Document]] = []
    for index, (content, metadata) in enumerate(
        zip(stored.get("documents", []), stored.get("metadatas", []))
    ):
        if not content:
            continue
        score = lexical_score(question, content)
        if score > 0:
            scored.append(
                (
                    score,
                    index,
                    Document(page_content=content, metadata=metadata or {}),
                )
            )
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [document for _, _, document in scored[:limit]]


def document_key(document: Document) -> tuple[Any, Any, int]:
    metadata = document.metadata or {}
    return (
        metadata.get("source"),
        metadata.get("page"),
        hash(document.page_content),
    )


def merge_documents(
    primary: list[Document], supplemental: list[Document]
) -> list[Document]:
    merged: list[Document] = []
    seen: set[tuple[Any, Any, int]] = set()
    for document in [*supplemental, *primary]:
        key = document_key(document)
        if key not in seen:
            seen.add(key)
            merged.append(document)
    return merged[:RETRIEVAL_K]


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves relevant documents based on the question in the state.
    Input: question, str
    Output: dict with question and retrieved documents
    """
    question = state["question"]
    documents = list(get_retriever().invoke(question))
    if is_local_document_question(question):
        documents = merge_documents(documents, lexical_retrieve(question))
    return {
        "documents": documents,
        "retrieved_documents": documents,
        "question": question,
    }
