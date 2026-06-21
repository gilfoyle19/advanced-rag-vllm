from typing import Any, Dict
from graph.state import GraphState
from ingestion import get_retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieves relevant documents based on the question in the state.
    Input: question, str
    Output: dict with question and retrieved documents
    """
    question = state["question"]
    documents = get_retriever().invoke(question)
    return {
        "documents": documents,
        "retrieved_documents": documents,
        "question": question,
    }
