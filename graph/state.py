from typing import List, NotRequired, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    question:question
    generation: LLM generation
    web_search: whether to add search or not
    documents: list of documents"""

    question: str
    generation: NotRequired[str]
    generation_attempts: NotRequired[int]
    web_search: NotRequired[bool]
    web_search_attempts: NotRequired[int]
    documents: NotRequired[List[Document]]
    retrieved_documents: NotRequired[List[Document]]
