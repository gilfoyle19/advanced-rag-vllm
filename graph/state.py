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
    web_search: NotRequired[bool]
    documents: NotRequired[List[Document]]
