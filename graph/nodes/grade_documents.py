from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for index, d in enumerate(documents, start=1):
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = str(score.binary_score).strip().lower()
        relevant = grade in {"yes", "true", "1"}
        source = d.metadata.get("source", "unknown") if d.metadata else "unknown"
        page = d.metadata.get("page") if d.metadata else None
        page_label = page + 1 if isinstance(page, int) else "unknown"
        decision = "RELEVANT" if relevant else "NOT RELEVANT"
        print(
            f"---GRADE: DOCUMENT {index} {decision} "
            f"(source={source}, page={page_label})---"
        )
        if relevant:
            filtered_docs.append(d)
    web_search = not filtered_docs
    return {
        "documents": filtered_docs,
        "graded_documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }
