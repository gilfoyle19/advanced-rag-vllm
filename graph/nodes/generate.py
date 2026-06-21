from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def format_documents(documents) -> str:
    formatted = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
        formatted.append(f"[{index}] Source: {source}\n{doc.page_content}")
    return "\n\n".join(formatted)


def generate(state: GraphState) -> Dict[str, Any]:
    generation_attempts = state.get("generation_attempts", 0) + 1
    print(f"---GENERATE (ATTEMPT {generation_attempts})---")
    question = state["question"]
    documents = state.get("documents", [])

    generation = generation_chain.invoke(
        {"context": format_documents(documents), "question": question}
    )
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "generation_attempts": generation_attempts,
    }
