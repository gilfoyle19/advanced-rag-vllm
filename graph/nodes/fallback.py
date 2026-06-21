from typing import Any, Dict

from graph.state import GraphState


def fallback(state: GraphState) -> Dict[str, Any]:
    print("---STOP: NO GROUNDED ANSWER AFTER BOUNDED RETRIES---")
    return {
        "documents": state.get("documents", []),
        "question": state["question"],
        "generation": (
            "I could not produce an answer that was sufficiently supported by the "
            "available local or web sources."
        ),
    }
