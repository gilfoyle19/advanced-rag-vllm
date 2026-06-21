from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()


@lru_cache(maxsize=1)
def get_web_search_tool() -> TavilySearch:
    return TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = list(state.get("documents") or [])
    web_search_attempts = state.get("web_search_attempts", 0) + 1

    tavily_results = get_web_search_tool().invoke({"query": question})["results"]
    for result in tavily_results:
        content = result.get("content")
        if not content:
            continue
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": result.get("url", "web_search"),
                    "title": result.get("title"),
                    "file_type": "web",
                },
            )
        )
    return {
        "documents": documents,
        "question": question,
        "web_search": True,
        "web_search_attempts": web_search_attempts,
    }


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
