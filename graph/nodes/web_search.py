from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = list(state.get("documents") or [])

    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(
        page_content=joined_tavily_result,
        metadata={"source": "web_search", "file_type": "web"},
    )
    documents.append(web_results)
    return {"documents": documents, "question": question, "web_search": True}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
