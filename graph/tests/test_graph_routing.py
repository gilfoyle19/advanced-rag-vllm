import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import graph.graph as graph_module
from graph.config import MAX_GENERATION_ATTEMPTS, MAX_WEB_SEARCH_ATTEMPTS
from graph.nodes.fallback import fallback


class StaticGrader:
    def __init__(self, result: bool):
        self.result = result

    def invoke(self, _input):
        return SimpleNamespace(binary_score=self.result)


def graph_state(**overrides):
    state = {
        "question": "What is the limit?",
        "documents": [Document(page_content="The limit is 10 bar.")],
        "generation": "The limit is 12 bar.",
        "generation_attempts": 1,
        "web_search_attempts": 0,
    }
    state.update(overrides)
    return state


def test_unsupported_generation_retries_below_limit(monkeypatch):
    monkeypatch.setattr(
        graph_module, "hallucination_grader", StaticGrader(result=False)
    )
    state = graph_state(generation_attempts=MAX_GENERATION_ATTEMPTS - 1)

    assert (
        graph_module.grade_generation_grounded_in_documents_and_question(state)
        == "not supported"
    )


def test_unsupported_generation_stops_at_limit(monkeypatch):
    monkeypatch.setattr(
        graph_module, "hallucination_grader", StaticGrader(result=False)
    )
    state = graph_state(generation_attempts=MAX_GENERATION_ATTEMPTS)

    assert (
        graph_module.grade_generation_grounded_in_documents_and_question(state)
        == "give up"
    )


def test_unhelpful_generation_searches_web_only_once(monkeypatch):
    monkeypatch.setattr(graph_module, "hallucination_grader", StaticGrader(result=True))
    monkeypatch.setattr(graph_module, "answer_grader", StaticGrader(result=False))

    before_web = graph_state(web_search_attempts=MAX_WEB_SEARCH_ATTEMPTS - 1)
    after_web = graph_state(web_search_attempts=MAX_WEB_SEARCH_ATTEMPTS)

    assert (
        graph_module.grade_generation_grounded_in_documents_and_question(before_web)
        == "not useful"
    )
    assert (
        graph_module.grade_generation_grounded_in_documents_and_question(after_web)
        == "give up"
    )


def test_local_document_question_does_not_use_web_when_retrieval_fails():
    state = graph_state(
        question="What is the research goal stated in Chapter 1?",
        documents=[],
        web_search=True,
    )

    assert graph_module.decide_to_generate(state) == graph_module.FALLBACK


def test_local_document_question_does_not_use_web_when_answer_not_useful(monkeypatch):
    monkeypatch.setattr(graph_module, "hallucination_grader", StaticGrader(result=True))
    monkeypatch.setattr(graph_module, "answer_grader", StaticGrader(result=False))
    state = graph_state(question="What is the research goal stated in Chapter 1?")

    assert (
        graph_module.grade_generation_grounded_in_documents_and_question(state)
        == "give up"
    )


def test_fallback_replaces_unsupported_generation():
    result = fallback(graph_state())

    assert "could not produce an answer" in result["generation"]
