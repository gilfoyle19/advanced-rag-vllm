from argparse import Namespace

import pytest
from langchain_core.documents import Document

from evaluation.run_evals import (
    aggregate_results,
    classify_route,
    percentile,
    retrieval_metrics,
)


def document(page: int) -> Document:
    return Document(page_content=f"page {page}", metadata={"page": page})


def local_case():
    return {
        "answerability": "local",
        "evidence_groups": [
            {"page_indices": [5]},
            {"page_indices": [8, 9]},
        ],
    }


def test_retrieval_metrics_cover_evidence_groups():
    metrics = retrieval_metrics([document(5), document(9), document(99)], local_case())

    assert metrics == pytest.approx(
        {
            "evidence_group_recall": 1.0,
            "hit": 1.0,
            "mrr": 1.0,
            "context_precision_proxy": 2 / 3,
        }
    )


def test_retrieval_metrics_are_not_scored_for_external_cases():
    case = {"answerability": "external", "evidence_groups": []}

    assert retrieval_metrics([document(5)], case) is None


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert percentile([], 0.95) is None


def test_classify_route_uses_judged_abstention():
    case = {"expected_route": "abstain"}
    record = {
        "web_search_used": True,
        "judgment": {"abstention_correct": True},
    }

    assert classify_route(case, record) == ("abstain", True)


def test_retrieval_only_aggregate_scores_25_points():
    record = {
        "id": "case-1",
        "run_index": 1,
        "answerability": "local",
        "success": True,
        "latency_seconds": 1.0,
        "pipeline_llm_calls": 0,
        "retrieval_metrics": {
            "evidence_group_recall": 1.0,
            "hit": 1.0,
            "mrr": 1.0,
            "context_precision_proxy": 1.0,
        },
        "graded_retrieval_metrics": None,
    }
    args = Namespace(mode="retrieval")

    summary = aggregate_results([record], args)

    assert summary["points"] == 25.0
    assert summary["maximum_evaluated"] == 25.0
    assert summary["normalized_score"] == 100.0
    assert summary["metrics"]["graded_evidence_group_recall"] is None
