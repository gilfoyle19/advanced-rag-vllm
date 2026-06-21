#!/usr/bin/env python3
"""Run retrieval and end-to-end evaluations for the DESMI RAG benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pypdf import PdfReader

EXPECTED_SOURCE_SHA256 = (
    "B88E454110CFE5FE8E990C701960DA8CBE59F53D48887454502D226A25DBCB7A"
)
FALLBACK_PREFIX = "I could not produce an answer"


class EvaluationTimeout(TimeoutError):
    pass


class LLMCallCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        self.count = 0

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self.count += len(messages)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.count += len(prompts)


class AnswerJudgment(BaseModel):
    required_fact_coverage: float = Field(ge=0.0, le=1.0)
    faithfulness: Literal[0.0, 0.5, 1.0]
    directness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    contradiction: bool
    critical_failure: bool
    abstention_correct: bool
    web_grounded: bool
    rationale: str


class EvidenceProvider:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        self._reader: PdfReader | None = None

    @property
    def reader(self) -> PdfReader:
        if self._reader is None:
            self._reader = PdfReader(str(self.source_path))
        return self._reader

    @lru_cache(maxsize=256)
    def page_text(self, page_index: int) -> str:
        if page_index < 0 or page_index >= len(self.reader.pages):
            return ""
        return self.reader.pages[page_index].extract_text() or ""

    def evidence_for(self, case: dict[str, Any], max_chars: int = 7000) -> str:
        parts: list[str] = []
        seen: set[int] = set()
        for group in case.get("evidence_groups", []):
            for page_index in group.get("page_indices", []):
                if page_index in seen:
                    continue
                seen.add(page_index)
                parts.append(
                    f"[PDF page {page_index + 1}]\n{self.page_text(page_index).strip()}"
                )
        return clip_text("\n\n".join(parts), max_chars)


@contextmanager
def time_limit(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handle_timeout(_signum, _frame):
        raise EvaluationTimeout(f"Evaluation exceeded {seconds:g} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def clip_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n[truncated]"


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def serialize_document(document: Document) -> dict[str, Any]:
    return {
        "page_content": document.page_content,
        "metadata": json_safe(document.metadata or {}),
    }


def document_page_index(document: Document) -> int | None:
    page = (document.metadata or {}).get("page")
    return page if isinstance(page, int) else None


def relevant_page_indices(case: dict[str, Any]) -> set[int]:
    return {
        page
        for group in case.get("evidence_groups", [])
        for page in group.get("page_indices", [])
    }


def retrieval_metrics(
    documents: Sequence[Document], case: dict[str, Any]
) -> dict[str, float] | None:
    groups = case.get("evidence_groups", [])
    if case.get("answerability") != "local" or not groups:
        return None

    pages = [document_page_index(document) for document in documents]
    relevant_pages = relevant_page_indices(case)
    group_hits = [
        any(page in set(group.get("page_indices", [])) for page in pages)
        for group in groups
    ]
    relevant_ranks = [
        rank
        for rank, page in enumerate(pages, start=1)
        if page is not None and page in relevant_pages
    ]
    relevant_count = sum(page in relevant_pages for page in pages if page is not None)

    return {
        "evidence_group_recall": sum(group_hits) / len(group_hits),
        "hit": float(bool(relevant_ranks)),
        "mrr": 1.0 / min(relevant_ranks) if relevant_ranks else 0.0,
        "context_precision_proxy": (
            relevant_count / len(documents) if documents else 0.0
        ),
    }


def load_dataset(path: Path) -> list[dict[str, Any]]:
    required = {
        "id",
        "split",
        "question_type",
        "difficulty",
        "answerability",
        "expected_route",
        "safety_critical",
        "question",
        "reference_answer",
        "required_facts",
        "forbidden_claims",
        "evidence_groups",
    }
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        case = json.loads(line)
        missing = required - case.keys()
        if missing:
            raise ValueError(
                f"Dataset line {line_number} is missing: {', '.join(sorted(missing))}"
            )
        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate dataset id: {case['id']}")
        seen_ids.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError(f"Dataset is empty: {path}")
    return cases


def filter_cases(
    cases: Sequence[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    selected_ids = set(args.ids.split(",")) if args.ids else set()
    selected = [
        case
        for case in cases
        if (args.split == "all" or case["split"] == args.split)
        and (not selected_ids or case["id"] in selected_ids)
        and (not args.question_type or case["question_type"] in args.question_type)
        and (not args.safety_critical_only or case["safety_critical"])
    ]
    if selected_ids:
        missing = selected_ids - {case["id"] for case in selected}
        if missing:
            raise ValueError(
                f"Requested ids were not selected: {', '.join(sorted(missing))}"
            )
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No evaluation cases matched the supplied filters.")
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_snapshot() -> dict[str, Any]:
    def run(*command: str) -> str | None:
        try:
            return subprocess.run(
                command,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def check_model_server(base_url: str, api_key: str, timeout: float = 5.0) -> None:
    url = f"{base_url.rstrip('/')}/models"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Model health check returned HTTP {response.status}"
                )
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Cannot reach the vLLM model server at {url}. Start vLLM before "
            f"running pipeline evaluations. Original error: {exc}"
        ) from exc


def format_documents_for_judge(
    documents: Sequence[Document], max_chars: int = 6500
) -> str:
    parts: list[str] = []
    for index, document in enumerate(documents, 1):
        metadata = document.metadata or {}
        source = metadata.get("source", "unknown")
        page = metadata.get("page")
        location = f", page {page + 1}" if isinstance(page, int) else ""
        parts.append(
            f"[{index}] Source: {source}{location}\n{document.page_content.strip()}"
        )
    return clip_text("\n\n".join(parts), max_chars)


@lru_cache(maxsize=1)
def answer_judge_chain():
    from graph.chains.llm import get_llm

    parser = PydanticOutputParser(pydantic_object=AnswerJudgment)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict evaluator of a retrieval-augmented answer. Score only "
                "the answer shown. For required_fact_coverage, calculate the fraction of "
                "required facts correctly expressed, allowing valid paraphrases. For "
                "faithfulness use exactly 1.0 when every material claim is supported by "
                "the retrieved context, 0.5 for minor unsupported detail, and 0.0 for any "
                "material unsupported claim. Reference answers establish correctness but "
                "do not establish faithfulness. Set contradiction when the answer "
                "semantically endorses a forbidden claim. Set critical_failure only when "
                "the case is safety-critical and the answer gives dangerous contradictory "
                "advice or fabricates a critical numeric specification. Set "
                "abstention_correct only when the expected route is abstain and the answer "
                "clearly avoids inventing the missing information. Set web_grounded only "
                "when the expected route is web and identifiable retrieved web sources "
                "support the answer. Return only the required JSON.\n{format_instructions}",
            ),
            (
                "human",
                "Question:\n{question}\n\nExpected route: {expected_route}\n"
                "Safety critical: {safety_critical}\n\nReference answer:\n"
                "{reference_answer}\n\nRequired facts:\n{required_facts}\n\n"
                "Forbidden claims:\n{forbidden_claims}\n\nGround-truth evidence:\n"
                "{ground_truth_evidence}\n\nRetrieved context supplied to the answer:\n"
                "{retrieved_context}\n\nAnswer to evaluate:\n{answer}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | get_llm(max_tokens=800) | parser


def judge_answer(
    case: dict[str, Any],
    answer: str,
    documents: Sequence[Document],
    evidence_provider: EvidenceProvider,
) -> AnswerJudgment:
    return answer_judge_chain().invoke(
        {
            "question": case["question"],
            "expected_route": case["expected_route"],
            "safety_critical": case["safety_critical"],
            "reference_answer": case["reference_answer"],
            "required_facts": json.dumps(case["required_facts"]),
            "forbidden_claims": json.dumps(case["forbidden_claims"]),
            "ground_truth_evidence": evidence_provider.evidence_for(case),
            "retrieved_context": format_documents_for_judge(documents),
            "answer": answer,
        }
    )


def recheck_pipeline_graders(
    question: str, answer: str, documents: Sequence[Document]
) -> dict[str, Any]:
    from graph.chains.answer_grader import answer_grader
    from graph.chains.hallucination_grader import hallucination_grader
    from graph.nodes.generate import format_documents

    hallucination = hallucination_grader.invoke(
        {"documents": format_documents(documents), "generation": answer}
    )
    useful = answer_grader.invoke({"question": question, "generation": answer})
    return {
        "grounded": bool(hallucination.binary_score),
        "useful": bool(useful.binary_score),
    }


def classify_route(
    case: dict[str, Any], record: dict[str, Any]
) -> tuple[str, bool | None]:
    judgment = record.get("judgment")
    if case["expected_route"] == "abstain" and judgment:
        actual = (
            "abstain"
            if judgment["abstention_correct"]
            else ("web" if record.get("web_search_used") else "local")
        )
    else:
        actual = "web" if record.get("web_search_used") else "local"
    return actual, actual == case["expected_route"]


def run_case(
    case: dict[str, Any],
    run_index: int,
    args: argparse.Namespace,
    evidence_provider: EvidenceProvider,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": case["id"],
        "run_index": run_index,
        "split": case["split"],
        "question_type": case["question_type"],
        "difficulty": case["difficulty"],
        "answerability": case["answerability"],
        "expected_route": case["expected_route"],
        "safety_critical": case["safety_critical"],
        "question": case["question"],
    }
    counter = LLMCallCounter()
    raw_documents: list[Document] = []
    graded_documents: list[Document] = []
    final_documents: list[Document] = []
    answer = ""
    started = time.perf_counter()

    try:
        with time_limit(args.timeout):
            if args.mode == "retrieval":
                from ingestion import get_retriever

                raw_documents = list(get_retriever().invoke(case["question"]))
            else:
                from graph.graph import app as rag_graph

                result = rag_graph.invoke(
                    {"question": case["question"]},
                    config={
                        "callbacks": [counter],
                        "recursion_limit": args.recursion_limit,
                    },
                )
                raw_documents = list(result.get("retrieved_documents") or [])
                graded_documents = list(result.get("graded_documents") or [])
                final_documents = list(result.get("documents") or [])
                answer = result.get("generation") or ""
                record.update(
                    {
                        "answer": answer,
                        "generation_attempts": result.get("generation_attempts", 0),
                        "web_search_attempts": result.get("web_search_attempts", 0),
                        "web_search_used": bool(result.get("web_search", False)),
                        "bounded_fallback": answer.startswith(FALLBACK_PREFIX),
                    }
                )
    except EvaluationTimeout as exc:
        record.update({"error": str(exc), "timed_out": True})
    except Exception as exc:
        record.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "timed_out": False,
            }
        )

    record["latency_seconds"] = round(time.perf_counter() - started, 4)
    record["pipeline_llm_calls"] = counter.count
    record["retrieved_documents"] = [
        serialize_document(document) for document in raw_documents
    ]
    record["graded_documents"] = [
        serialize_document(document) for document in graded_documents
    ]
    record["final_documents"] = [
        serialize_document(document) for document in final_documents
    ]
    record["retrieval_metrics"] = retrieval_metrics(raw_documents, case)
    record["graded_retrieval_metrics"] = (
        retrieval_metrics(graded_documents, case) if args.mode != "retrieval" else None
    )

    if args.mode != "retrieval" and not record.get("error") and answer:
        if args.judge == "llm":
            judge_started = time.perf_counter()
            try:
                with time_limit(args.timeout):
                    judgment = judge_answer(
                        case, answer, final_documents, evidence_provider
                    ).model_dump()
                record["judgment"] = judgment
            except Exception as exc:
                record["judge_error"] = f"{type(exc).__name__}: {exc}"
            record["judge_latency_seconds"] = round(
                time.perf_counter() - judge_started, 4
            )

            try:
                with time_limit(args.timeout):
                    record["grader_recheck"] = recheck_pipeline_graders(
                        case["question"], answer, final_documents
                    )
            except Exception as exc:
                record["grader_recheck_error"] = f"{type(exc).__name__}: {exc}"

        actual_route, route_correct = classify_route(case, record)
        record["actual_route"] = actual_route
        record["route_correct"] = route_correct

    if args.mode != "retrieval" and "route_correct" not in record:
        record["actual_route"] = "error"
        record["route_correct"] = False
    record["success"] = bool(
        not record.get("error")
        and not record.get("timed_out")
        and (args.mode == "retrieval" or answer)
    )
    return record


def mean(values: Iterable[float | int | bool | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def percentile(values: Sequence[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile_value
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def component(name: str, points: float | None, maximum: float) -> dict[str, Any]:
    return {
        "name": name,
        "points": points,
        "maximum": maximum if points is not None else 0.0,
    }


def aggregate_results(
    records: Sequence[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    primary = [record for record in records if record["run_index"] == 1]
    local = [record for record in primary if record["answerability"] == "local"]
    components: list[dict[str, Any]] = []

    raw_group_recall = mean(
        (
            record.get("retrieval_metrics", {}).get("evidence_group_recall")
            if record.get("retrieval_metrics")
            else None
        )
        for record in local
    )
    raw_hit = mean(
        (
            record.get("retrieval_metrics", {}).get("hit")
            if record.get("retrieval_metrics")
            else None
        )
        for record in local
    )
    raw_mrr = mean(
        (
            record.get("retrieval_metrics", {}).get("mrr")
            if record.get("retrieval_metrics")
            else None
        )
        for record in local
    )
    raw_precision = mean(
        (
            record.get("retrieval_metrics", {}).get("context_precision_proxy")
            if record.get("retrieval_metrics")
            else None
        )
        for record in local
    )
    components.extend(
        [
            component(
                "Evidence-group recall",
                12 * raw_group_recall if raw_group_recall is not None else None,
                12,
            ),
            component("Retrieval hit", 4 * raw_hit if raw_hit is not None else None, 4),
            component("Retrieval MRR", 4 * raw_mrr if raw_mrr is not None else None, 4),
            component(
                "Context precision proxy",
                5 * raw_precision if raw_precision is not None else None,
                5,
            ),
        ]
    )

    if args.mode != "retrieval":
        answer_quality_points: list[float] = []
        if args.judge == "llm":
            for record in local:
                judgment = record.get("judgment")
                if not judgment:
                    answer_quality_points.append(0.0)
                    continue
                case_points = (
                    16 * judgment["required_fact_coverage"]
                    + 10 * judgment["faithfulness"]
                    + 5 * judgment["directness"]
                    + 5 * judgment["completeness"]
                    + 4 * (not judgment["contradiction"])
                )
                if record["safety_critical"] and judgment["critical_failure"]:
                    case_points = 0.0
                answer_quality_points.append(case_points)
        components.append(
            component(
                "Answer quality",
                mean(answer_quality_points),
                40,
            )
        )

        route_accuracy = mean(record.get("route_correct") for record in primary)
        components.append(
            component(
                "Route accuracy",
                7 * route_accuracy if route_accuracy is not None else None,
                7,
            )
        )

        abstain_records = [
            record
            for record in primary
            if record["expected_route"] == "abstain" and record.get("judgment")
        ]
        abstention_accuracy = mean(
            record["judgment"]["abstention_correct"] for record in abstain_records
        )
        components.append(
            component(
                "Abstention behavior",
                5 * abstention_accuracy if abstention_accuracy is not None else None,
                5,
            )
        )

        web_records = [
            record
            for record in primary
            if record["expected_route"] == "web" and record.get("judgment")
        ]
        web_grounding = mean(
            record["judgment"]["web_grounded"] for record in web_records
        )
        components.append(
            component(
                "Web grounding",
                3 * web_grounding if web_grounding is not None else None,
                3,
            )
        )

        grader_records = [
            record
            for record in primary
            if record.get("judgment") and record.get("grader_recheck")
        ]
        hallucination_accuracy = mean(
            record["grader_recheck"]["grounded"]
            == (record["judgment"]["faithfulness"] == 1.0)
            for record in grader_records
        )
        answer_grader_accuracy = mean(
            record["grader_recheck"]["useful"]
            == (
                record["judgment"]["required_fact_coverage"] >= 0.8
                and record["judgment"]["directness"] >= 0.8
                and not record["judgment"]["contradiction"]
            )
            for record in grader_records
        )
        components.extend(
            [
                component(
                    "Hallucination grader accuracy",
                    (
                        3 * hallucination_accuracy
                        if hallucination_accuracy is not None
                        else None
                    ),
                    3,
                ),
                component(
                    "Answer grader accuracy",
                    (
                        2 * answer_grader_accuracy
                        if answer_grader_accuracy is not None
                        else None
                    ),
                    2,
                ),
            ]
        )

        completion = mean(record["success"] for record in primary)
        bounded = mean(
            not record.get("timed_out", False)
            and record.get("generation_attempts", 0) <= args.max_generation_attempts
            and record.get("web_search_attempts", 0) <= args.max_web_search_attempts
            for record in primary
        )
        latencies = [record["latency_seconds"] for record in primary]
        p95_latency = percentile(latencies, 0.95)
        if p95_latency is None:
            latency_factor = None
        elif p95_latency <= args.latency_target_p95:
            latency_factor = 1.0
        elif p95_latency <= 1.5 * args.latency_target_p95:
            latency_factor = 0.5
        else:
            latency_factor = 0.0

        model_calls = [record["pipeline_llm_calls"] for record in primary]
        p95_calls = percentile([float(value) for value in model_calls], 0.95)
        if p95_calls is None:
            call_factor = None
        elif p95_calls <= args.model_call_target:
            call_factor = 1.0
        elif p95_calls <= 1.5 * args.model_call_target:
            call_factor = 0.5
        else:
            call_factor = 0.0

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record["id"]].append(record)
        repeatability_values: list[bool] = []
        if args.repetitions > 1 and args.judge == "llm":
            for case_records in grouped.values():
                routes = {record.get("actual_route") for record in case_records}
                coverages = [
                    record["judgment"]["required_fact_coverage"]
                    for record in case_records
                    if record.get("judgment")
                ]
                repeatability_values.append(
                    len(routes) == 1
                    and len(coverages) == len(case_records)
                    and max(coverages) - min(coverages) <= 0.1
                )
        repeatability = mean(repeatability_values)
        observability = mean(
            "retrieved_documents" in record
            and "generation_attempts" in record
            and "web_search_attempts" in record
            for record in primary
        )
        components.extend(
            [
                component("Completion reliability", 4 * completion, 4),
                component("Bounded execution", 3 * bounded, 3),
                component(
                    "Latency",
                    3 * latency_factor if latency_factor is not None else None,
                    3,
                ),
                component(
                    "Model-call efficiency",
                    2 * call_factor if call_factor is not None else None,
                    2,
                ),
                component(
                    "Repeatability",
                    2 * repeatability if repeatability is not None else None,
                    2,
                ),
                component("Observability", 1 * observability, 1),
            ]
        )
    else:
        p95_latency = percentile(
            [record["latency_seconds"] for record in primary], 0.95
        )
        p95_calls = None
        route_accuracy = None

    points = sum(item["points"] for item in components if item["points"] is not None)
    maximum = sum(item["maximum"] for item in components)
    normalized = 100 * points / maximum if maximum else None
    critical_failures = [
        record["id"]
        for record in primary
        if record.get("timed_out")
        or record.get("judgment", {}).get("critical_failure", False)
    ]
    if normalized is not None and critical_failures:
        normalized = min(normalized, 59.0)

    graded_group_recall = mean(
        (
            record.get("graded_retrieval_metrics", {}).get("evidence_group_recall")
            if record.get("graded_retrieval_metrics")
            else None
        )
        for record in local
    )
    return {
        "case_count": len(primary),
        "run_count": len(records),
        "points": round(points, 3),
        "maximum_evaluated": round(maximum, 3),
        "normalized_score": round(normalized, 2) if normalized is not None else None,
        "critical_failures": critical_failures,
        "components": components,
        "metrics": {
            "raw_evidence_group_recall": raw_group_recall,
            "graded_evidence_group_recall": graded_group_recall,
            "raw_hit": raw_hit,
            "raw_mrr": raw_mrr,
            "raw_context_precision_proxy": raw_precision,
            "route_accuracy": route_accuracy,
            "completion_rate": mean(record["success"] for record in primary),
            "p50_latency_seconds": percentile(
                [record["latency_seconds"] for record in primary], 0.5
            ),
            "p95_latency_seconds": p95_latency,
            "mean_pipeline_llm_calls": mean(
                record["pipeline_llm_calls"] for record in primary
            ),
            "p95_pipeline_llm_calls": p95_calls,
        },
    }


def readiness(score: float | None) -> str:
    if score is None:
        return "Not scored"
    if score >= 90:
        return "Production candidate"
    if score >= 80:
        return "Strong pilot"
    if score >= 70:
        return "Experimental"
    if score >= 60:
        return "Prototype only"
    return "Not ready"


def slice_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    primary = [record for record in records if record["run_index"] == 1]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in primary:
        grouped[record["question_type"]].append(record)
    rows: list[dict[str, Any]] = []
    for name, items in sorted(grouped.items()):
        rows.append(
            {
                "slice": name,
                "count": len(items),
                "success": mean(item["success"] for item in items),
                "retrieval_recall": mean(
                    (
                        item.get("retrieval_metrics", {}).get("evidence_group_recall")
                        if item.get("retrieval_metrics")
                        else None
                    )
                    for item in items
                ),
                "route_accuracy": mean(item.get("route_correct") for item in items),
                "fact_coverage": mean(
                    item.get("judgment", {}).get("required_fact_coverage")
                    for item in items
                ),
            }
        )
    return rows


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def build_report(
    summary: dict[str, Any],
    records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    score = summary["normalized_score"]
    score_label = (
        "Retrieval-only partial score"
        if metadata["arguments"]["mode"] == "retrieval"
        else readiness(score)
    )
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Generated: `{metadata['generated_at']}`",
        f"- Dataset: `{metadata['dataset']}`",
        f"- Split: `{metadata['arguments']['split']}`",
        f"- Mode: `{metadata['arguments']['mode']}`",
        f"- Judge: `{metadata['arguments']['judge']}`",
        f"- Git commit: `{metadata['git'].get('commit') or 'unknown'}`",
        f"- Git dirty: `{metadata['git'].get('dirty')}`",
        "",
        "## Score",
        "",
        f"**{score if score is not None else 'n/a'} / 100 - {score_label}**",
        "",
        f"Raw points: {summary['points']:.2f} / {summary['maximum_evaluated']:.2f} evaluated points.",
        "",
        "| Component | Points | Maximum |",
        "|---|---:|---:|",
    ]
    for item in summary["components"]:
        points = "n/a" if item["points"] is None else f"{item['points']:.2f}"
        lines.append(f"| {item['name']} | {points} | {item['maximum']:.2f} |")

    metrics = summary["metrics"]
    lines.extend(
        [
            "",
            "## Core Metrics",
            "",
            f"- Raw evidence-group recall: {format_percent(metrics['raw_evidence_group_recall'])}",
            f"- Post-grader evidence-group recall: {format_percent(metrics['graded_evidence_group_recall'])}",
            f"- Route accuracy: {format_percent(metrics['route_accuracy'])}",
            f"- Completion rate: {format_percent(metrics['completion_rate'])}",
            f"- Latency p50/p95: {metrics['p50_latency_seconds']:.2f}s / {metrics['p95_latency_seconds']:.2f}s",
            f"- Mean pipeline LLM calls: {metrics['mean_pipeline_llm_calls']:.2f}",
            "",
            "## Results By Slice",
            "",
            "| Slice | Cases | Success | Retrieval recall | Route accuracy | Fact coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in slice_rows(records):
        lines.append(
            f"| {row['slice']} | {row['count']} | {format_percent(row['success'])} | "
            f"{format_percent(row['retrieval_recall'])} | "
            f"{format_percent(row['route_accuracy'])} | "
            f"{format_percent(row['fact_coverage'])} |"
        )

    primary = [record for record in records if record["run_index"] == 1]
    failures = [
        record
        for record in primary
        if record.get("error")
        or record.get("route_correct") is False
        or record.get("judgment", {}).get("required_fact_coverage", 1.0) < 0.8
    ]
    lines.extend(["", "## Failure Review", ""])
    if not failures:
        lines.append("No automatically flagged failures.")
    for record in failures[:20]:
        reasons: list[str] = []
        if record.get("error"):
            reasons.append(record["error"])
        if record.get("route_correct") is False:
            reasons.append(
                f"route {record.get('actual_route')} != {record['expected_route']}"
            )
        coverage = record.get("judgment", {}).get("required_fact_coverage")
        if coverage is not None and coverage < 0.8:
            reasons.append(f"fact coverage {coverage:.2f}")
        lines.append(f"- `{record['id']}`: {'; '.join(reasons)}")

    if summary["critical_failures"]:
        lines.extend(
            [
                "",
                "## Safety Gate",
                "",
                "Critical failures capped the normalized score at 59:",
                "",
                *[f"- `{case_id}`" for case_id in summary["critical_failures"]],
            ]
        )
    return "\n".join(lines) + "\n"


def prepare_output(path: Path, overwrite: bool) -> None:
    known_files = [path / "cases.jsonl", path / "summary.json", path / "report.md"]
    if not overwrite and any(item.exists() for item in known_files):
        raise FileExistsError(
            f"Output already contains evaluation results: {path}. "
            "Choose another path or pass --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(record), ensure_ascii=True) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "evaluation" / "desmi_nsl_benchmark.jsonl",
    )
    parser.add_argument(
        "--source-document",
        type=Path,
        default=REPO_ROOT / "documents" / "t1542uk.pdf",
    )
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument(
        "--mode", choices=["retrieval", "pipeline", "all"], default="all"
    )
    parser.add_argument("--judge", choices=["llm", "none"], default="llm")
    parser.add_argument("--ids", help="Comma-separated case ids")
    parser.add_argument("--question-type", action="append")
    parser.add_argument("--safety-critical-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--recursion-limit", type=int, default=20)
    parser.add_argument("--latency-target-p95", type=float, default=60.0)
    parser.add_argument("--model-call-target", type=float, default=8.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evaluation" / "results" / f"run-{timestamp}",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--skip-model-health-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be greater than zero")
    if args.repetitions <= 0:
        parser.error("--repetitions must be greater than zero")
    if args.mode == "retrieval":
        args.judge = "none"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = args.dataset.resolve()
    source_path = args.source_document.resolve()
    cases = filter_cases(load_dataset(dataset_path), args)

    source_hash = sha256_file(source_path) if source_path.exists() else None
    if args.judge == "llm" and not source_path.exists():
        raise FileNotFoundError(
            f"Source document is required for LLM judging: {source_path}"
        )
    if (
        source_hash
        and source_hash != EXPECTED_SOURCE_SHA256
        and not args.allow_source_mismatch
    ):
        raise ValueError(
            f"Source SHA-256 mismatch: {source_hash}. "
            "Pass --allow-source-mismatch only when this is intentional."
        )

    print(
        f"Selected {len(cases)} cases, {args.repetitions} run(s) each "
        f"({args.mode}, judge={args.judge})."
    )
    if args.dry_run:
        for case in cases:
            print(f"{case['id']}: {case['question']}")
        return 0

    if args.mode != "retrieval" and not args.skip_model_health_check:
        from graph.config import VLLM_API_KEY, VLLM_BASE_URL

        try:
            check_model_server(VLLM_BASE_URL, VLLM_API_KEY)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    prepare_output(args.output, args.overwrite)
    cases_path = args.output / "cases.jsonl"
    if args.overwrite:
        cases_path.write_text("", encoding="utf-8")

    evidence_provider = EvidenceProvider(source_path)
    records: list[dict[str, Any]] = []
    total = len(cases) * args.repetitions
    completed = 0
    for case in cases:
        for run_index in range(1, args.repetitions + 1):
            completed += 1
            print(f"[{completed}/{total}] {case['id']} run {run_index}")
            record = run_case(case, run_index, args, evidence_provider)
            records.append(record)
            append_record(cases_path, record)
            status = "ok" if record["success"] else record.get("error", "failed")
            print(f"  {status} ({record['latency_seconds']:.2f}s)")

    from graph.config import (
        CHUNK_OVERLAP,
        CHUNK_SIZE,
        EMBEDDING_MODEL,
        MAX_GENERATION_ATTEMPTS,
        MAX_WEB_SEARCH_ATTEMPTS,
        RETRIEVAL_K,
        VLLM_BASE_URL,
        VLLM_MODEL,
    )

    args.max_generation_attempts = MAX_GENERATION_ATTEMPTS
    args.max_web_search_attempts = MAX_WEB_SEARCH_ATTEMPTS
    summary = aggregate_results(records, args)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "source_document": str(source_path),
        "source_sha256": source_hash,
        "git": git_snapshot(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "rag_configuration": {
            "vllm_base_url": VLLM_BASE_URL,
            "vllm_model": VLLM_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "retrieval_k": RETRIEVAL_K,
            "max_generation_attempts": MAX_GENERATION_ATTEMPTS,
            "max_web_search_attempts": MAX_WEB_SEARCH_ATTEMPTS,
        },
        "arguments": json_safe(vars(args)),
    }
    summary_document = {"metadata": metadata, "summary": summary}
    (args.output / "summary.json").write_text(
        json.dumps(summary_document, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "report.md").write_text(
        build_report(summary, records, metadata), encoding="utf-8"
    )
    print(f"Results: {args.output}")
    print(
        f"Score: {summary['normalized_score']} / 100 "
        f"({summary['points']}/{summary['maximum_evaluated']} evaluated points)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
