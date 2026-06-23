import os

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.config import MAX_GENERATION_ATTEMPTS, MAX_WEB_SEARCH_ATTEMPTS
from graph.consts import FALLBACK, GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.local_questions import is_local_document_question
from graph.nodes import fallback, generate, grade_documents, retrieve, web_search
from graph.nodes.generate import format_documents
from graph.state import GraphState

load_dotenv()


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state.get("web_search") or not state.get("documents"):
        if is_local_document_question(state["question"]):
            print("---DECISION: LOCAL DOCUMENTS INSUFFICIENT, STOP---")
            return FALLBACK
        print("---DECISION: LOCAL DOCUMENTS INSUFFICIENT, INCLUDE WEB SEARCH---")
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state.get("documents", [])
    generation = state["generation"]
    generation_attempts = state.get("generation_attempts", 0)
    web_search_attempts = state.get("web_search_attempts", 0)

    score = hallucination_grader.invoke(
        {"documents": format_documents(documents), "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        if is_local_document_question(question):
            print("---DECISION: GENERATION DOES NOT ADDRESS LOCAL QUESTION, STOP---")
            return "give up"
        if web_search_attempts < MAX_WEB_SEARCH_ATTEMPTS:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION, SEARCH WEB---")
            return "not useful"
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION, STOP---")
        return "give up"

    if generation_attempts < MAX_GENERATION_ATTEMPTS:
        print("---DECISION: GENERATION IS NOT GROUNDED, RE-TRY---")
        return "not supported"
    print("---DECISION: GENERATION IS NOT GROUNDED AFTER MAX ATTEMPTS, STOP---")
    return "give up"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(FALLBACK, fallback)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
        FALLBACK: FALLBACK,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
        "give up": FALLBACK,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(FALLBACK, END)

app = workflow.compile()

if os.getenv("DRAW_GRAPH", "false").lower() == "true":
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
