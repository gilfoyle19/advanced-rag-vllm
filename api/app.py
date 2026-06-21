# api/app.py
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

from graph.graph import app as rag_graph
from api.schemas import ChatRequest, ChatResponse, DocumentItem
from api.auth import verify_api_key

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup and shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("✅ RAG graph loaded and ready.")
    yield
    # Shutdown
    print("🛑 API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
api = FastAPI(
    title="Advanced RAG API",
    description="REST API wrapper around the LangGraph Advanced RAG pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

api.state.limiter = limiter
api.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@api.get("/health")
async def health():
    """
    Returns 200 if the API is running.
    """
    return {"status": "ok"}


@api.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("20/minute")
async def chat(request: Request, body: ChatRequest):
    """
    Send a question to the Advanced RAG pipeline.
    Requires X-Api-Key header.
    Rate limited to 20 requests per minute per IP.
    """
    try:
        result = rag_graph.invoke(input={"question": body.question})
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Graph invocation failed: {str(e)}"
        )

    # Extract generation
    answer = result.get("generation", "")
    if not answer:
        raise HTTPException(status_code=500, detail="Graph returned no answer.")

    # Convert LangChain Documents → DocumentItem
    raw_docs = result.get("documents", [])
    documents = [
        DocumentItem(
            page_content=doc.page_content,
            source=doc.metadata.get("source") if doc.metadata else None,
        )
        for doc in raw_docs
    ]

    web_search_used = bool(result.get("web_search", False))

    return ChatResponse(
        answer=answer,
        documents=documents,
        web_search_used=web_search_used,
    )
