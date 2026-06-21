# Advanced RAG (Retrieval-Augmented Generation)

An intelligent retrieval-augmented generation system built with LangChain and LangGraph that combines document retrieval, relevance grading, hallucination detection, and web search to provide accurate, grounded answers.

## vLLM Serving Approach

This project uses **vLLM as a model server**. The RAG API does not load Qwen directly; it calls vLLM over HTTP at `http://localhost:8001/v1`, while the FastAPI RAG service runs separately on `http://localhost:8000`.

Recommended runtime layout:

```text
Terminal 1: vLLM server
  Qwen2.5-7B-Instruct Q4_K_M GGUF ->  API on port 8001

Terminal 2: RAG API
  LangGraph + Chroma + local docs + Tavily fallback -> FastAPI on port 8000
```

Why this approach:

- Keeps model serving isolated from the RAG application.
- Lets LangChain use vLLM through its OpenAI-compatible client.
- Makes it easy to swap models without rewriting graph logic.
- Keeps local document RAG and web-search fallback in the application layer.

Important vLLM notes:

- Run vLLM in WSL2/Linux/Docker, not native Windows.
- Use a **single-file GGUF** for Qwen2.5-7B Q4_K_M.
- Keep `--served-model-name qwen2.5-7b-instruct-q4_k_m`, because the app uses that model name.
- Keep vLLM running before starting the RAG API.

## Features

- **Local-First RAG**: Retrieves from local `.tex`, `.pdf`, and `.txt` files before falling back to web search
- **Document Retrieval**: Retrieves relevant documents from a Chroma vector store
- **Relevance Grading**: Automatically grades retrieved documents for relevance to the question
- **Hallucination Detection**: Verifies that generated answers are grounded in retrieved documents
- **Answer Validation**: Checks that generated answers actually address the original question
- **Web Search Fallback**: Automatically performs web search when local documents are insufficient
- **Iterative Refinement**: Regenerates answers if they don't meet quality standards
- **State Management**: Maintains conversation state throughout the reasoning process

## Project Structure

```
.
├── main.py                    # Entry point for the RAG system
├── ingestion.py              # Document loading and vector store setup
├── pyproject.toml            # Project configuration and dependencies
├── graph/
│   ├── graph.py             # LangGraph workflow definition
│   ├── state.py             # State schema for the graph
│   ├── consts.py            # Constants for graph nodes
│   ├── nodes/               # Graph node implementations
│   │   ├── retrieve.py      # Document retrieval node
│   │   ├── grade_documents.py # Document grading node
│   │   ├── generate.py      # Answer generation node
│   │   └── web_search.py    # Web search node
│   └── chains/              # LLM chains for specific tasks
│       ├── router.py        # Query routing chain
│       ├── retrieval_grader.py  # Document relevance grader
│       ├── hallucination_grader.py  # Hallucination detection
│       ├── answer_grader.py # Answer validation
│       └── generation.py    # Answer generation chain
```

## Workflow

The system follows this decision flow:

1. **Retrieve**: Retrieves relevant local document chunks from Chroma
2. **Grade Documents**: Evaluates whether retrieved chunks answer the question
3. **Decide**: Generates from local context, or switches to web search if local context is weak
4. **Generate**: Uses a vLLM-served Qwen model to generate an answer
5. **Check Hallucinations**: Verifies the answer is grounded in retrieved documents
6. **Validate Answer**: Confirms the answer addresses the original question
7. **Iterate**: Regenerates if needed, or searches web as fallback

## Setup

### Prerequisites

- WSL2 Ubuntu, Linux, or Docker for vLLM. vLLM does not run natively on Windows.
- NVIDIA GPU visible inside WSL. Verify with `nvidia-smi`.
- Python 3.11+ for this RAG API project.
- `uv` for project and vLLM environment management.
- A Tavily API key for web-search fallback.

### 1. Install Project Dependencies

Run these commands in WSL from the project directory:

```bash
cd /mnt/c/Users/Chivukula/Projects/advanced-rag
uv sync
```

You do not need to manually activate `.venv` when using `uv run`.

### 2. Configure Environment

Create `.env` in the repo root:

```env
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_API_KEY=local-vllm
VLLM_MODEL=qwen2.5-7b-instruct-q4_k_m
TAVILY_API_KEY=your_tavily_key
LOCAL_DOCS_DIR=documents
CHUNK_SIZE=900
CHUNK_OVERLAP=150
RETRIEVAL_K=3
API_SECRET_KEY=your_api_key_for_the_fastapi_endpoint
```

### 3. Install vLLM In WSL

This is the core model-serving layer. It runs independently from the RAG API and exposes an OpenAI-compatible endpoint.

Use a separate environment for vLLM:

```bash
uv venv ~/venvs/vllm --python 3.12 --seed
source ~/venvs/vllm/bin/activate

mkdir -p ~/tmp ~/.cache/uv
export TMPDIR=$HOME/tmp
export UV_CACHE_DIR=$HOME/.cache/uv
export UV_TORCH_BACKEND=cu129

uv pip install "vllm==0.19.1" --torch-backend=cu129
```

Install a C compiler for Triton. If `sudo` asks for a password you do not know, start WSL as root from Windows PowerShell with `wsl -d Ubuntu -u root`, then run the same `apt` commands.

```bash
sudo apt update
sudo apt install -y build-essential
```

Verify vLLM and CUDA:

```bash
python - <<'PY'
import torch
import importlib.metadata as md

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("vllm:", md.version("vllm"))
PY
```

### 4. Download A Single-File GGUF

vLLM's GGUF support expects a single `.gguf` file. The official Qwen GGUF repo can be split into multiple files, so use a single-file Q4_K_M download.

```bash
source ~/venvs/vllm/bin/activate
uv pip install -U "huggingface_hub[cli]"

mkdir -p ~/models/qwen2.5-7b-instruct-q4km

hf download matrixportal/Qwen2.5-7B-Instruct-GGUF \
  --include "qwen2.5-7b-instruct-q4_k_m.gguf" \
  --local-dir ~/models/qwen2.5-7b-instruct-q4km

ls -lh ~/models/qwen2.5-7b-instruct-q4km
```

The `.gguf` file should be several GB, not empty.

### 5. Start vLLM

Keep this terminal running. This is the model server that the RAG app calls through `VLLM_BASE_URL=http://localhost:8001/v1`:

```bash
source ~/venvs/vllm/bin/activate
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

vllm serve ~/models/qwen2.5-7b-instruct-q4km/qwen2.5-7b-instruct-q4_k_m.gguf \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --hf-config-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --served-model-name qwen2.5-7b-instruct-q4_k_m \
  --gpu-memory-utilization 0.75 \
  --max-model-len 4096 \
  --enforce-eager
```

Test from another terminal:

```bash
curl http://localhost:8001/v1/models

curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-7b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

## Usage

### Prepare Vector Store

Put local `.tex`, `.pdf`, and `.txt` files in `documents/`, then ingest them into the vector store:

```bash
uv run python ingestion.py --docs-dir documents --rebuild
```

To also seed the original web URLs, add `--include-web`:

```bash
uv run python ingestion.py --docs-dir documents --include-web --rebuild
```

This loads local files, splits them into chunks, and creates embeddings using HuggingFace models.

### Run the RAG System

Keep vLLM running on port `8001`, then start the FastAPI RAG API in a second WSL terminal:

```bash
uv run python main.py
```

For debugging, run Uvicorn without the reload process:

```bash
uv run uvicorn api.app:api --host 127.0.0.1 --port 8000 --log-level debug
```

Test the API:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your_api_key_for_the_fastapi_endpoint" \
  -d '{"question":"What do my local documents say about this topic?"}'
```

### Run the Streamlit Demo

Keep vLLM running on port `8001`, configure the same `.env` values used by the API, and start the demo:

```bash
uv run streamlit run streamlit_app.py
```

Open the URL printed by Streamlit, upload one or more `.pdf`, `.txt`, or `.tex` files, click **Ingest documents**, and use the chat box. Uploaded files are saved under `documents/streamlit_uploads/` and added to the same persistent Chroma collection used by the FastAPI service.

The Streamlit interface is intended as a local demo. Its uploaded documents and vector index are shared across browser sessions, and the visible chat history is not passed back to the graph as conversational memory.

### Custom Queries

Modify `main.py` to ask different questions:

```python
from graph.graph import app

result = app.invoke(input={"question": "Your question here?"})
print(result)
```

## Dependencies

Key packages:
- **LangChain**: LLM framework
- **LangGraph**: Graph-based workflow orchestration
- **Chroma**: Vector store for documents
- **Tavily**: Web search API
- **HuggingFace Transformers**: Embeddings and language models
- **BeautifulSoup4**: Web scraping
- **Tiktoken**: Token counting

See `pyproject.toml` for the complete list.

## Configuration

- **Vector Store**: Uses Chroma client (configured through environment variables)
- **Embeddings**: HuggingFace "all-MiniLM-L6-v2" model
- **LLM**: vLLM OpenAI-compatible endpoint serving Qwen2.5-7B-Instruct GGUF Q4_K_M
- **Chunking**: Configured with `CHUNK_SIZE` and `CHUNK_OVERLAP` environment variables. The defaults are 900 and 150 tokens to preserve page and table context in technical manuals.
- **Retrieval Count**: Configured with `RETRIEVAL_K`; the default is 3 to keep the assembled context within the model window.

After changing chunk size or overlap, rebuild the vector store because existing chunks are not rewritten automatically:

```bash
python ingestion.py --docs-dir documents --rebuild
```

Changing only `RETRIEVAL_K` does not require re-ingestion, but the running Streamlit or API process must be restarted to reload configuration.

## Testing

Run tests with:

```bash
pytest graph/chains/tests/
```

### Run the Evaluation Benchmark

Rebuild Chroma from the benchmark document, keep vLLM running, and run the development split:

```bash
source ~/.venvs/advanced-rag/bin/activate
python ingestion.py --docs-dir documents --rebuild
python evaluation/run_evals.py --split dev --mode all --judge llm
```

For a fast retrieval-only check that does not call the generation or grading graph:

```bash
python evaluation/run_evals.py --split dev --mode retrieval
```

Use `--ids desmi-007,desmi-009` to run selected cases, or `--dry-run` to validate and list the selected dataset without invoking the pipeline. Each completed run writes `cases.jsonl`, `summary.json`, and `report.md` under `evaluation/results/`.

Pipeline mode checks the configured vLLM `/models` endpoint before starting. Retrieval-only mode does not require vLLM.

## Development

- **Code Formatting**: Uses Black
- **Import Sorting**: Uses isort
- **Type Hints**: Leverages Python type hints for better IDE support

Reformat code:
```bash
black .
isort .
```

## Troubleshooting

### vLLM: `libcudart.so.13` Missing

The installed vLLM wheel expects CUDA 13. Recreate the vLLM environment with the CUDA 12.9 backend:

```bash
deactivate 2>/dev/null || true
rm -rf ~/venvs/vllm

uv venv ~/venvs/vllm --python 3.12 --seed
source ~/venvs/vllm/bin/activate

export UV_TORCH_BACKEND=cu129
uv pip install "vllm==0.19.1" --torch-backend=cu129
```

### vLLM: No Space Left During Install

If extraction fails under `/tmp`, use a disk-backed temp directory:

```bash
mkdir -p ~/tmp ~/.cache/uv
export TMPDIR=$HOME/tmp
export UV_CACHE_DIR=$HOME/.cache/uv
```

### vLLM: Remote GGUF Config Or Split File Errors

Use a local single-file `.gguf` and include `--hf-config-path Qwen/Qwen2.5-7B-Instruct`. Do not serve the official split GGUF repo directly.

### vLLM: Failed To Find C Compiler

Install build tools:

```bash
sudo apt update
sudo apt install -y build-essential
```

Then export:

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
```

### API: `curl` Cannot Connect To Port 8000

If `uv run python main.py` only prints `Started reloader process`, run without reload to see startup errors:

```bash
uv run uvicorn api.app:api --host 127.0.0.1 --port 8000 --log-level debug
```

Check that the server is listening:

```bash
ss -ltnp | grep 8000
```

### Import Errors
If you encounter `ImportError: cannot import name 'grade_documents'`:
- Check that all dependencies in `chains/` are properly installed
- Ensure there are no circular imports in the codebase
- Verify the `.env` file has required API keys

### Vector Store Issues
- Ensure the `.chroma` directory exists or run `ingestion.py` to create it
- Check that HuggingFace embeddings are downloaded
