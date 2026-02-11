# Patent Research Assistant

AI project for patent discovery, trend analysis, and forecasting using OpenRouter, OpenSearch, and CrewAI.

## Architecture

```text
+------------------------------+
| Frontend Layer               |
| - Streamlit UI (simple_ui.py)|
| - CLI (agentic_rag.py)       |
+--------------+---------------+
               |
               v
+------------------------------+
| Backend API Layer            |
| - FastAPI (backend_api.py)   |
| - /health, /status           |
| - /analysis, /search         |
| - /iterative-search          |
+--------------+---------------+
               |
               v
+------------------------------+
| Intelligence Layer           |
| - OpenRouter chat models     |
| - OpenRouter embeddings      |
| - CrewAI orchestration       |
+--------------+---------------+
               |
               v
+------------------------------+
| Retrieval/Storage Layer      |
| - OpenSearch (keyword/vector)|
| - Index: patents             |
+------------------------------+
```

## Files

- `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/backend_api.py`: backend API
- `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/simple_ui.py`: Streamlit UI
- `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/agentic_rag.py`: CLI app
- `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/patent_crew.py`: multi-agent analysis
- `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/patent_search_tools.py`: search tools
- `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/opensearch_client.py`: OpenSearch client/env config

## Requirements

- Python 3.11+
- OpenSearch instance
- OpenRouter API key

## Install

```bash
cd '/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI'
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Create `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=qwen/qwen3-coder
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=patents
# optional:
# OPENSEARCH_USE_SSL=false
# OPENSEARCH_VERIFY_CERTS=false
# OPENSEARCH_USERNAME=
# OPENSEARCH_PASSWORD=

# optional (for UI -> backend mode)
# BACKEND_API_URL=http://localhost:8000
```

## Run

### 1) Start backend API

```bash
cd '/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI'
source .venv/bin/activate
uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

### 2) Start Streamlit UI (same UI, backend-enabled)

```bash
cd '/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI'
source .venv/bin/activate
export BACKEND_API_URL='http://127.0.0.1:8000'
streamlit run simple_ui.py
```

### 3) CLI mode

```bash
cd '/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI'
source .venv/bin/activate
python agentic_rag.py
```

## Notes

- UI layout is unchanged; only backend wiring is updated.
- If `BACKEND_API_URL` is not set, `simple_ui.py` falls back to direct local calls.
- For production deployment, host backend on a long-running service (not serverless runtime).

## Credits

developed by anurag singh
github github.com/anurag-m1
instagram.com/ca_anuragsingh
