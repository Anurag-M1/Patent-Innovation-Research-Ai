# Patent Research Assistant

AI project for patent discovery, trend analysis, and forecasting using OpenRouter + OpenSearch + CrewAI

<img width="1285" height="665" alt="PIR" src="https://github.com/user-attachments/assets/6d9f76e5-a41b-4664-886d-b7b95e865ba7" />

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
| - Lightweight analysis flow  |
+--------------+---------------+
               |
               v
+------------------------------+
| Retrieval/Storage Layer      |
| - OpenSearch (keyword/vector)|
| - Index: patents             |
+------------------------------+
```

## Dependency Files

- `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/requirements.txt`: backend-only (Vercel-safe)
- `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/requirements-local.txt`: local UI + ingestion extras

## Environment

Create `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=qwen/qwen3-coder
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_INDEX=patents
# Optional:
# OPENSEARCH_USE_SSL=false
# OPENSEARCH_VERIFY_CERTS=false
# OPENSEARCH_USERNAME=
# OPENSEARCH_PASSWORD=

# Optional for UI->API mode
# BACKEND_API_URL=http://localhost:8000
```

## Local Run

```bash
cd '/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI'
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
```

Start backend:

```bash
uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

Start UI (same UI):

```bash
export BACKEND_API_URL='http://127.0.0.1:8000'
streamlit run simple_ui.py
```


## Credits

developed by anurag singh
github github.com/anurag-m1
instagram.com/ca_anuragsingh
