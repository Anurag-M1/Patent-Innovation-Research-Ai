# Patent Research Assistant

AI project for patent discovery, trend analysis, and forecasting using OpenRouter + OpenSearch.

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

## Vercel Deploy (Backend)

This repo includes `/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI/main.py` as FastAPI entrypoint for Vercel.

1. Import repo in Vercel.
2. Framework preset: Other.
3. Root directory: project root.
4. Env vars in Vercel:
   - `OPENROUTER_API_KEY`
   - `OPENROUTER_MODEL=qwen/qwen3-coder`
   - `OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small`
   - `OPENSEARCH_HOST`
   - `OPENSEARCH_PORT`
   - `OPENSEARCH_INDEX`
   - optional OpenSearch auth/SSL vars
5. Redeploy.

Health check endpoint:

```text
https://<your-vercel-domain>/health
```

## Credits

developed by anurag singh
github github.com/anurag-m1
instagram.com/ca_anuragsingh
