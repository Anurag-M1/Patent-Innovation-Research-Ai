import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from embedding import get_embedding
from openrouter_client import get_openrouter_base_url, get_openrouter_headers
from opensearch_client import get_default_opensearch_client
from patent_crew import run_patent_analysis
from patent_search_tools import hybrid_search, iterative_search, keyword_search, semantic_search

load_dotenv()

app = FastAPI(title="Patent Research Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    research_area: str = Field(min_length=1)
    model_name: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    search_type: str = "hybrid"
    top_k: int = Field(default=20, ge=1, le=100)


class IterativeSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    refinement_steps: int = Field(default=3, ge=1, le=12)
    top_k: int = Field(default=20, ge=1, le=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    result = {
        "opensearch": {"ok": False, "message": "Not checked"},
        "openrouter": {"ok": False, "message": "Not checked"},
        "embedding": {"ok": False, "message": "Not checked"},
    }

    try:
        client = get_default_opensearch_client()
        indices = client.cat.indices(format="json")
        result["opensearch"] = {
            "ok": True,
            "message": "Connected",
            "indices": len(indices),
        }
    except Exception as exc:
        result["opensearch"] = {"ok": False, "message": str(exc)}

    try:
        response = requests.get(
            f"{get_openrouter_base_url()}/models",
            headers=get_openrouter_headers(),
            timeout=10,
        )
        response.raise_for_status()
        models = response.json().get("data", [])
        result["openrouter"] = {
            "ok": True,
            "message": "Connected",
            "models": len(models),
        }
    except Exception as exc:
        result["openrouter"] = {"ok": False, "message": str(exc)}

    try:
        vector = get_embedding("status-check")
        result["embedding"] = {
            "ok": True,
            "message": "Connected",
            "dimension": len(vector),
        }
    except Exception as exc:
        result["embedding"] = {"ok": False, "message": str(exc)}

    return result


@app.post("/analysis")
def analysis(payload: AnalysisRequest):
    model_name = (payload.model_name or os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder")).strip()
    result = run_patent_analysis(payload.research_area.strip(), model_name)
    return {"result": result}


@app.post("/search")
def search(payload: SearchRequest):
    search_type = payload.search_type.strip().lower()

    if search_type == "keyword":
        results = keyword_search(payload.query.strip(), top_k=payload.top_k)
    elif search_type == "semantic":
        results = semantic_search(payload.query.strip(), top_k=payload.top_k)
    elif search_type == "hybrid":
        results = hybrid_search(payload.query.strip(), top_k=payload.top_k)
    else:
        raise HTTPException(status_code=400, detail="search_type must be one of: keyword, semantic, hybrid")

    return {"results": results}


@app.post("/iterative-search")
def iterative(payload: IterativeSearchRequest):
    results = iterative_search(
        payload.query.strip(),
        refinement_steps=payload.refinement_steps,
        top_k=payload.top_k,
    )
    return {"results": results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend_api:app",
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", "8000")),
        reload=False,
    )
