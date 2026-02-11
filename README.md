# Patent Research Assistant

An agentic AI project for patent discovery, trend analysis, and innovation forecasting using OpenRouter, OpenSearch, and CrewAI.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![OpenSearch](https://img.shields.io/badge/Search-OpenSearch-005EB8)](https://opensearch.org/)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-black)](https://openrouter.ai/)
[![CrewAI](https://img.shields.io/badge/Agents-CrewAI-orange)](https://www.crewai.com/)

## Overview

This project helps you:
- run multi-agent patent analysis for any research area
- search patents with keyword, semantic, and hybrid retrieval
- explore patent space iteratively
- check OpenSearch + OpenRouter + embedding system health

## New Architecture

```text
+------------------------------+
| User Layer                   |
| - CLI (agentic_rag.py)       |
| - Simple UI (simple_ui.py)   |
+--------------+---------------+
               |
               v
+------------------------------+
| Orchestration Layer          |
| - CrewAI agents              |
| - Research / Retrieval /     |
|   Trend / Forecast workflow  |
+--------------+---------------+
               |
               v
+------------------------------+
| Intelligence Layer           |
| - OpenRouter chat models     |
| - OpenRouter embeddings      |
+--------------+---------------+
               |
               v
+------------------------------+
| Retrieval Layer              |
| - Keyword search             |
| - Semantic vector search     |
| - Hybrid and iterative search|
+--------------+---------------+
               |
               v
+------------------------------+
| Storage Layer                |
| - OpenSearch index: patents  |
+------------------------------+
```

## Project Structure

- `agentic_rag.py`: CLI app
- `simple_ui.py`: Streamlit UI
- `patent_crew.py`: CrewAI multi-agent pipeline
- `patent_search_tools.py`: keyword/semantic/hybrid/iterative search
- `embedding.py`: OpenRouter embedding client
- `openrouter_client.py`: OpenRouter auth/config helpers
- `opensearch_client.py`: OpenSearch client + index mapping setup
- `information_collector.py`: optional SerpAPI data fetch
- `ingestion.py`: JSON -> embeddings -> OpenSearch indexing

## Prerequisites

- Python 3.11 or 3.12
- OpenSearch running on `localhost:9200`
- OpenRouter API key

## Installation

```bash
cd '/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI'
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create `.env` in project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=qwen/qwen3-coder
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small

# Optional: only for information_collector.py
SERPAPI_API_KEY=your_serpapi_key
```

## Run

### CLI

```bash
cd '/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI'
source .venv/bin/activate
python agentic_rag.py
```

### Simple UI

```bash
cd '/Users/anurag/Desktop/PATENT INNOVATION & RESEARCH AI'
source .venv/bin/activate
streamlit run simple_ui.py
```

## Operational Notes

- Status check menu option verifies:
  - OpenSearch connectivity
  - OpenRouter model API connectivity
  - embedding API response
- If OpenSearch is not running:

```bash
brew services start opensearch
```

## Credits

developed by anurag singh
github github.com/anurag-m1
instagram.com/ca_anuragsingh
