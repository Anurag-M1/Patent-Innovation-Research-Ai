import os
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

from embedding import get_embedding
from openrouter_client import get_openrouter_base_url, get_openrouter_headers
from opensearch_client import get_opensearch_client
from patent_crew import run_patent_analysis
from patent_search_tools import hybrid_search, iterative_search, keyword_search, semantic_search

load_dotenv()

st.set_page_config(page_title="Patent Research Assistant", layout="wide")
st.title("Patent Research Assistant")
st.caption("Simple UI for analysis, search, and system checks")


def render_results(results):
    if not results:
        st.info("No results found.")
        return

    st.success(f"Found {len(results)} results")
    for i, hit in enumerate(results, start=1):
        source = hit.get("_source", {})
        with st.expander(f"{i}. {source.get('title', 'Untitled patent')}"):
            st.write(f"**Score:** {hit.get('_score', 'N/A')}")
            st.write(f"**Date:** {source.get('publication_date', 'N/A')}")
            st.write(f"**Patent ID:** {source.get('patent_id', 'N/A')}")
            st.write(source.get("abstract", ""))


tab_analysis, tab_search, tab_iterative, tab_status = st.tabs(
    ["Analysis", "Search", "Iterative", "System Status"]
)

with tab_analysis:
    st.subheader("Run Complete Patent Analysis")
    research_area = st.text_input("Research area", placeholder="e.g. battery recycling")
    model_name = st.text_input(
        "OpenRouter model",
        value=os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder"),
    )

    if st.button("Run analysis", type="primary"):
        if not research_area.strip():
            st.warning("Research area is required.")
        else:
            with st.spinner("Running analysis..."):
                result = run_patent_analysis(research_area.strip(), model_name.strip())

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patent_analysis_{timestamp}.txt"
            with open(filename, "w") as file:
                file.write(str(result))

            st.success(f"Analysis complete. Saved to `{filename}`")
            st.text_area("Analysis output", value=str(result), height=360)

with tab_search:
    st.subheader("Search Patents")
    query = st.text_input("Search query")
    search_type = st.selectbox("Search type", ["Keyword", "Semantic", "Hybrid"], index=2)
    top_k = st.slider("Top results", min_value=5, max_value=50, value=20)

    if st.button("Search"):
        if not query.strip():
            st.warning("Search query is required.")
        else:
            with st.spinner("Searching..."):
                if search_type == "Keyword":
                    results = keyword_search(query.strip(), top_k=top_k)
                elif search_type == "Semantic":
                    results = semantic_search(query.strip(), top_k=top_k)
                else:
                    results = hybrid_search(query.strip(), top_k=top_k)
            render_results(results)

with tab_iterative:
    st.subheader("Iterative Exploration")
    iterative_query = st.text_input("Initial query")
    steps = st.slider("Refinement steps", min_value=1, max_value=8, value=3)
    top_k_iter = st.slider("Results per step", min_value=5, max_value=50, value=20)

    if st.button("Explore"):
        if not iterative_query.strip():
            st.warning("Initial query is required.")
        else:
            with st.spinner("Exploring..."):
                results = iterative_search(
                    iterative_query.strip(),
                    refinement_steps=steps,
                    top_k=top_k_iter,
                )
            render_results(results)

with tab_status:
    st.subheader("System Status")

    if st.button("Check status"):
        try:
            client = get_opensearch_client("localhost", 9200)
            indices = client.cat.indices(format="json")
            st.success("OpenSearch: connected")
            st.write(f"Indices found: {len(indices)}")
        except Exception as exc:
            st.error(f"OpenSearch: failed ({exc})")

        try:
            response = requests.get(
                f"{get_openrouter_base_url()}/models",
                headers=get_openrouter_headers(),
                timeout=10,
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            st.success("OpenRouter: connected")
            st.write(f"Models visible: {len(models)}")
        except Exception as exc:
            st.error(f"OpenRouter: failed ({exc})")

        try:
            vector = get_embedding("status-check")
            st.success(f"Embedding model: OK (dimension {len(vector)})")
        except Exception as exc:
            st.error(f"Embedding model: failed ({exc})")

st.markdown("---")
st.caption("developed by anurag singh | github github.com/anurag-m1 | instagram.com/ca_anuragsingh")
