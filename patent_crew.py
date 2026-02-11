import os
import re
from collections import Counter
from datetime import datetime

import requests

from openrouter_client import (
    get_openrouter_base_url,
    get_openrouter_headers,
    list_openrouter_models,
)
from opensearch_client import get_default_opensearch_client, get_opensearch_index_name

DEFAULT_OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-coder")

STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "against",
    "along",
    "also",
    "among",
    "and",
    "are",
    "been",
    "before",
    "between",
    "could",
    "does",
    "each",
    "from",
    "have",
    "into",
    "many",
    "more",
    "most",
    "other",
    "over",
    "same",
    "such",
    "than",
    "that",
    "their",
    "them",
    "there",
    "these",
    "this",
    "those",
    "through",
    "under",
    "using",
    "with",
    "within",
    "would",
}

TOPIC_PATTERNS = {
    "solid_state": ["solid state", "solid-state", "solid electrolyte"],
    "anode": ["anode", "silicon", "graphite", "lithium metal"],
    "cathode": ["cathode", "lfp", "nmc", "nickel", "manganese", "cobalt"],
    "electrolyte": ["electrolyte", "separator", "ionic liquid"],
    "thermal_management": ["thermal", "cooling", "heat dissipation", "temperature"],
    "charging": ["fast charging", "rapid charging", "charging protocol", "charger"],
    "battery_management_system": [
        "battery management",
        "bms",
        "state of charge",
        "state-of-health",
    ],
    "manufacturing": ["manufactur", "coating", "calendering", "formation", "assembly"],
    "safety": ["safety", "short circuit", "overcharge", "thermal runaway"],
    "recycling": ["recycling", "second life", "reuse", "recovery"],
}


def normalize_openrouter_model(model_name):
    if model_name.startswith("openrouter/"):
        return model_name[len("openrouter/") :]
    return model_name


def check_openrouter_availability():
    try:
        return list_openrouter_models(timeout=8)
    except Exception as e:
        print(f"Error connecting to OpenRouter: {e}")
        return []


def test_model(model_name):
    try:
        payload = {
            "model": normalize_openrouter_model(model_name),
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "temperature": 0,
            "max_tokens": 32,
        }
        response = requests.post(
            f"{get_openrouter_base_url()}/chat/completions",
            headers=get_openrouter_headers(),
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return bool(str(content).strip())
    except Exception as e:
        print(f"Error testing model {model_name}: {e}")
        return False


def extract_year(date_text):
    match = re.search(r"(19|20)\d{2}", date_text or "")
    return match.group(0) if match else "Unknown"


def detect_topics(text):
    lowered = text.lower()
    topics = []
    for topic, patterns in TOPIC_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            topics.append(topic)
    return topics or ["other"]


def extract_keywords(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text.lower())
    return [word for word in words if word not in STOPWORDS]


def _search_patents(research_area, top_k=40):
    client = get_default_opensearch_client()
    index_name = get_opensearch_index_name()

    search_query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {"match": {"title": {"query": research_area, "boost": 2}}},
                    {"match": {"abstract": research_area}},
                ],
                "minimum_should_match": 1,
            }
        },
        "_source": ["title", "abstract", "publication_date", "patent_id"],
    }

    response = client.search(index=index_name, body=search_query)
    return response.get("hits", {}).get("hits", [])


def _build_structured_summary(research_area, hits):
    rows = []
    for hit in hits:
        source = hit.get("_source", {})
        rows.append(
            {
                "title": source.get("title", "N/A"),
                "abstract": source.get("abstract", ""),
                "date": source.get("publication_date", "Unknown"),
                "patent_id": source.get("patent_id", "N/A"),
            }
        )

    year_counts = Counter(extract_year(row["date"]) for row in rows)
    topic_counts = Counter()
    keyword_counts = Counter()

    for row in rows:
        combined_text = f"{row['title']} {row['abstract']}"
        topic_counts.update(detect_topics(combined_text))
        keyword_counts.update(extract_keywords(combined_text))

    known_years = {year: count for year, count in year_counts.items() if year != "Unknown"}
    top_year = max(known_years, key=known_years.get) if known_years else "Unknown"

    lines = [
        "Patent Trend Analysis",
        f"- Research area: {research_area}",
        f"- Records analyzed: {len(rows)}",
        f"- Peak publication year: {top_year}",
        "",
        "Publication trend by year:",
    ]

    for year, count in sorted(year_counts.items(), key=lambda x: x[0], reverse=True):
        lines.append(f"- {year}: {count}")

    lines.append("")
    lines.append("Top technology themes:")
    for topic, count in topic_counts.most_common(8):
        lines.append(f"- {topic.replace('_', ' ')}: {count}")

    lines.append("")
    lines.append("Top recurring keywords:")
    for keyword, count in keyword_counts.most_common(12):
        lines.append(f"- {keyword}: {count}")

    lines.append("")
    lines.append("Sample patents:")
    for i, row in enumerate(rows[:10], start=1):
        abstract = (row["abstract"] or "").replace("\n", " ").strip()
        abstract = (abstract[:220] + "...") if len(abstract) > 220 else abstract
        lines.append(
            f"{i}. {row['title']} | {row['date']} | {row['patent_id']} | {abstract}"
        )

    return "\n".join(lines)


def _generate_forecast_with_openrouter(research_area, model_name, structured_summary):
    prompt = f"""
You are a patent innovation analyst.
Use the dataset summary below to produce a focused report.

Output format:
1) Executive summary (5 bullets)
2) Emerging sub-technologies (ranked)
3) Key companies/assignees likely to lead
4) 12-24 month innovation forecast
5) Recommended R&D bets (top 5)
6) Risks and blind spots

Research area: {research_area}

Dataset summary:
{structured_summary}
""".strip()

    payload = {
        "model": normalize_openrouter_model(model_name),
        "messages": [
            {"role": "system", "content": "Be concise, factual, and data-grounded."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 1400,
    }

    response = requests.post(
        f"{get_openrouter_base_url()}/chat/completions",
        headers=get_openrouter_headers(),
        json=payload,
        timeout=90,
    )
    response.raise_for_status()

    return (
        response.json()
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


def run_patent_analysis(research_area, model_name=DEFAULT_OPENROUTER_MODEL):
    if not research_area or not research_area.strip():
        return "Analysis failed: research area is required."

    research_area = research_area.strip()
    model_name = (model_name or DEFAULT_OPENROUTER_MODEL).strip()

    try:
        available_models = check_openrouter_availability()
        normalized_model = normalize_openrouter_model(model_name)
        if available_models and normalized_model not in available_models:
            print(
                f"Warning: model '{normalized_model}' not found in OpenRouter catalog; continuing."
            )

        if not test_model(normalized_model):
            raise RuntimeError(f"Model {normalized_model} is not responding to test prompts.")

        hits = _search_patents(research_area, top_k=40)
        if not hits:
            return (
                f"No patents found for '{research_area}'. "
                "Check OPENSEARCH_INDEX data and try a broader query."
            )

        structured_summary = _build_structured_summary(research_area, hits)

        try:
            forecast = _generate_forecast_with_openrouter(
                research_area, normalized_model, structured_summary
            )
        except Exception as llm_error:
            forecast = (
                "LLM forecast generation failed, returning deterministic trend summary only.\n"
                f"Reason: {llm_error}"
            )

        return (
            f"Model used: {normalized_model}\n"
            f"Records retrieved: {len(hits)}\n\n"
            f"{structured_summary}\n\n"
            f"Innovation Forecast\n{forecast}"
        )
    except Exception as e:
        return (
            f"Analysis failed: {str(e)}\n\nTroubleshooting tips:\n"
            "1. Set OPENROUTER_API_KEY in your environment or .env file\n"
            "2. Use a valid OpenRouter model id (for example: qwen/qwen3-coder)\n"
            "3. Verify OpenRouter API access and account quota\n"
            "4. Ensure OpenSearch is reachable and OPENSEARCH_INDEX has documents"
        )


if __name__ == "__main__":
    research_area = input("Enter the research area to analyze: ").strip()
    if not research_area:
        print("Research area is required.")
        raise SystemExit(1)

    model_name = input(
        f"Enter the OpenRouter model to use (default: {DEFAULT_OPENROUTER_MODEL}): "
    ).strip()
    if not model_name:
        model_name = DEFAULT_OPENROUTER_MODEL

    result = run_patent_analysis(research_area, model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patent_analysis_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(str(result))

    print(f"Analysis completed and saved to {filename}")
