import os
import re
from collections import Counter
from datetime import datetime

import requests

# Use CrewAI and import from crewai.tools
from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool  # Use CrewAI's own tool system

from openrouter_client import (
    get_openrouter_base_url,
    get_openrouter_headers,
    list_openrouter_models,
)
from opensearch_client import get_opensearch_client

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


def parse_patent_results(patents_data):
    """Parse patent records from tool-formatted text into normalized dict rows."""
    rows = []
    blocks = [block.strip() for block in patents_data.split("\n\n") if block.strip()]

    for block in blocks:
        title_match = re.search(r"Title:\s*(.+)", block)
        date_match = re.search(r"Date:\s*(.+)", block)
        patent_id_match = re.search(r"Patent ID:\s*(.+)", block)
        abstract_match = re.search(r"Abstract:\s*(.+)", block, re.DOTALL)

        if not any([title_match, date_match, patent_id_match, abstract_match]):
            continue

        rows.append(
            {
                "title": title_match.group(1).strip() if title_match else "N/A",
                "date": date_match.group(1).strip() if date_match else "N/A",
                "patent_id": (
                    patent_id_match.group(1).strip() if patent_id_match else "N/A"
                ),
                "abstract": abstract_match.group(1).strip() if abstract_match else "",
            }
        )

    return rows


def extract_year(date_text):
    """Extract publication year from date text."""
    match = re.search(r"(19|20)\d{2}", date_text or "")
    return match.group(0) if match else "Unknown"


def detect_topics(text):
    """Infer battery sub-topics from patent title + abstract text."""
    lowered = text.lower()
    topics = []
    for topic, patterns in TOPIC_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            topics.append(topic)
    return topics or ["other"]


def extract_keywords(text):
    """Extract normalized keywords from free text."""
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text.lower())
    return [word for word in words if word not in STOPWORDS]


def normalize_openrouter_model(model_name):
    """Normalize any model input into an OpenRouter model id."""
    if model_name.startswith("openrouter/"):
        return model_name[len("openrouter/"):]
    return model_name


def to_crewai_model(model_name):
    """Convert a model id to CrewAI/LiteLLM OpenRouter format."""
    normalized = normalize_openrouter_model(model_name)
    return f"openrouter/{normalized}"


def check_openrouter_availability():
    """Check if OpenRouter is reachable and return available model ids."""
    try:
        return list_openrouter_models(timeout=8)
    except Exception as e:
        print(f"Error connecting to OpenRouter: {e}")
        return []


# Test model with a simple query to verify it works
def test_model(model_name):
    """Test if the model can respond to a simple prompt."""
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


# Define custom tools by extending BaseTool from CrewAI
class SearchPatentsTool(BaseTool):
    name: str = "search_patents"
    description: str = "Search for patents matching a query"

    def _run(self, query: str, top_k: int = 20) -> str:
        client = get_opensearch_client("localhost", 9200)
        index_name = "patents"

        search_query = {
            "size": top_k,
            "query": {"bool": {"must": [{"match": {"abstract": query}}]}},
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        try:
            response = client.search(index=index_name, body=search_query)
            results = response["hits"]["hits"]

            # Format results as a string for better LLM consumption
            formatted_results = []
            for i, hit in enumerate(results):
                source = hit["_source"]
                formatted_results.append(
                    f"{i+1}. Title: {source.get('title', 'N/A')}\n"
                    f"   Date: {source.get('publication_date', 'N/A')}\n"
                    f"   Patent ID: {source.get('patent_id', 'N/A')}\n"
                    f"   Abstract: {source.get('abstract', 'N/A')[:200]}...\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error searching patents: {str(e)}"


class SearchPatentsByDateRangeTool(BaseTool):
    name: str = "search_patents_by_date_range"
    description: str = "Search for patents in a specific date range"

    def _run(self, query: str, start_date: str, end_date: str, top_k: int = 30) -> str:
        client = get_opensearch_client("localhost", 9200)
        index_name = "patents"

        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [{"match": {"abstract": query}}],
                    "filter": [
                        {
                            "range": {
                                "publication_date": {"gte": start_date, "lte": end_date}
                            }
                        }
                    ],
                }
            },
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        try:
            response = client.search(index=index_name, body=search_query)
            results = response["hits"]["hits"]

            # Format results as a string
            formatted_results = []
            for i, hit in enumerate(results):
                source = hit["_source"]
                formatted_results.append(
                    f"{i+1}. Title: {source.get('title', 'N/A')}\n"
                    f"   Date: {source.get('publication_date', 'N/A')}\n"
                    f"   Patent ID: {source.get('patent_id', 'N/A')}\n"
                    f"   Abstract: {source.get('abstract', 'N/A')[:200]}...\n"
                )

            return "\n".join(formatted_results)
        except Exception as e:
            return f"Error searching patents: {str(e)}"


class AnalyzePatentTrendsTool(BaseTool):
    name: str = "analyze_patent_trends"
    description: str = "Analyze trends in patent data"

    def _run(self, patents_data: str) -> str:
        rows = parse_patent_results(patents_data)
        if not rows:
            return "No patent records were provided for trend analysis."

        year_counts = Counter(extract_year(row["date"]) for row in rows)

        topic_counts = Counter()
        keyword_counts = Counter()
        for row in rows:
            combined_text = f"{row['title']} {row['abstract']}"
            topic_counts.update(detect_topics(combined_text))
            keyword_counts.update(extract_keywords(combined_text))

        unique_ids = {
            row["patent_id"]
            for row in rows
            if row["patent_id"] and row["patent_id"].upper() not in {"N/A", "NONE"}
        }

        known_years = {year: count for year, count in year_counts.items() if year != "Unknown"}
        top_year = max(known_years, key=known_years.get) if known_years else "Unknown"
        top_topics = topic_counts.most_common(5)
        top_keywords = keyword_counts.most_common(10)

        summary_lines = [
            "Patent Trend Analysis",
            f"- Records analyzed: {len(rows)}",
            f"- Unique patent IDs: {len(unique_ids)}",
            "",
            "Publication trend by year:",
        ]

        for year, count in sorted(year_counts.items(), key=lambda x: x[0], reverse=True):
            summary_lines.append(f"- {year}: {count}")

        summary_lines.append("")
        summary_lines.append("Top technology themes:")
        for topic, count in top_topics:
            summary_lines.append(f"- {topic.replace('_', ' ')}: {count}")

        summary_lines.append("")
        summary_lines.append("Top recurring keywords:")
        for keyword, count in top_keywords:
            summary_lines.append(f"- {keyword}: {count}")

        summary_lines.append("")
        summary_lines.append("Key insights:")
        summary_lines.append(
            f"- Highest concentration of publications appears in {top_year}."
        )
        if top_topics:
            summary_lines.append(
                f"- Dominant topic cluster: {top_topics[0][0].replace('_', ' ')}."
            )
        if len(known_years) >= 2:
            newest_year = max(known_years)
            oldest_year = min(known_years)
            trend = "upward" if known_years[newest_year] >= known_years[oldest_year] else "downward"
            summary_lines.append(
                f"- Publication volume trend from {oldest_year} to {newest_year}: {trend}."
            )

        return "\n".join(summary_lines)


# Define our agents
def create_patent_analysis_crew(model_name=DEFAULT_OPENROUTER_MODEL):
    """
    Create a CrewAI crew for patent analysis using OpenRouter.

    Args:
        model_name: OpenRouter model id to use

    Returns:
        Crew: A CrewAI crew configured for patent analysis
    """
    # Check OpenRouter availability
    available_models = check_openrouter_availability()
    if not available_models:
        raise RuntimeError(
            "OpenRouter API is not available. Check OPENROUTER_API_KEY and network access."
        )

    normalized_model = normalize_openrouter_model(model_name)
    if normalized_model not in available_models:
        print(
            f"Warning: model '{normalized_model}' was not found in OpenRouter model catalog. "
            "Proceeding with request anyway."
        )

    # Test configured model
    if not test_model(normalized_model):
        raise RuntimeError(f"Model {normalized_model} is not responding to test prompts.")

    print("Model found and tested successfully")

    llm = to_crewai_model(normalized_model)

    # Create tools using CrewAI's BaseTool subclasses
    tools = [
        SearchPatentsTool(),
        SearchPatentsByDateRangeTool(),
        AnalyzePatentTrendsTool(),
    ]

    # Create agents with the correct tools
    research_director = Agent(
        role="Research Director",
        goal="Coordinate research efforts and define the scope of patent analysis",
        backstory="You are an experienced research director who specializes in technological innovation analysis.",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools=tools,
    )

    patent_retriever = Agent(
        role="Patent Retriever",
        goal="Find and retrieve the most relevant patents related to the research area",
        backstory="You are a specialized patent researcher with expertise in information retrieval systems.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    data_analyst = Agent(
        role="Patent Data Analyst",
        goal="Analyze patent data to identify trends, patterns, and emerging technologies",
        backstory="You are a data scientist specializing in patent analysis with years of experience in technology forecasting.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    innovation_forecaster = Agent(
        role="Innovation Forecaster",
        goal="Predict future innovations and technologies based on patent trends",
        backstory="You are an expert in technological forecasting with a track record of accurate predictions in emerging technologies.",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=tools,
    )

    # Create tasks with shorter, simpler descriptions (to reduce LLM load)
    task1 = Task(
        description="""
        Define a research plan for {research_area} patents:
        1. Key technology areas to focus on
        2. Time periods for analysis (focus on last 3 years)
        3. Specific technological aspects to analyze
        """,
        expected_output="""A research plan with focus areas, time periods, and key technological aspects.""",
        agent=research_director,
    )

    task2 = Task(
        description="""
        Using the research plan, retrieve patents related to {research_area} from the last 3 years.
        Use the search_patents and search_patents_by_date_range tools to gather comprehensive data.
        Focus on the most relevant and innovative patents.
        Group patents by sub-technologies within the target domain.
        Provide a summary of the retrieved patents, including:
        - Total number of patents found
        - Key companies/assignees
        - Main technological categories
        """,
        expected_output="""A comprehensive patent retrieval report containing:
        - Summary of total patents found
        - List of key patents grouped by sub-technology
        - Analysis of top companies/assignees
        - Overview of main technological categories
        - List of the most innovative patents with summaries
        """,
        agent=patent_retriever,
        dependencies=[task1],
    )

    task3 = Task(
        description="""
        Analyze the retrieved patent data to identify trends and patterns:
        1. Identify growing vs. declining areas of innovation
        2. Analyze technology evolution over time
        3. Identify key companies and their focus areas
        4. Determine emerging sub-technologies within {research_area}
        5. Analyze patent claims to understand technological improvements
        
        Create a comprehensive analysis with specific trends, supported by data.
        """,
        expected_output="""A trend analysis report containing:
        - Identification of growing vs. declining technology areas
        - Timeline of technology evolution
        - Company focus analysis
        - Emerging sub-technologies list
        - Technical improvement trends
        - Data-backed conclusions on innovation patterns
        """,
        agent=data_analyst,
        dependencies=[task2],
    )

    task4 = Task(
        description="""
        Based on the patent analysis, predict future innovations in {research_area}:
        1. Identify technologies likely to see breakthroughs in the next 2-3 years
        2. Recommend specific areas for R&D investment
        3. Predict which companies are positioned to lead innovation
        4. Identify potential disruptive technologies
        5. Outline specific technical improvements likely to emerge
        
        Create a detailed forecast with specific technology predictions and justification.
        """,
        expected_output="""A future innovation forecast containing:
        - Predicted breakthrough technologies for next 2-3 years
        - Prioritized list of R&D investment areas
        - Companies likely to lead future innovation
        - Potential disruptive technologies and their impact
        - Timeline of expected technical improvements
        - Justification for all predictions based on patent data
        """,
        agent=innovation_forecaster,
        dependencies=[task3],
    )

    # Create the crew with debugging enabled
    crew = Crew(
        agents=[
            research_director,
            patent_retriever,
            data_analyst,
            innovation_forecaster,
        ],
        tasks=[task1, task2, task3, task4],
        verbose=True,
        process=Process.sequential,
        cache=False,  # Disable cache to prevent issues
    )

    return crew


def run_patent_analysis(
    research_area, model_name=DEFAULT_OPENROUTER_MODEL
):
    """
    Run the patent analysis crew for the specified research area.

    Args:
        research_area (str): The research area to analyze
        model_name (str): OpenRouter model id to use

    Returns:
        str: Analysis results
    """
    if not research_area or not research_area.strip():
        return "Analysis failed: research area is required."

    try:
        crew = create_patent_analysis_crew(model_name)
        result = crew.kickoff(inputs={"research_area": research_area})

        # Extract the string output from the CrewOutput object
        if hasattr(result, "output"):
            # Recent CrewAI versions store results in the 'output' attribute
            return result.output
        elif hasattr(result, "result"):
            # Some versions might use 'result'
            return result.result
        else:
            # Last resort - convert to string
            return str(result)
    except Exception as e:
        return (
            f"Analysis failed: {str(e)}\n\nTroubleshooting tips:\n"
            + "1. Set OPENROUTER_API_KEY in your environment or .env file\n"
            + "2. Use a valid OpenRouter model id (for example: qwen/qwen3-coder)\n"
            + "3. Verify OpenRouter API access and account quota\n"
            + "4. Try a simpler model or reduce task complexity"
        )


if __name__ == "__main__":
    # Get the research area from user input
    research_area = input(
        "Enter the research area to analyze: "
    )
    if not research_area:
        print("Research area is required.")
        raise SystemExit(1)

    # Get the model name from user input
    model_name = input(
        f"Enter the OpenRouter model to use (default: {DEFAULT_OPENROUTER_MODEL}): "
    )
    if not model_name:
        model_name = DEFAULT_OPENROUTER_MODEL

    # Run the analysis
    result = run_patent_analysis(research_area, model_name)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"patent_analysis_{timestamp}.txt"

    # Ensure result is a string before writing to file
    if not isinstance(result, str):
        result = str(result)

    with open(filename, "w") as f:
        f.write(result)

    print(f"Analysis completed and saved to {filename}")
