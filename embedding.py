import os

import requests
from dotenv import load_dotenv

from openrouter_client import get_openrouter_base_url, get_openrouter_headers

load_dotenv()


def get_embedding(prompt, model=None):
    """
    Get the embedding for the given prompt using the specified model.

    Args:
        prompt (str): The prompt to embed.
        model (str): The model to use for embedding.

    Returns:
        list: The embedding vector.
    """
    embedding_model = model or os.getenv(
        "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    url = f"{get_openrouter_base_url()}/embeddings"
    data = {"input": prompt, "model": embedding_model}

    response = requests.post(url, headers=get_openrouter_headers(), json=data, timeout=30)
    response.raise_for_status()
    response_data = response.json()
    vectors = response_data.get("data", [])

    if not vectors:
        raise RuntimeError(f"No embedding returned by model '{embedding_model}'.")

    return vectors[0].get("embedding", [])


if __name__ == "__main__":
    sample_prompt = "The sky is blue because of Rayleigh scattering."
    try:
        embedding = get_embedding(sample_prompt)
        print("Embedding Dimesion:", len(embedding))
        print("Embedding:", embedding)
    except Exception as e:
        print(f"Failed to get embedding: {e}")
