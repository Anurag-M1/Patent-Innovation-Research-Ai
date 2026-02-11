from embedding import get_embedding
from opensearch_client import get_default_opensearch_client, get_opensearch_index_name


def _client_and_index():
    return get_default_opensearch_client(), get_opensearch_index_name()


def keyword_search(query_text, top_k=20):
    """
    Perform keyword search using OpenSearch.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search results
    """
    client, index_name = _client_and_index()

    try:
        search_query = {
            "size": top_k,
            "query": {"match": {"abstract": query_text}},
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        response = client.search(index=index_name, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Keyword search error: {e}")
        return []


def semantic_search(query_text, top_k=20):
    """
    Perform semantic search using vector embeddings.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search results
    """
    client, index_name = _client_and_index()

    try:
        query_embedding = get_embedding(query_text)

        search_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k,
                    }
                }
            },
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        response = client.search(index=index_name, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []


def hybrid_search(query_text, top_k=20):
    """
    Perform hybrid search using both keyword and semantic search.

    Args:
        query_text (str): The query text to search for
        top_k (int): Number of results to return

    Returns:
        list: Search results
    """
    client, index_name = _client_and_index()

    try:
        query_embedding = get_embedding(query_text)

        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
                        {"match": {"abstract": query_text}},
                    ]
                }
            },
            "_source": ["title", "abstract", "publication_date", "patent_id"],
        }

        response = client.search(index=index_name, body=search_query)
        return response["hits"]["hits"]
    except Exception as e:
        print(f"Hybrid search error: {e}")
        try:
            fallback_query = {
                "size": top_k,
                "query": {"match": {"abstract": query_text}},
                "_source": ["title", "abstract", "publication_date", "patent_id"],
            }
            response = client.search(index=index_name, body=fallback_query)
            return response["hits"]["hits"]
        except Exception as e2:
            print(f"Fallback search error: {e2}")
            return []


def iterative_search(query_text, refinement_steps=3, top_k=20):
    """
    Perform iterative search with query refinement.

    Args:
        query_text (str): The initial query text
        refinement_steps (int): Number of search refinement iterations
        top_k (int): Number of results per iteration

    Returns:
        list: Search results
    """
    client, index_name = _client_and_index()

    all_results = []
    current_query = query_text

    for i in range(refinement_steps):
        try:
            search_query = {
                "size": top_k,
                "query": {"match": {"abstract": current_query}},
                "_source": ["title", "abstract", "publication_date", "patent_id"],
            }

            response = client.search(index=index_name, body=search_query)
            results = response["hits"]["hits"]

            for result in results:
                if result not in all_results:
                    all_results.append(result)

            if not results:
                break

            top_result = results[0]
            current_query = f"{current_query} {top_result['_source']['title']}"

        except Exception as e:
            print(f"Iterative search error at step {i}: {e}")
            break

    return all_results


if __name__ == "__main__":
    query = "lithium battery"

    print("\nHybrid Search Results:")
    hybrid_results = hybrid_search(query)
    for res in hybrid_results:
        print(res, end="\n\n")
