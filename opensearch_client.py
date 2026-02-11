import os

from opensearchpy import OpenSearch


def _to_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def get_opensearch_index_name():
    return os.getenv("OPENSEARCH_INDEX", "patents")


def get_opensearch_host():
    return os.getenv("OPENSEARCH_HOST", "localhost")


def get_opensearch_port():
    try:
        return int(os.getenv("OPENSEARCH_PORT", "9200"))
    except ValueError:
        return 9200


def get_opensearch_client(host=None, port=None, check_connection=True):
    resolved_host = host or get_opensearch_host()
    resolved_port = int(port) if port is not None else get_opensearch_port()

    use_ssl = _to_bool(os.getenv("OPENSEARCH_USE_SSL"), default=False)
    verify_certs = _to_bool(os.getenv("OPENSEARCH_VERIFY_CERTS"), default=False)

    username = os.getenv("OPENSEARCH_USERNAME")
    password = os.getenv("OPENSEARCH_PASSWORD")

    client_config = {
        "hosts": [{"host": resolved_host, "port": resolved_port}],
        "http_compress": True,
        "timeout": 30,
        "max_retries": 3,
        "retry_on_timeout": True,
        "use_ssl": use_ssl,
        "verify_certs": verify_certs,
    }

    if username and password:
        client_config["http_auth"] = (username, password)

    client = OpenSearch(**client_config)

    if check_connection:
        if client.ping():
            print("Connected to OpenSearch!")
            info = client.info()
            print(f"Cluster name: {info['cluster_name']}")
            print(f"OpenSearch version: {info['version']['number']}")
        else:
            print("Connection failed!")
            raise ConnectionError("Failed to connect to OpenSearch.")
    return client


def get_default_opensearch_client(check_connection=True):
    return get_opensearch_client(check_connection=check_connection)


def create_index_if_not_exists(client, index_name):
    """
    Create an OpenSearch index with proper mapping for vector search if it doesn't exist.

    Args:
        client: OpenSearch client instance
        index_name: Name of the index to create
    """
    # Delete the index if it exists (to ensure proper mapping)
    if client.indices.exists(index=index_name):
        print(
            f"Deleting existing index '{index_name}' to recreate with proper mappings..."
        )
        client.indices.delete(index=index_name)

    # Get dimension from a sample embedding
    from embedding import get_embedding

    sample_embedding = get_embedding("Sample text for dimension detection")
    dimension = len(sample_embedding)
    print(f"Using embedding dimension: {dimension}")

    # Define mappings with vector field for embeddings
    mappings = {
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "abstract": {"type": "text"},
                "publication_date": {
                    "type": "date",
                    "format": "yyyy-MM-dd||yyyy||epoch_millis||strict_date_optional_time",
                },
                "patent_id": {"type": "keyword"},
                "pdf": {"type": "keyword"},
                "token_count": {"type": "integer"},
                "embedding": {"type": "knn_vector", "dimension": dimension},
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil",  # Use cosine similarity for embeddings
            }
        },
    }

    try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Created index '{index_name}' with vector search capabilities.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


if __name__ == "__main__":
    client = get_default_opensearch_client()

    # List all indices
    indices = client.cat.indices(format="json")
    print("Available indices:")
    for index in indices:
        print(f"  - {index['index']}")
