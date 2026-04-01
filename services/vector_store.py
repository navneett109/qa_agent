from langchain_pinecone import PineconeVectorStore
from services.vector_db import index
from services.embeddings import get_embeddings
from functools import lru_cache


@lru_cache(maxsize=16)
def _cached_vector_store(token: str):
    embeddings = get_embeddings(token)
    return PineconeVectorStore(index=index, embedding=embeddings)

def get_vector_store(hf_token: str = None, embeddings=None):
    """
    Returns a dynamic Pinecone vector store instance using the provided HF token.
    Uses get_embeddings() to dynamically initialize the embeddings engine.
    """
    if embeddings is not None:
        return PineconeVectorStore(index=index, embedding=embeddings)

    token = hf_token or ""
    return _cached_vector_store(token)