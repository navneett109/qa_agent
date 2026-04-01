from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from functools import lru_cache


def _mask_token(token: str) -> str:
    if len(token) > 8:
        return f"{token[:4]}****{token[-4:]}"
    return "<short-token>"


@lru_cache(maxsize=16)
def _build_embeddings(token: str) -> HuggingFaceEndpointEmbeddings:
    print(f"[HF EMBEDDINGS] ✅ Initializing cached client with token: {_mask_token(token)}")
    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=token,
    )

def get_embeddings(hf_token: str = None):
    """
    Returns a dynamic HuggingFace embeddings instance. 
    Strictly prioritizes the provided hf_token to ensure 'your keys, not mine'.
    """
    # Prefer provided token, then env
    token = hf_token or os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN missing for embeddings")

    return _build_embeddings(token)