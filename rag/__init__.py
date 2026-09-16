"""
Apex RAG Module - Enterprise Retrieval-Augmented Generation for Apex Harness.
"""

from .engine import ApexRAG
from .chunker import chunk_text
from .embeddings import OllamaEmbedder, cosine_similarity
from .store import VectorStore
from .reranker import reciprocal_rank_fusion, simple_lexical_rerank

__all__ = [
    "ApexRAG",
    "chunk_text",
    "OllamaEmbedder",
    "cosine_similarity",
    "VectorStore",
    "reciprocal_rank_fusion",
    "simple_lexical_rerank",
]
