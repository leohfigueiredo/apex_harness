"""
Unified Apex RAG Engine.
Coordinates Document Chunking, Ollama Embeddings, Vector/BM25 storage, and Reranking.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from .chunker import chunk_text
from .embeddings import OllamaEmbedder
from .store import VectorStore
from .reranker import reciprocal_rank_fusion, simple_lexical_rerank


class ApexRAG:
    def __init__(
        self,
        db_path: Optional[str] = None,
        embed_model: str = "nomic-embed-text:latest",
        ollama_host: str = "http://localhost:11434"
    ):
        if not db_path:
            # Default to storing in user config or apex harness root
            default_dir = Path.home() / ".apex_rag"
            default_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(default_dir / "apex_knowledge.db")
        
        self.db_path = db_path
        self.store = VectorStore(db_path=self.db_path)
        self.embedder = OllamaEmbedder(model=embed_model, host=ollama_host)

    def ingest_text(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None, chunk_size: int = 500) -> int:
        """Chunk, embed, and store a raw text snippet."""
        chunks = chunk_text(text, chunk_size=chunk_size, metadata=metadata or {"source": doc_id})
        if not chunks:
            return 0
        
        texts_to_embed = [c["text"] for c in chunks]
        embeddings = self.embedder.get_embeddings_batch(texts_to_embed)
        self.store.add_chunks(doc_id=doc_id, chunks=chunks, embeddings=embeddings)
        return len(chunks)

    def ingest_file(self, file_path: str, chunk_size: int = 500) -> int:
        """Read a file (.md, .txt, .py, etc.), chunk, embed, and index it."""
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            raise RuntimeError(f"Could not read {path}: {e}")

        meta = {
            "source": str(path),
            "filename": path.name,
            "extension": path.suffix
        }
        return self.ingest_text(text=content, doc_id=path.name, metadata=meta, chunk_size=chunk_size)

    def ingest_directory(self, dir_path: str, extensions: Optional[List[str]] = None, recursive: bool = True) -> Dict[str, int]:
        """Ingest all matching files in a directory."""
        path = Path(dir_path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        allowed_exts = set(extensions) if extensions else {".md", ".txt", ".py", ".sh", ".json", ".html"}
        results = {}

        pattern = "**/*" if recursive else "*"
        for p in path.glob(pattern):
            if p.is_file() and p.suffix.lower() in allowed_exts:
                try:
                    count = self.ingest_file(str(p))
                    results[str(p)] = count
                except Exception as e:
                    results[str(p)] = f"Error: {e}"

        return results

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Execute search across the indexed knowledge.
        Modes:
          - 'dense': Pure vector similarity search
          - 'lexical': Pure Okapi BM25 keyword search
          - 'hybrid': Reciprocal Rank Fusion of Dense + BM25
          - 'rerank': Hybrid search with second-stage phrase & density reranking
        """
        if mode == "dense":
            q_emb = self.embedder.get_embedding(query)
            return self.store.dense_search(q_emb, top_k=top_k)
        
        elif mode == "lexical":
            return self.store.bm25_search(query, top_k=top_k)

        elif mode == "hybrid":
            q_emb = self.embedder.get_embedding(query)
            dense_results = self.store.dense_search(q_emb, top_k=top_k * 2)
            lexical_results = self.store.bm25_search(query, top_k=top_k * 2)
            return reciprocal_rank_fusion([dense_results, lexical_results], top_n=top_k)

        elif mode == "rerank":
            # Retrieve via hybrid first, then apply reranker
            candidates = self.search(query, mode="hybrid", top_k=top_k * 3)
            return simple_lexical_rerank(query, candidates, top_n=top_k)

        else:
            raise ValueError(f"Unknown search mode: {mode}. Choose from 'dense', 'lexical', 'hybrid', 'rerank'.")
