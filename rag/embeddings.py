"""
Local vector embedding client for Ollama.
Default model: nomic-embed-text
"""

import json
import urllib.request
import urllib.error
import numpy as np
from typing import List, Union


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text:latest", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def get_embedding(self, text: str) -> List[float]:
        """Compute embedding for a single string."""
        url = f"{self.host}/api/embeddings"
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("embedding", [])
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding with Ollama model '{self.model}': {e}")

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a list of texts."""
        return [self.get_embedding(t) for t in texts]


def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """Compute cosine similarity between two 1D vectors."""
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))
