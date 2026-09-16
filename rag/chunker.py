"""
Chunking utilities for Apex RAG.
Supports semantic markdown header splitting and sliding window chunking.
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Normalize whitespace and line breaks."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    metadata: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks respecting paragraph boundaries where possible.
    Returns a list of dicts: [{'text': ..., 'chunk_index': ..., 'metadata': ...}]
    """
    text = clean_text(text)
    if not text:
        return []

    metadata = metadata or {}
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    chunk_idx = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
        
        words = para_clean.split()
        if len(words) + current_len <= chunk_size:
            current_chunk.append(para_clean)
            current_len += len(words)
        else:
            if current_chunk:
                chunk_str = "\n\n".join(current_chunk)
                chunks.append({
                    "text": chunk_str,
                    "chunk_index": chunk_idx,
                    "metadata": {**metadata, "chunk_index": chunk_idx}
                })
                chunk_idx += 1
                
                # Overlap: keep trailing words from previous chunk
                overlap_words = chunk_str.split()[-chunk_overlap:] if chunk_overlap > 0 else []
                current_chunk = [" ".join(overlap_words), para_clean] if overlap_words else [para_clean]
                current_len = len(" ".join(current_chunk).split())
            else:
                # If a single paragraph is larger than chunk_size, split by words
                start = 0
                while start < len(words):
                    end = start + chunk_size
                    slice_words = words[start:end]
                    chunks.append({
                        "text": " ".join(slice_words),
                        "chunk_index": chunk_idx,
                        "metadata": {**metadata, "chunk_index": chunk_idx}
                    })
                    chunk_idx += 1
                    start = end - chunk_overlap if chunk_overlap > 0 and end < len(words) else end
                current_chunk = []
                current_len = 0

    if current_chunk:
        chunk_str = "\n\n".join(current_chunk).strip()
        if chunk_str:
            chunks.append({
                "text": chunk_str,
                "chunk_index": chunk_idx,
                "metadata": {**metadata, "chunk_index": chunk_idx}
            })

    return chunks
