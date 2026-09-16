"""
Reranker and Fusion algorithms for Retrieve-and-Rerank.
Implements Reciprocal Rank Fusion (RRF) and Cross-Evaluation reranking.
"""

from typing import List, Dict, Any


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Combines multiple rankings (e.g. Dense Vector + BM25 Lexical) using RRF:
    RRF_score(d) = SUM_m ( 1 / (k + rank_m(d)) )
    """
    rrf_scores: Dict[int, float] = {}
    doc_lookup: Dict[int, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = item["id"]
            doc_lookup[doc_id] = item
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_n]
    
    results = []
    for doc_id in sorted_ids:
        item = dict(doc_lookup[doc_id])
        item["rrf_score"] = rrf_scores[doc_id]
        results.append(item)

    return results


def simple_lexical_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Reranks candidates based on exact query phrase match, token coverage, and positional density.
    """
    query_lower = query.lower()
    query_tokens = set(query_lower.split())

    scored = []
    for item in candidates:
        content_lower = item["content"].lower()
        
        # 1. Exact phrase boost
        exact_match = 1.0 if query_lower in content_lower else 0.0
        
        # 2. Token overlap ratio
        content_tokens = set(content_lower.split())
        overlap = len(query_tokens.intersection(content_tokens)) / max(1, len(query_tokens))
        
        # 3. Combined score
        rerank_score = (item.get("score", 0.0) * 0.5) + (exact_match * 0.3) + (overlap * 0.2)
        
        entry = dict(item)
        entry["rerank_score"] = round(rerank_score, 4)
        scored.append(entry)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_n]
