"""
Hybrid Similarity — Reciprocal Rank Fusion (RRF).

Combines sparse (TF-IDF) and dense (embedding) retrieval results
using RRF to get the best of both worlds:
  - Dense catches semantic meaning / synonyms
  - Sparse catches exact keywords / rare terms
  - RRF merges rankings without needing score calibration

This is the same approach used by ElasticSearch 8.x and Qdrant hybrid search.

Formula: RRF_score(doc) = Σ 1 / (k + rank_i)   where k = 60 (default)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import structlog

from app.models.base_encoder import BaseEncoder
from app.similarity.dense import DenseSearch
from app.similarity.sparse import SparseSearch

logger = structlog.get_logger(__name__)


class HybridSearch:
    """
    Combines dense and sparse search results via Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dense: DenseSearch,
        sparse: SparseSearch,
        rrf_k: int = 60,
    ):
        self._dense = dense
        self._sparse = sparse
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        encoder: BaseEncoder,
        top_k: int = 5,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Run both dense and sparse search, then fuse with RRF.

        Args:
            query:          Search text.
            encoder:        Encoder for dense search.
            top_k:          Number of final results.
            dense_weight:   Weight multiplier for dense RRF scores.
            sparse_weight:  Weight multiplier for sparse RRF scores.

        Returns:
            Fused results sorted by combined RRF score.
        """
        # Fetch more candidates than top_k from each source
        candidate_k = top_k * 3

        # ── Run both searches ───────────────────────────────
        dense_results = self._dense.search(
            query=query,
            encoder=encoder,
            top_k=candidate_k,
        )

        sparse_results = self._sparse.search(
            query=query,
            top_k=candidate_k,
        )

        logger.info(
            "hybrid_search_candidates",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        # ── Fuse via RRF ────────────────────────────────────
        fused = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        # Sort by fused score descending, take top_k
        fused_sorted = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        results = fused_sorted[:top_k]

        # Add final ranks
        for rank, item in enumerate(results, 1):
            item["rank"] = rank

        logger.info(
            "hybrid_search_complete",
            num_results=len(results),
            top_score=results[0]["rrf_score"] if results else 0,
        )
        return results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merge two ranked lists using RRF.

        RRF_score(doc) = Σ weight_i / (k + rank_i)
        """
        fused: Dict[str, Dict[str, Any]] = {}
        k = self._rrf_k

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = str(result["id"])
            rrf_contribution = dense_weight / (k + rank)

            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "rrf_score": 0.0,
                    "dense_score": result.get("score", 0),
                    "sparse_score": 0.0,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "payload": result.get("payload", {}),
                }

            fused[doc_id]["rrf_score"] += rrf_contribution

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = str(result["id"])
            rrf_contribution = sparse_weight / (k + rank)

            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "rrf_score": 0.0,
                    "dense_score": 0.0,
                    "sparse_score": result.get("score", 0),
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "payload": result.get("payload", {}),
                }

            fused[doc_id]["rrf_score"] += rrf_contribution
            fused[doc_id]["sparse_score"] = result.get("score", 0)
            fused[doc_id]["sparse_rank"] = rank

        return fused
