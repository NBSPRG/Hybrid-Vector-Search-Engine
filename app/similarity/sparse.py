"""
Sparse Similarity — TF-IDF keyword search.

Provides keyword-based retrieval using scikit-learn's TF-IDF vectorizer.
Good at exact keyword matching where dense search misses.
CPU-only, no GPU needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = structlog.get_logger(__name__)


class SparseSearch:
    """
    TF-IDF based sparse retrieval.

    Maintains an in-memory corpus index and supports incremental updates.
    For production scale, this would be replaced by Elasticsearch or
    Qdrant's sparse vector support.
    """

    def __init__(self, max_features: int = 10_000):
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams
            sublinear_tf=True,    # log-scaled TF
        )
        self._corpus: List[str] = []
        self._payloads: List[Dict[str, Any]] = []
        self._doc_ids: List[str] = []
        self._tfidf_matrix = None
        self._is_fitted = False

    def add_documents(
        self,
        texts: List[str],
        payloads: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to the TF-IDF index.

        NOTE: This recomputes the full TF-IDF matrix on each call.
        For production, use incremental indexing or an external engine.
        """
        self._corpus.extend(texts)
        self._payloads.extend(payloads or [{}] * len(texts))

        if doc_ids:
            self._doc_ids.extend(doc_ids)
        else:
            start = len(self._doc_ids)
            self._doc_ids.extend([str(start + i) for i in range(len(texts))])

        # Refit the vectorizer on the full corpus
        if len(self._corpus) > 0:
            self._tfidf_matrix = self._vectorizer.fit_transform(self._corpus)
            self._is_fitted = True
            logger.info(
                "tfidf_index_updated",
                corpus_size=len(self._corpus),
                vocab_size=len(self._vectorizer.vocabulary_),
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search the TF-IDF index for the most relevant documents.

        Args:
            query:           Search query text.
            top_k:           Number of results to return.
            score_threshold: Minimum TF-IDF similarity score.

        Returns:
            List of dicts with keys: id, score, payload, rank.
        """
        if not self._is_fitted or self._tfidf_matrix is None:
            logger.warning("tfidf_search_on_empty_index")
            return []

        # Transform query using the fitted vectorizer
        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Rank by score
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            score = float(similarities[idx])
            if score < score_threshold:
                continue
            results.append(
                {
                    "id": self._doc_ids[idx],
                    "score": score,
                    "payload": {
                        "text": self._corpus[idx],
                        **self._payloads[idx],
                    },
                    "rank": rank + 1,
                }
            )

        logger.info(
            "sparse_search_complete",
            query_len=len(query),
            num_results=len(results),
        )
        return results

    @property
    def corpus_size(self) -> int:
        """Return the number of indexed documents."""
        return len(self._corpus)
