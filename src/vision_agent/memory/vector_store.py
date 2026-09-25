"""Semantic Vector Store for SIVAC Phase 5 (ChromaDB Integration).

Maintains two local persistent vector collections:
    1. 'episodic_trajectories': Past task demonstrations embedded by (Goal + App)
       for few-shot retrieval.
    2. 'learned_heuristics': Reusable operational rules embedded by
       (Trigger + App + Rule) for contextual prompt injection.
"""

import math
import hashlib
import logging
from typing import Optional, List, Dict, Any, Union

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings

from ..config import settings
from .schema import SimilarTaskResult, HeuristicQueryResult

logger = logging.getLogger("sivac.memory.vector_store")


# ---------------------------------------------------------------------------
# Fallback / Fast Offline Embedding Function
# ---------------------------------------------------------------------------

class FastDeterministicEmbeddingFunction(EmbeddingFunction):
    """Deterministic, zero-network 384-dimensional embedding function.

    Computes normalized semantic feature vectors using token hash hashing
    and character n-grams. Used when running offline, in test suites, or as
    a fallback when ONNX weight downloads are unavailable.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            tokens = text.lower().replace("|", " ").replace(":", " ").split()
            if not tokens:
                embeddings.append(vec)
                continue

            # Unigrams and bigrams
            ngrams = list(tokens)
            for i in range(len(tokens) - 1):
                ngrams.append(f"{tokens[i]}_{tokens[i+1]}")

            for token in ngrams:
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if ((h >> 8) & 1) else -1.0
                vec[idx] += sign

            # L2 normalise
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            embeddings.append(vec)

        return embeddings


# ---------------------------------------------------------------------------
# ChromaVectorStore
# ---------------------------------------------------------------------------

class ChromaVectorStore:
    """Manages persistent ChromaDB collections for episodic and heuristic memory."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        client: Optional[chromadb.ClientAPI] = None,
        embedding_fn: Optional[EmbeddingFunction] = None,
        use_fast_embedder: bool = False,
    ) -> None:
        """Initialise ChromaVectorStore with persistence path and collections.

        Args:
            persist_dir: Directory path for ChromaDB storage.
            client: Optional pre-configured Chroma client (e.g. EphemeralClient for tests).
            embedding_fn: Optional custom embedding function.
            use_fast_embedder: If True, uses FastDeterministicEmbeddingFunction immediately.
        """
        self.persist_dir = persist_dir or str(settings.chroma_path)

        if client is not None:
            self.client = client
        else:
            self.client = chromadb.PersistentClient(path=self.persist_dir)

        # Select embedding function
        if embedding_fn is not None:
            self.embedding_fn = embedding_fn
        elif use_fast_embedder:
            self.embedding_fn = FastDeterministicEmbeddingFunction()
        else:
            try:
                # Try Chroma's default embedding function
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                self.embedding_fn = DefaultEmbeddingFunction()
            except Exception as e:
                logger.warning(f"Could not load DefaultEmbeddingFunction ({e}), using fast offline embedder")
                self.embedding_fn = FastDeterministicEmbeddingFunction()

        # Initialize collections with cosine distance
        self.traj_collection = self.client.get_or_create_collection(
            name="episodic_trajectories",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self.heuristic_collection = self.client.get_or_create_collection(
            name="learned_heuristics",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"ChromaVectorStore initialized at: {self.persist_dir}")

    # ------------------------------------------------------------------
    # Episodic Trajectory Operations
    # ------------------------------------------------------------------

    def index_task(
        self,
        task_id: str,
        goal: str,
        application: str = "Desktop",
        status: str = "COMPLETED",
        total_steps: int = 0,
        duration_seconds: float = 0.0,
        model_used: str = "default",
        timestamp: Optional[float] = None,
    ) -> None:
        """Embed and index a task trajectory for semantic retrieval."""
        doc_text = f"Goal: {goal} | Application: {application} | Status: {status}"
        metadata = {
            "task_id": task_id,
            "goal": goal,
            "application": application,
            "status": status,
            "total_steps": int(total_steps),
            "duration_seconds": float(duration_seconds),
            "model_used": model_used,
            "timestamp": float(timestamp or 0.0),
        }

        # Upsert into Chroma
        self.traj_collection.upsert(
            ids=[task_id],
            documents=[doc_text],
            metadatas=[metadata],
        )
        logger.debug(f"Indexed task '{task_id}' into episodic_trajectories: {goal}")

    def query_similar_tasks(
        self,
        goal: str,
        application: Optional[str] = None,
        top_k: int = 3,
        only_successful: bool = False,
    ) -> List[SimilarTaskResult]:
        """Query for similar past tasks given a natural language goal."""
        if self.traj_collection.count() == 0:
            return []

        query_text = f"Goal: {goal}"
        if application:
            query_text += f" | Application: {application}"

        where_filter: Optional[Dict[str, Any]] = None
        if only_successful and application:
            where_filter = {
                "$and": [
                    {"status": "COMPLETED"},
                    {"application": application},
                ]
            }
        elif only_successful:
            where_filter = {"status": "COMPLETED"}
        elif application:
            where_filter = {"application": application}

        try:
            results = self.traj_collection.query(
                query_texts=[query_text],
                n_results=min(top_k, self.traj_collection.count()),
                where=where_filter,
            )
        except Exception as e:
            logger.warning(f"Error querying trajectory collection: {e}")
            return []

        similar_tasks: List[SimilarTaskResult] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return similar_tasks

        ids = results["ids"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for tid, meta, dist in zip(ids, metadatas, distances):
            # Cosine distance: similarity = 1 - distance
            similarity = max(0.0, min(1.0, 1.0 - (dist if dist is not None else 0.0)))
            similar_tasks.append(
                SimilarTaskResult(
                    task_id=tid,
                    goal=meta.get("goal", ""),
                    application=meta.get("application", "Desktop"),
                    status=meta.get("status", "COMPLETED"),
                    total_steps=meta.get("total_steps", 0),
                    duration_seconds=meta.get("duration_seconds", 0.0),
                    similarity_score=round(similarity, 4),
                )
            )

        return similar_tasks

    def delete_task(self, task_id: str) -> None:
        """Remove a task from the trajectory collection."""
        try:
            self.traj_collection.delete(ids=[task_id])
        except Exception as e:
            logger.warning(f"Could not delete task {task_id} from Chroma: {e}")

    # ------------------------------------------------------------------
    # Learned Heuristics Operations
    # ------------------------------------------------------------------

    def index_heuristic(
        self,
        skill_id: str,
        application_name: str,
        trigger_condition: str,
        heuristic_rule: str,
        confidence_score: float = 0.8,
        success_count: int = 1,
        failure_count: int = 0,
    ) -> None:
        """Embed and index a learned heuristic rule."""
        doc_text = (
            f"Trigger: {trigger_condition} | "
            f"Application: {application_name} | "
            f"Rule: {heuristic_rule}"
        )
        metadata = {
            "skill_id": skill_id,
            "application_name": application_name,
            "trigger_condition": trigger_condition,
            "heuristic_rule": heuristic_rule,
            "confidence_score": float(confidence_score),
            "success_count": int(success_count),
            "failure_count": int(failure_count),
        }

        self.heuristic_collection.upsert(
            ids=[skill_id],
            documents=[doc_text],
            metadatas=[metadata],
        )
        logger.debug(f"Indexed heuristic '{skill_id}' for application '{application_name}'")

    def query_heuristics(
        self,
        query_text: str,
        application_name: Optional[str] = None,
        top_k: int = 5,
        min_confidence: float = 0.5,
    ) -> List[HeuristicQueryResult]:
        """Query for matching operational heuristics given context."""
        if self.heuristic_collection.count() == 0:
            return []

        where_filter: Optional[Dict[str, Any]] = None
        if application_name:
            where_filter = {"application_name": application_name}

        try:
            results = self.heuristic_collection.query(
                query_texts=[query_text],
                n_results=min(top_k, self.heuristic_collection.count()),
                where=where_filter,
            )
        except Exception as e:
            logger.warning(f"Error querying heuristic collection: {e}")
            return []

        heuristics: List[HeuristicQueryResult] = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return heuristics

        ids = results["ids"][0]
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for hid, meta, dist in zip(ids, metadatas, distances):
            conf = meta.get("confidence_score", 0.8)
            if conf < min_confidence:
                continue

            similarity = max(0.0, min(1.0, 1.0 - (dist if dist is not None else 0.0)))
            heuristics.append(
                HeuristicQueryResult(
                    skill_id=hid,
                    application_name=meta.get("application_name", "Desktop"),
                    trigger_condition=meta.get("trigger_condition", ""),
                    heuristic_rule=meta.get("heuristic_rule", ""),
                    confidence_score=round(conf, 4),
                    similarity_score=round(similarity, 4),
                )
            )

        return heuristics

    def update_heuristic_stats(
        self,
        skill_id: str,
        confidence_score: float,
        success_count: int,
        failure_count: int,
    ) -> None:
        """Update metadata stats for an existing heuristic in Chroma."""
        try:
            existing = self.heuristic_collection.get(ids=[skill_id])
            if existing and existing["metadatas"] and existing["metadatas"][0]:
                meta = existing["metadatas"][0]
                meta["confidence_score"] = float(confidence_score)
                meta["success_count"] = int(success_count)
                meta["failure_count"] = int(failure_count)
                self.heuristic_collection.update(
                    ids=[skill_id],
                    metadatas=[meta],
                )
        except Exception as e:
            logger.warning(f"Could not update heuristic stats for {skill_id}: {e}")

    def delete_heuristic(self, skill_id: str) -> None:
        """Remove a heuristic from ChromaDB."""
        try:
            self.heuristic_collection.delete(ids=[skill_id])
        except Exception as e:
            logger.warning(f"Could not delete heuristic {skill_id} from Chroma: {e}")

    def count_tasks(self) -> int:
        """Return total number of indexed tasks."""
        return self.traj_collection.count()

    def count_heuristics(self) -> int:
        """Return total number of indexed heuristics."""
        return self.heuristic_collection.count()
