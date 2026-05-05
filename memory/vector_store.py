"""
Vector Store — FAISS-based semantic memory.
Stores story text chunks as embeddings for semantic retrieval.
Uses sentence-transformers for lightweight CPU-based embedding.
"""
import os
import json
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a stored text chunk."""
    chapter: int
    scene: int
    characters: list = field(default_factory=list)
    location: str = ""
    memory_type: str = "plot"
    chunk_index: int = 0

    def to_dict(self) -> dict:
        return {
            "chapter": self.chapter,
            "scene": self.scene,
            "characters": self.characters,
            "location": self.location,
            "memory_type": self.memory_type,
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkMetadata":
        return cls(**d)


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    text: str
    score: float
    metadata: ChunkMetadata


class VectorStore:
    """
    FAISS-based vector memory for semantic search over story content.
    Uses all-MiniLM-L6-v2 for embeddings (~90MB, CPU-only).
    """

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self.index_dir = os.path.join(project_dir, "vector_index")
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.meta_path = os.path.join(self.index_dir, "metadata.json")
        self.texts_path = os.path.join(self.index_dir, "texts.json")

        os.makedirs(self.index_dir, exist_ok=True)

        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.Index] = None
        self._texts: List[str] = []
        self._metadata: List[dict] = []
        self._dimension: int = 384  # MiniLM-L6-v2 dimension
        self._embedding_cache: dict[str, np.ndarray] = {}

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self._dimension}")
        return self._model

    @property
    def index(self) -> faiss.Index:
        """Lazy-load or create the FAISS index."""
        if self._index is None:
            if os.path.exists(self.index_path):
                self._index = faiss.read_index(self.index_path)
                self._load_metadata()
                logger.info(
                    f"Loaded FAISS index: {self._index.ntotal} vectors"
                )
            else:
                self._index = faiss.IndexFlatIP(self._dimension)
                logger.info("Created new FAISS index (Inner Product)")
        return self._index

    # ─── Chunking ─────────────────────────────────────────────────────

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = None,
        overlap: int = None,
    ) -> List[str]:
        """
        Split text into overlapping chunks by approximate word count.
        Uses word-level splitting for more natural breaks.
        """
        chunk_size = chunk_size or config.CHUNK_SIZE
        overlap = overlap or config.CHUNK_OVERLAP

        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    # ─── Add / Index ──────────────────────────────────────────────────

    def add_text(
        self,
        text: str,
        chapter: int,
        scene: int,
        characters: list = None,
        location: str = "",
        memory_type: str = "plot",
    ) -> int:
        """
        Chunk text and add all chunks to the FAISS index.
        Returns the number of chunks added.
        """
        chunks = self.chunk_text(text)
        if not chunks:
            return 0

        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # For cosine similarity via inner product
        ).astype("float32")

        # Add to FAISS
        self.index.add(embeddings)

        # Store texts and metadata
        for i, chunk in enumerate(chunks):
            meta = ChunkMetadata(
                chapter=chapter,
                scene=scene,
                characters=characters or [],
                location=location,
                memory_type=memory_type,
                chunk_index=len(self._texts),
            )
            self._texts.append(chunk)
            self._metadata.append(meta.to_dict())

        # Persist
        self._save()

        logger.debug(
            f"Added {len(chunks)} chunks from Ch{chapter}/Sc{scene} "
            f"(total: {self.index.ntotal})"
        )
        return len(chunks)

    # ─── Search ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = None,
        chapter_filter: int = None,
    ) -> List[SearchResult]:
        """
        Semantic search over stored content.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            chapter_filter: Optional - only return results from this chapter

        Returns:
            List of SearchResult sorted by relevance
        """
        top_k = top_k or config.TOP_K_RETRIEVAL

        if self.index.ntotal == 0:
            return []

        # Encode query
        query_embedding = self._embedding_cache.get(query)
        if query_embedding is None:
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")
            if len(self._embedding_cache) >= 512:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
            self._embedding_cache[query] = query_embedding

        # Search with extra results if filtering
        search_k = top_k * 3 if chapter_filter is not None else top_k
        search_k = min(search_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._texts):
                continue

            meta = ChunkMetadata.from_dict(self._metadata[idx])

            if chapter_filter is not None and meta.chapter != chapter_filter:
                continue

            results.append(SearchResult(
                text=self._texts[idx],
                score=float(score),
                metadata=meta,
            ))

            if len(results) >= top_k:
                break

        return results

    # ─── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return stats about the vector store."""
        return {
            "total_vectors": self.index.ntotal,
            "total_chunks": len(self._texts),
            "dimension": self._dimension,
            "index_file_exists": os.path.exists(self.index_path),
        }

    # ─── Persistence ──────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        faiss.write_index(self._index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False)
        with open(self.texts_path, "w", encoding="utf-8") as f:
            json.dump(self._texts, f, ensure_ascii=False)

    def _load_metadata(self) -> None:
        """Load texts and metadata from disk."""
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        if os.path.exists(self.texts_path):
            with open(self.texts_path, "r", encoding="utf-8") as f:
                self._texts = json.load(f)
