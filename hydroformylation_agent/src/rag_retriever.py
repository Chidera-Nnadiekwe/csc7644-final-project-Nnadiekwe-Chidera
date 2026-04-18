"""
rag_retriever.py
-----------------
Retrieval-Augmented Generation (RAG) module.

Steps:
  1. Load .txt files from a corpus directory
  2. Split them into overlapping chunks
  3. Embed each chunk using OpenAI text-embedding-3-small
  4. Index them in FAISS (IndexFlatIP with L2-normalised vectors = cosine similarity)
  5. At query time, embed the query and return the top-k most similar chunks

Requirements:
    pip install openai faiss-cpu numpy
"""

# Import necessary libraries
import json
import os
import numpy as np
from pathlib import Path

# Attempt to import FAISS and OpenAI, with graceful degradation if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("[RAG] Warning: faiss-cpu not installed. RAG will return empty results.")
    print("      Install with: pip install faiss-cpu")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[RAG] Warning: openai package not installed. RAG will return placeholder results.")

# Constants for embedding and chunking
EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 400   # words per chunk (matches report: CHUNK_SIZE=400)
CHUNK_OVERLAP = 50    # words of overlap (matches report: CHUNK_OVERLAP=50)
EMBEDDING_DIM = 1536  # dimension for text-embedding-3-small

# RAGRetriever class definition
class RAGRetriever:
    def __init__(self, corpus_dir: str = "data/corpus",
                 index_dir: str = "data/faiss_index"):
        self.corpus_dir = corpus_dir
        # CHANGE: index stored in a dedicated directory, not CWD
        self.index_dir  = Path(index_dir)
        self.index_file = self.index_dir / "index.faiss"
        self.chunks_file = self.index_dir / "chunks.json"
        self.metadata_file = self.index_dir / "metadata.json"

        self.chunks = []
        self.index  = None

        api_key = os.getenv("OPENAI_API_KEY")
        if OPENAI_AVAILABLE and api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            if OPENAI_AVAILABLE and not api_key:
                print("[RAG] Warning: OPENAI_API_KEY not set. Using placeholder embeddings.")
            self.client = None

        # Try to load cached index; build if not found
        if self.index_file.exists() and self.chunks_file.exists():
            self._load_index()
        else:
            self._build_index()

    # Define helper methods for loading texts
    def _load_texts_from_corpus(self) -> list:
        """Read all .txt files from the corpus directory."""
        texts = []
        corpus_path = Path(self.corpus_dir)
        if not corpus_path.exists():
            print(f"[RAG] Corpus directory '{self.corpus_dir}' not found. Creating it.")
            corpus_path.mkdir(parents=True, exist_ok=True)
            return texts

        for filename in sorted(corpus_path.glob("*.txt")):
            content = filename.read_text(encoding="utf-8", errors="ignore").strip()
            if content:
                texts.append({"source": filename.name, "text": content})

        print(f"[RAG] Loaded {len(texts)} document(s) from '{self.corpus_dir}'.")
        return texts

    # Define helper method for chunking text
    def _chunk_text(self, text: str, source: str) -> list:
        """Split a document into overlapping word-based chunks."""
        words  = text.split()
        chunks = []
        start  = 0
        chunk_idx = 0
        while start < len(words):
            end        = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "chunk_id": f"{source}_{chunk_idx}",
                "text":     chunk_text,
                "source":   source,
            })
            chunk_idx += 1
            if end == len(words):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # Define helper method for embedding texts
    def _embed_texts(self, texts: list) -> np.ndarray:
        """Call OpenAI embedding API and return an array of vectors."""
        if not self.client:
            print("[RAG] Using random placeholder embeddings (OpenAI not available).")
            return np.random.rand(len(texts), EMBEDDING_DIM).astype("float32")

        print(f"[RAG] Embedding {len(texts)} chunks via OpenAI API...")
        # Batch to stay within API limits
        all_vecs = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch    = texts[i: i + batch_size]
            response = self.client.embeddings.create(model=EMBED_MODEL, input=batch)
            all_vecs.extend([item.embedding for item in response.data])
        return np.array(all_vecs, dtype="float32")

    # Define method to build the FAISS index
    def _build_index(self) -> None:
        """Build the FAISS index from the corpus and cache it to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        raw_docs = self._load_texts_from_corpus()

        if not raw_docs:
            print("[RAG] No documents to index. Retrieval will return fallback text.")
            print("      Add .txt files to 'data/corpus/' and re-run build_index.py.")
            self.chunks = []
            self.index  = None
            return

        all_chunks = []
        for doc in raw_docs:
            all_chunks.extend(self._chunk_text(doc["text"], doc["source"]))

        print(f"[RAG] Created {len(all_chunks)} chunk(s) total.")
        self.chunks = all_chunks

        texts_only = [c["text"] for c in all_chunks]
        vectors    = self._embed_texts(texts_only)

        if not FAISS_AVAILABLE:
            print("[RAG] FAISS not available — index not built.")
            return

        # Use IndexFlatIP with L2-normalised vectors (cosine similarity)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(vectors)

        # Save index and metadata
        faiss.write_index(self.index, str(self.index_file))
        self.chunks_file.write_text(json.dumps(self.chunks, indent=2))
        metadata = {
            "index_version":   "1.0",
            "embedding_model": EMBED_MODEL,
            "num_chunks":      len(all_chunks),
            "chunk_size":      CHUNK_SIZE,
            "chunk_overlap":   CHUNK_OVERLAP,
            "sources":         [d["source"] for d in raw_docs],
        }
        self.metadata_file.write_text(json.dumps(metadata, indent=2))
        print(f"[RAG] Index saved to '{self.index_file}'.")

    # Define method to load the FAISS index from disk
    def _load_index(self) -> None:
        """Load a previously built FAISS index from disk."""
        if not FAISS_AVAILABLE:
            return
        try:
            self.index  = faiss.read_index(str(self.index_file))
            self.chunks = json.loads(self.chunks_file.read_text())
            print(f"[RAG] Loaded cached index with {len(self.chunks)} chunk(s).")
        except Exception as e:
            print(f"[RAG] Could not load cached index: {e}. Rebuilding...")
            self._build_index()

    #  Retrieval method
    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Embed the query and return the top_k most relevant text chunks.

        Returns:
            list of dicts, each with keys: 'text', 'source', 'score'
        """
        if self.index is None or not self.chunks:
            return [{
                "text": (
                    "No literature corpus available. Add .txt files to 'data/corpus/' "
                    "and run 'python scripts/build_index.py' to enable RAG. "
                    "General knowledge: for Rh-catalyzed hydroformylation, "
                    "increasing CO pressure and using bulky phosphine ligands "
                    "(higher cone angle) tends to improve linear selectivity (L:B ratio)."
                ),
                "source": "fallback_knowledge",
                "score":  0.0,
            }]

        # Embed query
        if self.client:
            response = self.client.embeddings.create(model=EMBED_MODEL, input=[query])
            q_vec    = np.array([response.data[0].embedding], dtype="float32")
        else:
            q_vec = np.random.rand(1, EMBEDDING_DIM).astype("float32")

        # Normalise query vector for cosine similarity
        faiss.normalize_L2(q_vec) if FAISS_AVAILABLE else None

        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(dist)   # cosine similarity score
                results.append(chunk)

        return results

    # Define method to force rebuild of the index (e.g., after adding new papers)
    def rebuild_index(self) -> None:
        """Force a complete rebuild of the index (e.g., after adding new papers)."""
        for f in [self.index_file, self.chunks_file, self.metadata_file]:
            if f.exists():
                f.unlink()
        self._build_index()
