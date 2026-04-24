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

# Try to import FAISS and OpenAI, with graceful degradation if not available
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
EMBED_MODEL     = "text-embedding-3-small"
CHUNK_SIZE      = 400   # words per chunk
CHUNK_OVERLAP   = 50    # words of overlap
EMBEDDING_DIM   = 1536  # dimension for text-embedding-3-small

# IVF index parameters used when corpus >= IVF_THRESHOLD chunks
IVF_NLIST       = 32    # number of Voronoi cells
IVF_NPROBE      = 8     # cells to visit at query time

# RAGRetriever class definition
class RAGRetriever:
    def __init__(self, corpus_dir: str = "data/corpus",
                 index_dir: str = "data/faiss_index"):
        self.corpus_dir = corpus_dir
        # Index stored in a dedicated directory
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
        """Split a document into sentence-boundary-aware overlapping chunks.

        Strategy:
          1. Split the document into sentences using a lightweight regex tokeniser
             (no NLTK dependency required).
          2. Accumulate sentences until the running word count would exceed
             CHUNK_SIZE.  At that boundary, flush the current chunk and slide
             the window back by CHUNK_OVERLAP words (carrying over the tail
             sentences whose total word count is closest to that overlap target).

        This avoids the hard mid-sentence cuts produced by the original
        word-index slicer, which can break key chemical facts across chunks
        and hurt retrieval precision.
        """
        import re

        # Sentence tokenisation 
        # Split on '. ', '! ', '? ', '\n\n' while keeping the delimiter attached
        # to the preceding sentence.
        sentence_endings = re.compile(r'(?<=[.!?])\s+|(?<=\n)\n+')
        raw_sentences = sentence_endings.split(text.strip())
        sentences = [s.strip() for s in raw_sentences if s.strip()]

        if not sentences:
            return []

        # Create overlapping chunks based on sentence boundaries
        chunks: list = []
        chunk_idx   = 0
        window: list[str] = []   # sentence strings in current window
        word_count  = 0

        def flush(window: list[str]) -> None:
            nonlocal chunk_idx
            chunk_text = " ".join(window)
            chunks.append({
                "chunk_id": f"{source}_{chunk_idx}",
                "text":     chunk_text,
                "source":   source,
            })
            chunk_idx += 1

        for sent in sentences:
            sent_wc = len(sent.split())
            # If adding this sentence would overflow, flush first
            if word_count + sent_wc > CHUNK_SIZE and window:
                flush(window)
                overlap_budget = CHUNK_OVERLAP
                keep: list[str] = []
                for s in reversed(window):
                    s_wc = len(s.split())
                    if overlap_budget - s_wc >= 0:
                        keep.insert(0, s)
                        overlap_budget -= s_wc
                    else:
                        break
                window     = keep
                word_count = sum(len(s.split()) for s in window)

            window.append(sent)
            word_count += sent_wc

        # Flush the last partial window
        if window:
            flush(window)

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
        """Build the FAISS index from the corpus and cache it to disk.

        Index type selection:
          - n_chunks < 256  → IndexFlatIP  (exact cosine, no training needed)
          - n_chunks ≥ 256  → IndexIVFFlat (approximate, sub-linear at scale)
        The IVF index is trained on the embedding matrix before adding vectors.
        At query time, nprobe=IVF_NPROBE cells are visited for a good
        precision / speed tradeoff.
        """
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

        print(f"[RAG] Created {len(all_chunks)} chunk(s) total (sentence-boundary chunker).")
        self.chunks = all_chunks

        texts_only = [c["text"] for c in all_chunks]
        vectors    = self._embed_texts(texts_only)

        if not FAISS_AVAILABLE:
            print("[RAG] FAISS not available — index not built.")
            return

        # L2-normalise so inner-product == cosine similarity
        faiss.normalize_L2(vectors)
        n = len(all_chunks)

        if n >= 256:
            # IVF index: train a quantiser, then add vectors
            print(f"[RAG] Building IVFFlat index (nlist={IVF_NLIST}, n={n})...")
            quantiser  = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.index = faiss.IndexIVFFlat(quantiser, EMBEDDING_DIM, IVF_NLIST,
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(vectors)
            self.index.nprobe = IVF_NPROBE
        else:
            print(f"[RAG] Building FlatIP index (n={n}, below IVF threshold of 256).")
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)

        self.index.add(vectors)

        # Save index and metadata
        faiss.write_index(self.index, str(self.index_file))
        self.chunks_file.write_text(json.dumps(self.chunks, indent=2))
        metadata = {
            "index_version":   "2.0",
            "embedding_model": EMBED_MODEL,
            "num_chunks":      len(all_chunks),
            "chunk_size":      CHUNK_SIZE,
            "chunk_overlap":   CHUNK_OVERLAP,
            "chunker":         "sentence_boundary",
            "index_type":      "IVFFlat" if n >= 256 else "FlatIP",
            "ivf_nlist":       IVF_NLIST if n >= 256 else None,
            "ivf_nprobe":      IVF_NPROBE if n >= 256 else None,
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
            # Restore nprobe for IVF indexes
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = IVF_NPROBE
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
