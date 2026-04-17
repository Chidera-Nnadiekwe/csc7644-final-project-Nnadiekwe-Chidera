"""
rag_retriever.py
-----------------
Retrieval-Augmented Generation (RAG) module.

Steps:
  1. Load .txt files from a corpus directory
  2. Split them into overlapping chunks
  3. Embed each chunk using OpenAI text-embedding-3-small
  4. Index them in FAISS for fast nearest-neighbor search
  5. At query time, embed the query and return the top-k most similar chunks

Requirements:
    pip install openai faiss-cpu numpy

The index is built once and cached on disk (corpus_index.faiss + corpus_chunks.json)
so you don't re-embed on every run.
"""

import json
import os
import numpy as np

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

EMBED_MODEL = "text-embedding-3-small"
INDEX_FILE  = "corpus_index.faiss"
CHUNKS_FILE = "corpus_chunks.json"
CHUNK_SIZE  = 400    # words per chunk
CHUNK_OVERLAP = 50   # words of overlap between chunks


class RAGRetriever:
    def __init__(self, corpus_dir: str = "corpus/"):
        self.corpus_dir = corpus_dir
        self.chunks = []
        self.index = None

        if OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            self.client = None

        # Try to load cached index; build if not found
        if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
            self._load_index()
        else:
            self._build_index()

    # ─────────────────────────────────────────────
    #  Index Building
    # ─────────────────────────────────────────────

    def _load_texts_from_corpus(self) -> list:
        """Read all .txt files from the corpus directory."""
        texts = []
        if not os.path.exists(self.corpus_dir):
            print(f"[RAG] Corpus directory '{self.corpus_dir}' not found. Creating empty folder.")
            os.makedirs(self.corpus_dir, exist_ok=True)
            return texts

        for filename in os.listdir(self.corpus_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.corpus_dir, filename)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if content:
                    texts.append({"source": filename, "text": content})

        print(f"[RAG] Loaded {len(texts)} documents from '{self.corpus_dir}'.")
        return texts

    def _chunk_text(self, text: str, source: str) -> list:
        """Split a document into overlapping word-based chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({"text": chunk_text, "source": source})
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _embed_texts(self, texts: list) -> np.ndarray:
        """Call OpenAI embedding API and return an array of vectors."""
        if not self.client:
            # Return random vectors as placeholder when OpenAI not available
            print("[RAG] Using random placeholder embeddings (OpenAI not available).")
            return np.random.rand(len(texts), 1536).astype("float32")

        print(f"[RAG] Embedding {len(texts)} chunks via OpenAI API...")
        response = self.client.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype="float32")

    def _build_index(self):
        """Build the FAISS index from the corpus and cache it to disk."""
        raw_docs = self._load_texts_from_corpus()

        if not raw_docs:
            print("[RAG] No documents to index. Retrieval will return empty results.")
            print("      Add .txt files to the 'corpus/' folder and re-run to enable RAG.")
            self.chunks = []
            self.index = None
            return

        # Chunk all documents
        all_chunks = []
        for doc in raw_docs:
            chunks = self._chunk_text(doc["text"], doc["source"])
            all_chunks.extend(chunks)

        print(f"[RAG] Created {len(all_chunks)} chunks total.")
        self.chunks = all_chunks

        # Embed chunks
        texts_only = [c["text"] for c in all_chunks]
        vectors = self._embed_texts(texts_only)

        # Build FAISS index (L2 distance; cosine would require normalization)
        if FAISS_AVAILABLE:
            dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(vectors)

            # Save to disk
            faiss.write_index(self.index, INDEX_FILE)
            with open(CHUNKS_FILE, "w") as f:
                json.dump(self.chunks, f, indent=2)
            print(f"[RAG] Index saved to '{INDEX_FILE}' and '{CHUNKS_FILE}'.")
        else:
            print("[RAG] FAISS not available — index not built.")

    def _load_index(self):
        """Load a previously built FAISS index from disk."""
        if not FAISS_AVAILABLE:
            return
        try:
            self.index = faiss.read_index(INDEX_FILE)
            with open(CHUNKS_FILE, "r") as f:
                self.chunks = json.load(f)
            print(f"[RAG] Loaded cached index with {len(self.chunks)} chunks.")
        except Exception as e:
            print(f"[RAG] Could not load cached index: {e}. Rebuilding...")
            self._build_index()

    # ─────────────────────────────────────────────
    #  Retrieval
    # ─────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Embed the query and return the top_k most relevant text chunks.
        Each result is a dict with keys: 'text', 'source'.
        """
        if self.index is None or not self.chunks:
            # Graceful fallback when no corpus is available
            return [{
                "text": (
                    "No literature corpus available. "
                    "Add .txt files to the 'corpus/' folder to enable RAG. "
                    "General knowledge: for Rh-catalyzed hydroformylation, "
                    "increasing CO pressure and using bulky phosphine ligands "
                    "tends to improve linear selectivity (higher L:B ratio)."
                ),
                "source": "fallback_knowledge"
            }]

        # Embed the query
        if self.client:
            response = self.client.embeddings.create(model=EMBED_MODEL, input=[query])
            q_vec = np.array([response.data[0].embedding], dtype="float32")
        else:
            q_vec = np.random.rand(1, 1536).astype("float32")

        # Search the FAISS index
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results

    def rebuild_index(self):
        """Force a complete rebuild of the index (e.g., after adding new papers)."""
        if os.path.exists(INDEX_FILE):
            os.remove(INDEX_FILE)
        if os.path.exists(CHUNKS_FILE):
            os.remove(CHUNKS_FILE)
        self._build_index()
