"""
build_index.py
--------------
One-time script to chunk literature .txt files, embed them with OpenAI's
text-embedding-3-small, and persist the resulting FAISS index to disk.

Run once before starting the agent (from the project root):
    python scripts/build_index.py
"""

# Import necessary libraries
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import openai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_index")


# Project root and default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"    
DEFAULT_INDEX_DIR  = PROJECT_ROOT / "data" / "faiss_index"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536


# Chunking functions 
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Split text into overlapping word-count chunks.

    Parameters
    ----------
    text : str
        Full document text.
    chunk_size : int
        Approximate number of words per chunk.
    overlap : int
        Number of words shared between consecutive chunks.

    Returns
    -------
    list of str
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks

# Function to load literature files
def load_corpus(corpus_dir: Path) -> List[dict]:
    """Load all .txt files from the corpus directory.

    Returns
    -------
    list of dict, each with 'source' (filename) and 'text' (full file content).
    """
    docs      = []
    txt_files = sorted(corpus_dir.glob("*.txt"))
    if not txt_files:
        logger.warning("No .txt files found in %s", corpus_dir)
    for path in txt_files:
        logger.info("Loading: %s", path.name)
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append({"source": path.name, "text": text})
    return docs


# Embedding function
def embed_chunks(texts: List[str], batch_size: int = 50) -> np.ndarray:
    """Embed a list of text strings using OpenAI's embedding API.

    Returns
    -------
    np.ndarray of shape (n_chunks, EMBEDDING_DIM), dtype float32.
    """
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    all_embeddings = []

    # Process texts in batches to respect API limits and improve efficiency
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        logger.info(
            "Embedding batch %d-%d of %d ...",
            i + 1,
            min(i + batch_size, len(texts)),
            len(texts),
        )
        response       = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        batch_vecs     = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)

    return np.array(all_embeddings, dtype="float32")



# FAISS index construction 
def build_faiss_index(embeddings: np.ndarray):
    """Create a FAISS flat inner-product index from L2-normalised embeddings.

    Parameters
    ----------
    embeddings : np.ndarray of shape (n_chunks, EMBEDDING_DIM)

    Returns
    -------
    faiss.Index (IndexFlatIP, cosine similarity via L2-normalisation)
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required. Install: pip install faiss-cpu") from exc

    # L2-normalise embeddings for cosine similarity search
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    logger.info("FAISS index built with %d vectors.", index.ntotal)
    return index


# Main function to orchestrate the pipeline
def main(
    corpus_dir: Path  = DEFAULT_CORPUS_DIR,
    index_dir: Path   = DEFAULT_INDEX_DIR,
    chunk_size: int   = 400,               
    chunk_overlap: int = 50,
) -> None:
    """Full pipeline: load -> chunk -> embed -> index -> save."""
    index_dir.mkdir(parents=True, exist_ok=True)

    # Load documents
    docs = load_corpus(corpus_dir)
    if not docs:
        logger.error("No literature files found in %s. Aborting.", corpus_dir)
        sys.exit(1)

    # Chunk documents
    all_chunks: List[dict] = []
    for doc in docs:
        text_chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "chunk_id": f"{doc['source']}_{i}",
                "source":   doc["source"],
                "text":     chunk,
            })
    logger.info("Total chunks created: %d", len(all_chunks))

    # Embed chunks
    texts      = [c["text"] for c in all_chunks]
    embeddings = embed_chunks(texts)

    # Build FAISS index
    import faiss as faiss_mod
    index = build_faiss_index(embeddings)

    # Persist index
    index_path = index_dir / "index.faiss"
    faiss_mod.write_index(index, str(index_path))
    logger.info("FAISS index saved to %s", index_path)

    # Persist chunk metadata
    chunks_path = index_dir / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as fh:
        json.dump(all_chunks, fh, indent=2)
    logger.info("Chunk metadata saved to %s", chunks_path)

    # Write index metadata
    metadata_path = index_dir / "metadata.json"
    metadata = {
        "index_version":   "1.0",
        "embedding_model": EMBEDDING_MODEL,
        "num_chunks":      len(all_chunks),
        "chunk_size":      chunk_size,
        "chunk_overlap":   chunk_overlap,
        "sources":         [d["source"] for d in docs],
    }
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Index metadata written to %s", metadata_path)
    logger.info("Index build complete. Run 'python src/agent_controller.py' to start the agent.")

# Entry point for command-line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS literature index.")
    parser.add_argument(
        "--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
        help="Directory containing .txt corpus files (default: data/corpus/)"
    )
    parser.add_argument(
        "--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
        help="Output directory for FAISS index files (default: data/faiss_index/)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=400,
        help="Words per chunk (default: 400)"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=50,
        help="Overlapping words between chunks (default: 50)"
    )
    args = parser.parse_args()
    main(
        corpus_dir=args.corpus_dir,
        index_dir=args.index_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
