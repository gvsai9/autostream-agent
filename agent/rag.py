"""
RAG Pipeline for AutoStream Agent
-----------------------------------
Loads the local knowledge base, chunks it, embeds it using a
lightweight HuggingFace model (runs 100% locally, no API cost),
and stores it in a FAISS vector index.

At query time, retrieves the top-K most relevant chunks so the
LLM can answer accurately without hallucinating.
"""

import os
import logging
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# Path constants
KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.md")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")

# Lightweight embedding model — runs locally, no API key needed
# all-MiniLM-L6-v2 is fast, small (~80MB), and excellent for semantic search
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store() -> FAISS:
    """
    Loads the knowledge base markdown, splits it by headers,
    embeds each chunk, and stores in FAISS.
    Called once at startup; saves index to disk for reuse.
    """
    logger.info("[RAG] Building vector store from knowledge base...")

    # Load raw markdown file
    loader = TextLoader(KB_PATH, encoding="utf-8")
    docs = loader.load()

    # RecursiveCharacterTextSplitter: industry standard for small KBs
    # chunk_size=400 keeps each chunk focused; overlap=50 preserves context at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)
    logger.info(f"[RAG] Split knowledge base into {len(split_docs)} chunks.")

    # Load local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Build FAISS index and save to disk
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(INDEX_PATH)

    logger.info(f"[RAG] Vector store built and saved to {INDEX_PATH}")
    return vector_store


def load_vector_store() -> FAISS:
    """
    Loads FAISS index from disk if it exists.
    Builds it fresh if not (first run).
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    if os.path.exists(INDEX_PATH):
        logger.info("[RAG] Loading existing FAISS index from disk...")
        return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return build_vector_store()


def retrieve_context(query: str, k: int = 2) -> str:
    """
    Main retrieval function called by the agent.

    Args:
        query: The user's question or message
        k: Number of top chunks to retrieve (2 is enough for focused answers)

    Returns:
        A single string of the most relevant KB content for the LLM to use
    """
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the knowledge base."

    # Combine chunks into one context block
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    logger.info(f"[RAG] Retrieved {len(docs)} chunks for query: '{query[:50]}...'")
    return context


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("\n=== RAG Smoke Test ===\n")

    test_queries = [
        "What is the price of the Pro plan?",
        "Is there a refund policy?",
        "What support do I get on Basic plan?",
    ]

    for q in test_queries:
        print(f"Query: {q}")
        result = retrieve_context(q)
        print(f"Retrieved:\n{result}\n{'-'*60}\n")
