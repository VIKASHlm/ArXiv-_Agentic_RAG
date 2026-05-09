"""
Retrieval module: HyDE (Hypothetical Document Embedding)

Instead of embedding the raw query, we:
1. Ask the LLM to generate a fake 2-sentence paper excerpt that would answer the query
2. Embed that hypothesis
3. Retrieve real chunks using the hypothesis vector

This is the non-naive retrieval technique for the assignment.
"""

from fastembed import TextEmbedding
from groq import Groq
from qdrant_client import QdrantClient

from src.utils.config import (
    GROQ_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
    RETRIEVAL_TOP_K,
)

# Singletons — initialized once, reused across calls
_embedder: TextEmbedding | None = None
_qdrant: QdrantClient | None = None
_groq: Groq | None = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(EMBEDDING_MODEL)
    return _embedder


def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=GROQ_API_KEY)
    return _groq


def generate_hypothesis(query: str) -> str:
    """Generate a hypothetical paper excerpt (HyDE step 1)."""
    client = get_groq()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a 2-sentence excerpt from an AI research paper that directly answers: '{query}'\n"
                    "Write only the excerpt, no preamble."
                ),
            }
        ],
    )
    return response.choices[0].message.content.strip()


def retrieve(query: str) -> tuple[list[dict], float, str]:
    """
    HyDE retrieval.

    Returns:
        chunks: list of payload dicts from Qdrant
        top1_score: cosine similarity of best match (0-1)
        hypothesis: the generated hypothesis text (for logging)
    """
    hypothesis = generate_hypothesis(query)
    embedder = get_embedder()
    vec = next(embedder.embed([hypothesis]))

    qdrant = get_qdrant()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec.tolist(),
        limit=RETRIEVAL_TOP_K,
        with_payload=True,
    )

    if not results:
        return [], 0.0, hypothesis

    chunks = [r.payload for r in results]
    top1_score = results[0].score
    return chunks, top1_score, hypothesis


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not chunks:
        return "No relevant context found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Title: {chunk.get('title', 'Unknown')}\n"
            f"    Published: {chunk.get('published', '')[:10]}\n"
            f"    Text: {chunk.get('text', '')}"
        )
    return "\n\n".join(parts)
