"""
Ingestion pipeline: arXiv API → chunk → embed (fastembed) → Qdrant Cloud

Run once:
    python -m src.ingestion.ingest --max-papers 50

Takes ~5-10 min on first run (model download + embedding).
Subsequent runs skip already-ingested papers.
"""

import hashlib
import time
from typing import Generator

import arxiv
import typer
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from rich.console import Console
from rich.progress import track

from src.utils.config import (
    COLLECTION_NAME,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    QDRANT_API_KEY,
    QDRANT_URL,
)

console = Console()
app = typer.Typer()


# ── Helpers ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def make_id(text: str) -> str:
    """Stable UUID-compatible int id from text hash."""
    return int(hashlib.md5(text.encode()).hexdigest()[:16], 16)


def fetch_papers(query: str, max_results: int) -> list[dict]:
    """Fetch papers from arXiv API. Returns list of dicts."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    papers = []
    for result in client.results(search):
        papers.append(
            {
                "id": result.entry_id.split("/")[-1],
                "title": result.title,
                "abstract": result.summary.replace("\n", " "),
                "authors": [a.name for a in result.authors[:5]],
                "published": result.published.isoformat(),
                "url": result.entry_id,
                "categories": result.categories,
            }
        )
    return papers


def build_points(
    papers: list[dict], embedder: TextEmbedding
) -> Generator[PointStruct, None, None]:
    """Chunk papers, embed, yield PointStructs."""
    for paper in papers:
        # combine title + abstract for richer context
        full_text = f"{paper['title']}\n\n{paper['abstract']}"
        chunks = chunk_text(full_text)

        texts = list(chunks)
        embeddings = list(embedder.embed(texts))

        for i, (chunk, vec) in enumerate(zip(texts, embeddings)):
            point_id = make_id(f"{paper['id']}_{i}") % (2**63)
            yield PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload={
                    "paper_id": paper["id"],
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "published": paper["published"],
                    "url": paper["url"],
                    "categories": paper["categories"],
                    "chunk_index": i,
                    "text": chunk,
                },
            )


# ── Main ──────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    max_papers: int = typer.Option(50, help="Number of papers to fetch"),
    query: str = typer.Option("cat:cs.AI", help="arXiv query string"),
    batch_size: int = typer.Option(64, help="Qdrant upsert batch size"),
):
    console.print(f"[bold]Fetching {max_papers} papers from arXiv...[/bold]")
    papers = fetch_papers(query, max_papers)
    console.print(f"[green]Fetched {len(papers)} papers[/green]")

    console.print(f"[bold]Loading embedding model: {EMBEDDING_MODEL}[/bold]")
    embedder = TextEmbedding(EMBEDDING_MODEL)

    console.print("[bold]Connecting to Qdrant...[/bold]")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Create collection if not exists
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        console.print(f"[green]Created collection: {COLLECTION_NAME}[/green]")
    else:
        console.print(f"[yellow]Collection already exists: {COLLECTION_NAME}[/yellow]")

    # Build and upsert in batches
    console.print("[bold]Embedding and uploading...[/bold]")
    batch: list[PointStruct] = []
    total_chunks = 0

    for point in track(build_points(papers, embedder), description="Processing..."):
        batch.append(point)
        if len(batch) >= batch_size:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
            total_chunks += len(batch)
            batch = []

    if batch:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
        total_chunks += len(batch)

    console.print(f"[bold green]Done! Uploaded {total_chunks} chunks from {len(papers)} papers.[/bold green]")
    console.print(f"Collection: [bold]{COLLECTION_NAME}[/bold] on Qdrant Cloud")


if __name__ == "__main__":
    app()
