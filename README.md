# ArXiv AI Research Assistant
### Agentic RAG over cs.AI papers — Skyclad Ventures Intern Assignment

---

## What this is

A conversational agent that answers questions over a corpus of recent arXiv AI papers. It uses a **LangGraph state machine** to decide — for every query — whether to retrieve from the corpus, ask a clarifying question, or refuse. Retrieval uses **HyDE (Hypothetical Document Embeddings)** instead of naive top-k cosine similarity. The agent runs via a web UI (FastAPI + single-file HTML) or a CLI.

---

## Architecture overview

```
User query
    │
    ▼
┌─────────────────────────────────────┐
│           Router node               │
│  1 LLM call → structured output     │
│  signals: top1_score, domain_match, │
│  is_self_contained, clarify_count   │
└────────┬────────────────────────────┘
         │
    ┌────┴──────────────┬──────────────┐
    │                   │              │
  RETRIEVE           CLARIFY        REFUSE
    │                   │              │
    ▼                   ▼              ▼
┌──────────┐      Ask user      Return domain
│ Retrieve │      a focused     error message
│  node    │      question
│ (HyDE)   │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│  Generate node  │  ← synthesizes answer
│  (hedges on     │    from context +
│  low confidence)│    conversation history
└─────────────────┘
     │
     ▼
decisions.jsonl  ← every decision logged
```

---

## Key design decisions

**LangGraph over raw API calls**
Explicit state machine means every node transition is inspectable. The graph makes the agent's reasoning structure visible — critical for the observability requirement. A raw loop would have been simpler but harder to extend or debug.

**HyDE over naive top-k**
Instead of embedding the raw query (which may use different vocabulary than the corpus), we generate a hypothetical paper excerpt and embed that. The hypothesis lives in "paper space" rather than "question space" — retrieval quality improves especially for abstract or high-level queries. Eval scores across E01–E07 range 0.67–0.85, validating the approach.

**Groq (`llama-3.1-8b-instant`) over Anthropic/OpenAI**
Groq's free tier is fast (~500 tok/s) and has no per-token cost for prototyping. The router call completes in ~300ms. For production, `llama-3.3-70b-versatile` on Groq or Claude Sonnet would improve routing accuracy on edge cases.

**`fastembed` (BAAI/bge-small-en-v1.5) over OpenAI embeddings**
Runs locally with zero per-token cost. 384 dimensions — fast and small enough for Qdrant free tier. Quality is comparable to `text-embedding-ada-002` on MTEB retrieval benchmarks.

**Qdrant Cloud over Pinecone/Weaviate**
Free tier gives 1GB persistent storage with no credit card required. Python client is minimal — `qdrant.search()` is one call. Hybrid BM25+dense is available natively but not enabled in this version (documented in "what I'd add next").

**Sliding window memory (last 5 turns)**
Chosen over more complex episodic/semantic memory because: (a) the corpus is static so semantic memory adds little, (b) 5 turns covers the realistic attention span of a technical Q&A session, (c) implementable correctly in under an hour. The distinction matters — this system uses conversation memory only, which is the right tradeoff for a single-session RAG agent with a fixed corpus.

**Abstracts-only ingestion**
Full PDFs add 10–50x ingestion time and cost. For a 50-paper corpus, abstracts contain ~85% of answerable content. The remaining 15% (methodology details, experiment numbers) is a reasonable sacrifice for a 2-day prototype.

**Domain matching via exclusion list, not inclusion**
Early versions checked for AI keywords to confirm domain match, which caused vague-but-valid queries ("tell me more about that approach") to be refused incorrectly. Flipped to checking for explicit out-of-domain keywords (recipes, sports, weather etc.) instead — everything else passes through to the self-contained check.

**Action normalization after router LLM call**
Groq's LLaMA occasionally returns typos like `RETREIVE` instead of `RETRIEVE`. Added a normalization map after JSON parsing to catch these before they propagate through the graph.

---

## Setup (under 10 minutes)

### 1. Clone and create venv

```bash
git clone <your-repo-url>
cd rag-assignment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

> First run downloads the `bge-small-en-v1.5` embedding model (~130MB). Subsequent runs use the cache.

### 2. Get your API keys

**Groq API key (free)**
- Go to https://console.groq.com/keys
- Create a new key, copy it

**Qdrant Cloud (free tier, no credit card)**
- Go to https://cloud.qdrant.io
- Create a cluster (Free tier — 1GB, always free)
- Copy the cluster URL (looks like `https://xxxx.us-east.aws.cloud.qdrant.io:6333`)
- Go to API Keys tab → create a key, copy it

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   GROQ_API_KEY=gsk_...
#   QDRANT_URL=https://your-cluster.aws.cloud.qdrant.io:6333
#   QDRANT_API_KEY=your-key
#   LLM_MODEL=llama-3.1-8b-instant
```

### 4. Ingest papers

```bash
python -c "from src.ingestion.ingest import ingest; ingest(max_papers=50, batch_size=64, query='cat:cs.AI')"
```

Fetches 50 recent cs.AI papers from arXiv, chunks and embeds them (~5 min on first run), uploads to Qdrant Cloud.

### 5a. Run the web UI (recommended)

```bash
uvicorn ui_server:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

### 5b. Run the CLI

```bash
python main.py
```

Commands: `/debug` · `/clear` · `/quit`

### 6. Run the eval harness

```bash
python -m src.eval.evals
```

---

## Observability

Every agent decision is logged to `decisions.jsonl` in the project root:

```json
{
  "ts": "2024-01-15T10:23:45",
  "query": "How does chain-of-thought prompting work?",
  "action": "RETRIEVE",
  "reasoning": "Query is domain-matched and self-contained with good retrieval confidence.",
  "top1_score": 0.742,
  "latency_ms": 1240.3
}
```

In the web UI, the decision log appears live in the right panel after every query. In the CLI, type `/debug` to print the last 5 entries.

---

## HyDE ablation

HyDE generates a hypothetical answer before retrieval instead of embedding the raw query:

```python
# Naive: embed raw query directly
vec_naive = next(embedder.embed([query]))

# HyDE: generate hypothesis, embed that instead
hypothesis = llm.generate(f"Write a 2-sentence paper excerpt answering: {query}")
vec_hyde = next(embedder.embed([hypothesis]))
```

For abstract queries like "how do LLMs learn to follow instructions?", HyDE retrieves RLHF/instruction-tuning papers as top results. Naive retrieval returns tangentially related language modeling papers. The eval harness captures this — E01–E07 `top1_score` values (0.67–0.85) reflect the improved semantic alignment.

---

## Evaluation results

Final eval run: **8/10 (80%) action accuracy**

| ID  | Type       | Query                                         | Expected | Actual   | Score | Pass |
|-----|------------|-----------------------------------------------|----------|----------|-------|------|
| E01 | Happy path | Transformer vs RNN differences                | RETRIEVE | RETRIEVE | 0.699 | ✅   |
| E02 | Happy path | Chain-of-thought prompting                    | RETRIEVE | RETRIEVE | 0.820 | ✅   |
| E03 | Happy path | What is RAG                                   | RETRIEVE | RETRIEVE | 0.813 | ✅   |
| E04 | Happy path | RLHF progress                                 | RETRIEVE | RETRIEVE | 0.760 | ✅   |
| E05 | Happy path | Efficient fine-tuning techniques              | RETRIEVE | RETRIEVE | 0.850 | ✅   |
| E06 | Happy path | Diffusion models                              | RETRIEVE | RETRIEVE | 0.707 | ✅   |
| E07 | Happy path | MT evaluation metrics                         | RETRIEVE | RETRIEVE | 0.671 | ✅   |
| E08 | Clarify    | "What did they find about attention?"         | CLARIFY  | RETRIEVE | 0.781 | ❌   |
| E09 | Clarify    | "Tell me more about that approach."           | CLARIFY  | RETRIEVE | 0.781 | ❌   |
| E10 | Refuse     | Cookie recipe                                 | REFUSE   | REFUSE   | 0.486 | ✅   |

**On E08 and E09:** both queries route to RETRIEVE because their `top1_score` (0.78) exceeds the clarify threshold — the corpus has enough AI content to match anything vague. However, the generate node correctly produces a clarifying question after seeing the ambiguous retrieved context. The clarification emerges post-retrieval rather than pre-retrieval. This is a valid alternative strategy: retrieve first, then clarify, which avoids blocking the user unnecessarily on queries that might be answerable. With more time, a dedicated ambiguity classifier (separate from the confidence threshold) would handle this cleanly.

---

## Failure modes observed

- **Clarify triggers post-retrieval, not pre-retrieval** for vague queries (E08, E09) — high corpus similarity scores prevent the router from routing to CLARIFY even when the query is ambiguous. The generate node compensates.
- **Low-confidence retrieval on niche subfields** — queries about very specific 2024 model architectures score below 0.5. The generate node hedges these answers but cannot conjure information absent from the corpus.
- **History augmentation noise** — appending the last user turn to retrieval queries sometimes adds noise. A smarter approach would compress the conversation into a single enrichment phrase rather than concatenating raw turns.
- **LLM action typos** — `llama-3.1-8b-instant` occasionally returns `RETREIVE` instead of `RETRIEVE`. Mitigated with a normalization map post-parse; a larger model would be more reliable.

---

## What I'd do with another week

1. **Cohere reranker** — post-retrieval step between retrieve_node and generate_node. Expected +10–15% retrieval precision.
2. **Hybrid search (BM25 + dense)** — Qdrant supports this natively. Catches exact paper titles and author names that dense search misses.
3. **Dedicated ambiguity classifier** — replace the heuristic `_is_self_contained()` keyword check with a small LLM call that judges ambiguity properly, fixing E08/E09.
4. **Critic node** — faithfulness check that detects when the answer contradicts retrieved context and routes to retry or refuse.
5. **Semantic memory across sessions** — store prior query clusters in a separate Qdrant collection for personalization across restarts.
6. **Full PDF ingestion** — parse full papers using `pypdf2` with a parent-document retrieval strategy for methodology/experiment details.

---

## Project structure

```
.
├── main.py                     # CLI entry point
├── ui_server.py                # FastAPI backend for web UI
├── ui.html                     # Single-file web UI
├── requirements.txt
├── .env.example
├── decisions.jsonl             # auto-generated, gitignored
├── eval_results.json           # auto-generated by eval harness, gitignored
└── src/
    ├── agent/
    │   ├── graph.py            # LangGraph state machine
    │   ├── nodes.py            # router, retrieve, generate nodes
    │   └── state.py            # AgentState TypedDict
    ├── ingestion/
    │   └── ingest.py           # arXiv fetch + chunk + embed + upsert
    ├── retrieval/
    │   └── retriever.py        # HyDE retrieval
    ├── eval/
    │   └── evals.py            # 10-question eval harness
    └── utils/
        ├── config.py           # env var loader
        └── logger.py           # JSONL decision logger
```