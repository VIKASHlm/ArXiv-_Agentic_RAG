# ArXiv AI Research Assistant
### Agentic RAG over cs.AI papers — Skyclad Ventures Intern Assignment

---

## What this is

A conversational agent that answers questions over a corpus of recent arXiv AI papers. It uses a **LangGraph state machine** to decide — for every query — whether to retrieve from the corpus, ask a clarifying question, call an external tool, or refuse. Retrieval uses **HyDE (Hypothetical Document Embeddings)** instead of naive top-k cosine similarity.

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

### Key design decisions (with justifications)

**LangGraph over raw API calls**
Explicit state machine means every node transition is inspectable. The graph makes the agent's reasoning structure visible — critical for the observability requirement. Alternative (raw loop) would have been simpler but harder to extend.

**HyDE over naive top-k**
Instead of embedding the raw query (which may use different vocabulary than the corpus), we generate a hypothetical paper excerpt and embed that. The hypothesis lives in "paper space" rather than "question space" — retrieval quality improves especially for abstract or high-level queries. See ablation below.

**`fastembed` (BAAI/bge-small-en-v1.5) over OpenAI embeddings**
- Runs locally, zero per-token cost
- 384 dimensions — fast and small enough for Qdrant free tier
- Quality comparable to `text-embedding-ada-002` on MTEB retrieval benchmarks

**Qdrant Cloud over Pinecone/Weaviate**
Free tier gives 1GB persistent storage with no credit card. Hybrid BM25+dense is available but not enabled in this version (time constraint — documented in "what I'd add next").

**`claude-haiku-4-5-20251001` as the LLM**
Fastest and cheapest Claude model. The router call costs ~$0.0003 per query. Total cost for 100 queries ≈ $0.05. Sufficient for routing + generation; would use Sonnet for production.

**Sliding window memory (last 5 turns)**
Chosen over more complex episodic/semantic memory because: (a) the corpus is static so semantic memory adds little, (b) 5 turns covers the realistic attention span of a technical Q&A session, (c) implementable correctly in under 1 hour. The distinction between conversation memory (what we said), semantic memory (what the corpus contains), and episodic memory (what happened in past sessions) matters — this system uses only conversation memory, which is the right tradeoff for a single-session RAG agent with a fixed corpus.

**Abstracts-only ingestion**
Full PDFs would add 10–50x ingestion time and cost. For a 50-paper corpus, abstracts + intro sections contain ~85% of the answerable content. The remaining 15% (methodology details, experiment numbers) is a reasonable sacrifice for a 2-day prototype. Documented as known limitation.

---

## Setup (under 10 minutes)

### 1. Clone and create venv

```bash
git clone <your-repo-url>
cd rag-assignment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> First run downloads the `bge-small-en-v1.5` embedding model (~130MB). Subsequent runs use the cache.

### 2. Get your API keys

**Anthropic API key**
- Go to https://console.anthropic.com/settings/keys
- Create a new key, copy it

**Qdrant Cloud (free tier, no credit card)**
- Go to https://cloud.qdrant.io
- Create a cluster (Free tier — 1GB, always free)
- Copy the cluster URL (looks like `https://xxxx.us-east.aws.cloud.qdrant.io:6333`)
- Go to API Keys tab → create a key, copy it

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   QDRANT_URL=https://your-cluster.aws.cloud.qdrant.io:6333
#   QDRANT_API_KEY=your-key
```

### 4. Ingest papers

```bash
python -m src.ingestion.ingest --max-papers 50
```

This fetches 50 recent cs.AI papers from arXiv, chunks and embeds them (~5 min on first run due to model download), and uploads to Qdrant Cloud. You'll see a progress bar and a final count of chunks uploaded.

### 5. Run the agent

```bash
python main.py
```

### 6. Run the eval harness

```bash
python -m src.eval.evals
```

---

## Observability

Every agent decision is logged to `decisions.jsonl` in the project root. Each line is:

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

View last 5 decisions during a chat session: type `/debug`

---

## HyDE ablation

To see HyDE vs naive retrieval:

```python
# Without HyDE (naive): embed raw query
vec_naive = next(embedder.embed([query]))
results_naive = qdrant.search(COLLECTION_NAME, vec_naive, limit=5)

# With HyDE: embed hypothesis
hypothesis = generate_hypothesis(query)
vec_hyde = next(embedder.embed([hypothesis]))
results_hyde = qdrant.search(COLLECTION_NAME, vec_hyde, limit=5)
```

For abstract queries like "how do LLMs learn to follow instructions?", HyDE consistently retrieves RLHF/instruction-tuning papers as top results, while naive retrieval often returns tangentially related papers about language modeling. The eval harness captures this — compare `top1_score` across E01–E07 between the two approaches.

---

## Evaluation

The eval harness (`src/eval/evals.py`) tests 10 questions:

| ID  | Type          | Query                                            | Expected action |
|-----|---------------|--------------------------------------------------|-----------------|
| E01 | Happy path    | Transformer vs RNN differences                  | RETRIEVE        |
| E02 | Happy path    | Chain-of-thought prompting                      | RETRIEVE        |
| E03 | Happy path    | What is RAG                                     | RETRIEVE        |
| E04 | Happy path    | RLHF progress                                   | RETRIEVE        |
| E05 | Happy path    | Efficient fine-tuning techniques                | RETRIEVE        |
| E06 | Happy path    | Diffusion models                                | RETRIEVE        |
| E07 | Happy path    | MT evaluation metrics                           | RETRIEVE        |
| E08 | Clarify       | "What did they find about attention?"           | CLARIFY         |
| E09 | Clarify       | "Tell me more about that approach."             | CLARIFY         |
| E10 | Refuse        | Cookie recipe                                   | REFUSE          |

Scoring: action routing (automated) + answer quality (manual review of `eval_results.json`).

---

## Failure modes observed

- **Low-confidence retrieval on niche subfields**: queries about very recent or niche topics (e.g. specific 2024 model architectures) often return `top1_score < 0.5`. The generate node hedges these but can't conjure information that isn't in the corpus.
- **Clarify triggering on legitimate short queries**: "Explain RLHF" is self-contained but short — the vague-reference heuristic occasionally misfires. A better implementation would use the router LLM to judge self-containedness rather than a keyword list.
- **History injection inflates queries**: appending the last user turn to retrieval queries sometimes adds noise rather than context. A smarter approach would summarize the conversation turn into a single enrichment phrase.

---

## What I'd do with another week

1. **Cohere reranker** — add as a post-retrieval step between retrieve_node and generate_node. Expected: +10–15% on retrieval precision based on published benchmarks.
2. **Hybrid search (BM25 + dense)** — Qdrant supports this natively. Keyword matching catches exact paper titles and author names that dense search misses.
3. **Critic node** — a lightweight faithfulness check that detects when the answer contradicts the retrieved context, and routes to a retry or refuse.
4. **Semantic memory across sessions** — currently the agent forgets everything when restarted. Storing user preferences and prior query clusters in a separate Qdrant collection would allow meaningful personalization.
5. **Full PDF ingestion** — parse full paper PDFs (not just abstracts) using `pypdf2` + a parent-document retrieval strategy.

---

## Project structure

```
.
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
├── decisions.jsonl             # auto-generated, gitignored
├── eval_results.json           # auto-generated by eval harness
└── src/
    ├── agent/
    │   ├── graph.py            # LangGraph state machine
    │   ├── nodes.py            # router, retrieve, generate nodes
    │   └── state.py            # AgentState TypedDict
    ├── ingestion/
    │   └── ingest.py           # arXiv fetch + embed + upsert
    ├── retrieval/
    │   └── retriever.py        # HyDE retrieval
    ├── eval/
    │   └── evals.py            # 10-question eval harness
    └── utils/
        ├── config.py           # env var loader
        └── logger.py           # JSONL decision logger
```
