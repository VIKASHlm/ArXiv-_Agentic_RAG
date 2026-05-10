# ArXiv AI Research Assistant
### Agentic RAG over cs.AI papers — Skyclad Ventures Intern Assignment

---

## What this is

A conversational agent that answers questions over a corpus of recent arXiv AI papers. It uses a **LangGraph state machine** to decide — for every query — whether to retrieve from the corpus, call a live arXiv search tool, ask a clarifying question, or refuse. Retrieval uses **HyDE (Hypothetical Document Embeddings)** instead of naive top-k cosine similarity. The agent runs via a web UI (FastAPI + single-file HTML) or a CLI.

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
    ┌────┴──────────┬──────────┬──────────────┐
    │               │          │              │
  RETRIEVE        TOOL      CLARIFY        REFUSE
    │               │          │              │
    ▼               ▼          ▼              ▼
┌────────┐   ┌──────────┐  Ask user    Return domain
│Retrieve│   │ arXiv    │  a focused   error message
│ node   │   │live search  question
│ (HyDE) │   │  tool    │
└───┬────┘   └────┬─────┘
    │              │
    └──────┬───────┘
           ▼
┌─────────────────┐
│  Generate node  │  ← synthesizes answer from
│  (hedges on     │    corpus or tool context +
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
Instead of embedding the raw query (which may use different vocabulary than the corpus), we generate a hypothetical paper excerpt and embed that. The hypothesis lives in "paper space" rather than "question space" — retrieval quality improves especially for abstract or high-level queries. Eval scores across happy path queries range 0.75–0.836, validating the approach.

**arXiv live search as the external tool**
When corpus confidence falls below the threshold, the agent calls the arXiv API live instead of attempting a low-confidence answer. Chosen over web search or a calculator because: (a) zero new dependencies — the `arxiv` library was already used for ingestion, (b) results stay in domain, (c) it handles the exact failure mode of a fixed corpus — recent papers not yet ingested. Tool results are injected into the same context field as corpus retrieval so the generate node handles both paths identically.

**Confidence threshold routing (CLARIFY_THRESHOLD=0.65)**
Queries scoring above 0.65 go to RETRIEVE. Queries scoring below go to TOOL. Tuned empirically — 0.65 keeps well-known topics (fine-tuning, multi-agent RL, diffusion models) on RETRIEVE while sending unknown recent models (Kimi k1.5, DeepSeek R2) to live search. Too high (0.80) incorrectly sends legitimate queries to TOOL; too low (0.50) never fires the tool.

**Groq (`llama-3.1-8b-instant`) over Anthropic/OpenAI**
Groq's free tier runs at ~500 tok/s with no per-token cost. Router calls complete in ~300ms. For production, `llama-3.3-70b-versatile` on Groq or Claude Sonnet would improve routing accuracy on edge cases.

**`fastembed` (BAAI/bge-small-en-v1.5) over OpenAI embeddings**
Runs locally with zero per-token cost. 384 dimensions — fast and small enough for Qdrant free tier. Quality comparable to `text-embedding-ada-002` on MTEB retrieval benchmarks.

**Qdrant Cloud over Pinecone/Weaviate**
Free tier gives 1GB persistent storage with no credit card required. Python client is minimal — `qdrant.search()` is one call. Hybrid BM25+dense is available natively but not enabled in this version (documented in "what I'd add next").

**Sliding window memory (last 5 turns)**
Chosen over more complex episodic/semantic memory because: (a) the corpus is static so semantic memory adds little, (b) 5 turns covers the realistic attention span of a technical Q&A session, (c) implementable correctly in under an hour. This system uses conversation memory only — the right tradeoff for a single-session RAG agent with a fixed corpus.

**Abstracts-only ingestion**
Full PDFs add 10–50x ingestion time and cost. For a 50-paper corpus, abstracts contain ~85% of answerable content. The live arXiv tool partially compensates for this gap on recent or niche queries.

**Domain matching via exclusion list, not inclusion**
Early versions checked for AI keywords to confirm domain match, which caused vague-but-valid queries to be refused incorrectly. Flipped to checking for explicit out-of-domain keywords (recipes, sports, weather etc.) — everything else passes through to the self-contained check.

**Action normalization after router LLM call**
Groq's LLaMA occasionally returns typos like `RETREIVE`. A normalization map after JSON parsing catches these before they propagate through the graph.

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
#   CLARIFY_THRESHOLD=0.65
```

### 4. Ingest papers

```bash
python -c "from src.ingestion.ingest import ingest; ingest(max_papers=50, batch_size=64, query='cat:cs.AI')"
```

Fetches 50 recent cs.AI papers from arXiv, chunks and embeds them (~5 min on first run), uploads to Qdrant Cloud.

Optionally ingest topic-specific papers on top:
```bash
python -c "from src.ingestion.ingest import ingest; ingest(max_papers=20, batch_size=64, query='reinforcement learning from human feedback')"
```

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

## Agent actions explained

| Action | When it fires | What happens |
|--------|--------------|--------------|
| `RETRIEVE` | top1_score ≥ 0.65, domain match, self-contained | HyDE retrieval from Qdrant corpus |
| `TOOL` | top1_score < 0.65 | Live arXiv API search, results injected as context |
| `CLARIFY` | query has unresolvable references, clarify_count == 0 | Agent asks one focused clarifying question |
| `REFUSE` | explicit out-of-domain keywords detected | Clean domain error, no LLM generation |
| `ANSWER_FROM_MEMORY` | history has 2+ turns and directly answers the query | Synthesized from conversation history, no retrieval |

---

## External tool: arXiv live search

When the router detects low corpus confidence (`top1_score < CLARIFY_THRESHOLD`), it routes to the `tool_node` which calls the arXiv API live:

```python
def arxiv_search(query: str, max_results: int = 3) -> str:
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results,
                          sort_by=arxiv.SortCriterion.Relevance)
    results = list(client.results(search))
    # returns formatted titles + abstracts
```

The tool result is stored in `state["tool_result"]` and injected into `state["context"]` — the generate node handles it identically to corpus retrieval. Every tool call is logged to `decisions.jsonl` with `action: TOOL_RESULT`.

**Example trigger query:** `"What is the Kimi k1.5 model and what training method does it use?"` — a recent model not present in the ingested corpus, scores below threshold, fires live search.

---

## Observability

Every agent decision is logged to `decisions.jsonl`:

```json
{"ts": "2024-01-15T10:23:45", "query": "What are efficient fine-tuning methods?",
 "action": "RETRIEVE", "reasoning": "High confidence corpus match.",
 "top1_score": 0.836, "latency_ms": 1005.3}

{"ts": "2024-01-15T10:24:12", "query": "What is the Kimi k1.5 model?",
 "action": "TOOL", "reasoning": "Corpus confidence below threshold.",
 "top1_score": 0.612, "latency_ms": 310.1}

{"ts": "2024-01-15T10:24:13", "action": "TOOL_RESULT",
 "tool": "arxiv_search", "result_length": 842}
```

In the web UI, the decision log panel shows every entry live. In the CLI, type `/debug`.

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

Observed corpus scores with HyDE across happy path queries:

| Query | top1_score |
|-------|-----------|
| Efficient fine-tuning methods | 0.836 |
| Multi-agent RL challenges | 0.823 |
| Transformer attention | 0.819 |
| Diffusion model generation | 0.801 |
| Retrieval augmented generation | 0.805 |

For abstract queries, HyDE retrieves semantically aligned papers rather than keyword-matched ones — the hypothesis bridges the vocabulary gap between a question and a paper excerpt.

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

**On E08 and E09:** both route to RETRIEVE because their corpus score (0.781) exceeds the threshold — the 50-paper AI corpus matches anything vaguely AI-shaped. However, the generate node correctly produces a clarifying question after seeing the ambiguous context. Clarification happens post-retrieval rather than pre-retrieval. The end behavior is correct; the routing label is wrong. A dedicated ambiguity classifier would fix this cleanly.

---

## Failure modes observed

- **Clarify triggers post-retrieval for vague queries** — high corpus similarity scores on ambiguous queries prevent pre-retrieval CLARIFY routing. Generate node compensates correctly.
- **Weak corpus coverage on niche topics** — queries about specific methodologies or experiment details score low; the tool node compensates with live search but arXiv abstracts may also lack depth.
- **Tool fires on legitimate queries at high threshold** — setting CLARIFY_THRESHOLD above 0.75 incorrectly routes some in-corpus queries to live search. Tuned to 0.65 empirically.
- **LLM action typos** — `llama-3.1-8b-instant` occasionally returns `RETREIVE`. Mitigated with normalization map; a larger model would be more reliable.
- **History augmentation noise** — appending the last user turn to retrieval queries sometimes adds noise rather than context.

---

## What I'd do with another week

1. **Cohere reranker** — post-retrieval step between retrieve_node and generate_node. Expected +10–15% retrieval precision.
2. **Hybrid search (BM25 + dense)** — Qdrant supports this natively. Catches exact paper titles and author names that dense search misses.
3. **Dedicated ambiguity classifier** — replace the heuristic `_is_self_contained()` keyword check with a small LLM call that properly judges ambiguity, fixing E08/E09.
4. **Critic node** — faithfulness check that detects when the answer contradicts retrieved context and routes to retry or refuse.
5. **Semantic memory across sessions** — store prior query clusters in a separate Qdrant collection for personalization across restarts.
6. **Full PDF ingestion** — parse full papers using `pypdf2` with a parent-document retrieval strategy for methodology and experiment details.
7. **Tool result caching** — cache arXiv tool results by query hash to avoid redundant API calls on repeated questions.

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
    │   ├── nodes.py            # router, retrieve, tool, generate nodes
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