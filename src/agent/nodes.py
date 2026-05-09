"""
LangGraph nodes: router → retrieve → generate

Router: one structured-output LLM call deciding RETRIEVE / CLARIFY / REFUSE / ANSWER_FROM_MEMORY
Retrieve: HyDE retrieval from Qdrant
Generate: answer synthesis with hedging on low confidence
"""

import json
import time

from groq import Groq

from src.agent.state import AgentState
from src.retrieval.retriever import format_context, retrieve
from src.utils.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    RETRIEVAL_CONFIDENCE_THRESHOLD,
    CLARIFY_THRESHOLD,
)
from src.utils.logger import log_decision

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compress_history(history: list[dict], max_turns: int = 5) -> str:
    """Take last N turns and format as a string for the router prompt."""
    recent = history[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = msg["role"].upper()
        content = msg["content"][:300]
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _is_self_contained(query: str) -> bool:
    """Heuristic: does the query contain unresolvable references?"""
    vague = ["that paper", "that method", "this approach", "they found", "the authors",
             "it shows", "as mentioned", "the same", "above", "previous"]
    q = query.lower()
    return not any(v in q for v in vague)


def _domain_match(query: str) -> bool:
    """Heuristic: is the query clearly outside the AI/ML research domain?"""
    # Only refuse if it contains explicit out-of-domain signals
    out_of_domain = [
        "recipe", "cook", "food", "restaurant", "movie", "sport", "weather",
        "stock", "price", "celebrity", "song", "music", "game", "travel",
        "hotel", "flight", "shopping", "fashion", "fitness", "workout",
    ]
    q = query.lower()
    # If clearly out of domain, refuse
    if any(k in q for k in out_of_domain):
        return False
    # Short/vague queries pass through — let self-contained check handle them
    return True


# ── Node: Router ──────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """\
You are a routing agent for an AI research paper QA system.
Given a user query and context signals, decide what action to take.

Output ONLY valid JSON with exactly these keys:
  "action": one of RETRIEVE | CLARIFY | REFUSE | ANSWER_FROM_MEMORY
  "reasoning": one sentence explaining your choice

Rules (apply in order):
1. If domain_match is false → REFUSE
2. ANSWER_FROM_MEMORY only if history contains 2+ turns AND the exact answer is already stated in the history. If history is empty or has fewer than 2 turns, never use ANSWER_FROM_MEMORY.
3. If is_self_contained is false AND clarify_count == 0 → CLARIFY
4. If is_self_contained is false AND clarify_count >= 1 → RETRIEVE
5. If top1_score < clarify_threshold AND clarify_count == 0 → CLARIFY
6. Otherwise → RETRIEVE
"""
def router_node(state: AgentState) -> AgentState:
    t0 = time.time()
    query = state["query"]
    history = state.get("history", [])
    clarify_count = state.get("clarify_count", 0)

    # Cheap pre-LLM signals
    from src.retrieval.retriever import get_embedder, get_qdrant
    from src.utils.config import COLLECTION_NAME

    embedder = get_embedder()
    vec = next(embedder.embed([query]))
    qdrant = get_qdrant()
    probe = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec.tolist(),
        limit=1,
        with_payload=False,
    )
    top1_score = probe[0].score if probe else 0.0
    is_self_contained = _is_self_contained(query)
    domain_match = _domain_match(query)

    signals = {
        "top1_score": round(top1_score, 3),
        "is_self_contained": is_self_contained,
        "domain_match": domain_match,
        "clarify_count": clarify_count,
        "clarify_threshold": CLARIFY_THRESHOLD,
    }
    history_str = _compress_history(history)

    user_msg = (
        f"Query: {query}\n\n"
        f"Signals: {json.dumps(signals)}\n\n"
        f"Recent conversation:\n{history_str}"
    )

    client = get_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=200,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if Groq wraps JSON in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        action = parsed["action"].strip().upper()
        reasoning = parsed["reasoning"]
    except Exception:
        action = "RETRIEVE"
        reasoning = f"JSON parse failed, defaulting to RETRIEVE. Raw: {raw[:100]}"

    # Normalize LLM typos and variants
    action_map = {
        "RETREIVE": "RETRIEVE",
        "RETREIVAL": "RETRIEVE",
        "RETRIVAL": "RETRIEVE",
        "CLARIFICATION": "CLARIFY",
        "ANSWER": "ANSWER_FROM_MEMORY",
        "ANSWER_FROM_HISTORY": "ANSWER_FROM_MEMORY",
    }
    action = action_map.get(action, action)
    if action not in {"RETRIEVE", "CLARIFY", "REFUSE", "ANSWER_FROM_MEMORY"}:
        reasoning = f"Unknown action '{action}' normalized to RETRIEVE."
        action = "RETRIEVE"

    latency_ms = (time.time() - t0) * 1000
    log_decision(query, action, reasoning, top1_score, latency_ms)

    return {
        **state,
        "action": action,
        "reasoning": reasoning,
        "top1_score": top1_score,
        "latency_ms": latency_ms,
    }


# ── Node: Retrieve ────────────────────────────────────────────────────────────

def retrieve_node(state: AgentState) -> AgentState:
    query = state["query"]
    history = state.get("history", [])

    if history:
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        augmented_query = f"{last_user} {query}".strip() if last_user != query else query
    else:
        augmented_query = query

    chunks, top1_score, hypothesis = retrieve(augmented_query)
    context = format_context(chunks)

    return {
        **state,
        "chunks": chunks,
        "hypothesis": hypothesis,
        "context": context,
        "top1_score": top1_score,
    }


# ── Node: Generate ────────────────────────────────────────────────────────────

GENERATE_SYSTEM = """\
You are an expert AI research assistant. Answer questions about AI/ML research papers.

Guidelines:
- Answer based only on the provided context.
- If the context is weak (score below 0.65), say "Based on limited matches in the corpus..." before answering.
- If the context contains contradictions, flag them explicitly.
- If you truly cannot answer from the context, say so clearly — do not hallucinate.
- Keep answers concise: 3-5 sentences unless the question requires more detail.
- Cite paper titles when relevant.
"""


def generate_node(state: AgentState) -> AgentState:
    query = state["query"]
    context = state.get("context", "")
    history = state.get("history", [])
    top1_score = state.get("top1_score", 0.0)
    action = state.get("action", "RETRIEVE")

    client = get_client()

    if action == "CLARIFY":
        clarify_prompt = (
            f"The user asked: '{query}'\n"
            "Ask one focused clarifying question to resolve the ambiguity. "
            "Be friendly and specific about what information you need."
        )
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=150,
            messages=[{"role": "user", "content": clarify_prompt}],
        )
        answer = resp.choices[0].message.content.strip()
        return {**state, "answer": answer}

    if action == "REFUSE":
        answer = (
            "I can only answer questions about AI/ML research papers in my corpus. "
            "Your question appears to be outside that domain. "
            "Try asking about topics like neural networks, transformers, reinforcement learning, "
            "computer vision, or NLP research."
        )
        return {**state, "answer": answer}

    if action == "ANSWER_FROM_MEMORY":
        history_str = _compress_history(history, max_turns=6)
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": f"Based on our conversation:\n{history_str}\n\nAnswer: {query}",
                }
            ],
        )
        answer = resp.choices[0].message.content.strip()
        return {**state, "answer": answer}

    # Standard RETRIEVE path
    confidence_note = ""
    if top1_score < RETRIEVAL_CONFIDENCE_THRESHOLD:
        confidence_note = (
            f"Note: retrieval confidence is low ({top1_score:.2f}). "
            "The answer may be incomplete.\n\n"
        )

    messages = [
        {"role": "system", "content": GENERATE_SYSTEM},
        *[{"role": m["role"], "content": m["content"]} for m in history[-6:]],
        {
            "role": "user",
            "content": (
                f"Context from corpus:\n{context}\n\n"
                f"{confidence_note}"
                f"Question: {query}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=500,
        messages=messages,
    )
    answer = resp.choices[0].message.content.strip()
    return {**state, "answer": answer}
