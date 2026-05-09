"""
Evaluation harness — 10 questions covering happy path, clarify, and refuse cases.

Scoring is manual pass/fail with an expected_behavior description.
Run: python -m src.eval.evals

Output: eval_results.json + printed summary table
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.agent.graph import get_graph
from src.agent.state import AgentState

console = Console()


@dataclass
class EvalCase:
    id: str
    query: str
    expected_action: str        # RETRIEVE | CLARIFY | REFUSE
    expected_behavior: str      # human-readable description for manual scoring
    history: list[dict] = field(default_factory=list)


# ── 10 eval questions ─────────────────────────────────────────────────────────

EVAL_CASES: list[EvalCase] = [
    # Happy path — should retrieve and answer
    EvalCase(
        id="E01",
        query="What are the main differences between transformer and RNN architectures?",
        expected_action="RETRIEVE",
        expected_behavior="Should retrieve relevant papers and explain key differences: attention mechanism, parallelism, long-range dependencies.",
    ),
    EvalCase(
        id="E02",
        query="How does chain-of-thought prompting improve LLM reasoning?",
        expected_action="RETRIEVE",
        expected_behavior="Should retrieve papers on CoT prompting and explain step-by-step reasoning benefits.",
    ),
    EvalCase(
        id="E03",
        query="What is retrieval-augmented generation and why is it useful?",
        expected_action="RETRIEVE",
        expected_behavior="Should retrieve RAG-related papers and explain grounding, reduced hallucination, knowledge updating.",
    ),
    EvalCase(
        id="E04",
        query="Summarize recent progress in reinforcement learning from human feedback.",
        expected_action="RETRIEVE",
        expected_behavior="Should retrieve RLHF-related papers and summarize key methods like PPO, reward modeling.",
    ),
    EvalCase(
        id="E05",
        query="What techniques are used for efficient fine-tuning of large language models?",
        expected_action="RETRIEVE",
        expected_behavior="Should retrieve papers on LoRA, adapter methods, prefix tuning and summarize them.",
    ),
    EvalCase(
        id="E06",
        query="How do diffusion models generate images?",
        expected_action="RETRIEVE",
        expected_behavior="Should retrieve diffusion model papers and explain the forward/reverse diffusion process.",
    ),
    EvalCase(
        id="E07",
        query="What evaluation metrics are commonly used for machine translation?",
        expected_action="RETRIEVE",
        expected_behavior="Should retrieve NLP papers and mention BLEU, METEOR, chrF, BERTScore.",
    ),

    # Clarify cases — ambiguous queries that should trigger a question
    EvalCase(
        id="E08",
        query="What did they find about the attention mechanism?",
        expected_action="CLARIFY",
        expected_behavior="Should ask for clarification — 'they' is unresolvable. Should ask which paper or authors.",
        history=[],
    ),
    EvalCase(
        id="E09",
        query="Tell me more about that approach.",
        expected_action="CLARIFY",
        expected_behavior="Should ask which approach — 'that approach' has no referent in an empty history.",
        history=[],
    ),

    # Refuse cases — out of domain
    EvalCase(
        id="E10",
        query="What is the best recipe for chocolate chip cookies?",
        expected_action="REFUSE",
        expected_behavior="Should refuse — completely outside the AI/ML research domain.",
    ),
]


# ── Runner ────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    id: str
    query: str
    expected_action: str
    actual_action: str
    expected_behavior: str
    answer: str
    top1_score: float
    latency_ms: float
    action_correct: bool


def run_eval_case(case: EvalCase, graph) -> EvalResult:
    initial_state: AgentState = {
        "query": case.query,
        "history": case.history,
        "action": "RETRIEVE",
        "reasoning": "",
        "top1_score": 0.0,
        "chunks": [],
        "hypothesis": "",
        "context": "",
        "answer": "",
        "clarify_count": 0,
        "latency_ms": 0.0,
    }

    t0 = time.time()
    result = graph.invoke(initial_state)
    latency_ms = (time.time() - t0) * 1000

    return EvalResult(
        id=case.id,
        query=case.query,
        expected_action=case.expected_action,
        actual_action=result.get("action", "UNKNOWN"),
        expected_behavior=case.expected_behavior,
        answer=result.get("answer", ""),
        top1_score=result.get("top1_score", 0.0),
        latency_ms=latency_ms,
        action_correct=result.get("action", "") == case.expected_action,
    )


def run_evals(output_path: str = "eval_results.json") -> list[EvalResult]:
    graph = get_graph()
    results: list[EvalResult] = []

    console.print("\n[bold]Running evaluation harness...[/bold]\n")

    for case in EVAL_CASES:
        console.print(f"  [{case.id}] {case.query[:60]}...")
        result = run_eval_case(case, graph)
        results.append(result)
        status = "[green]PASS[/green]" if result.action_correct else "[red]FAIL[/red]"
        console.print(f"        Action: {result.actual_action} (expected {result.expected_action}) {status}")
        console.print(f"        Score: {result.top1_score:.3f} | Latency: {result.latency_ms:.0f}ms")

    # Save results
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    console.print(f"\n[bold]Results saved to {output_path}[/bold]")

    # Print summary table
    table = Table(title="Eval Summary")
    table.add_column("ID")
    table.add_column("Expected")
    table.add_column("Actual")
    table.add_column("Score")
    table.add_column("Pass?")

    correct = 0
    for r in results:
        passed = r.action_correct
        correct += int(passed)
        table.add_row(
            r.id,
            r.expected_action,
            r.actual_action,
            f"{r.top1_score:.3f}",
            "[green]YES[/green]" if passed else "[red]NO[/red]",
        )

    console.print(table)
    console.print(f"\n[bold]Action accuracy: {correct}/{len(results)} ({100*correct//len(results)}%)[/bold]")
    console.print("\n[dim]Note: action accuracy measures routing only.")
    console.print("Answer quality requires manual review of eval_results.json[/dim]")

    return results


if __name__ == "__main__":
    run_evals()
