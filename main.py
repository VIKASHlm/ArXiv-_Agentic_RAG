"""
Entry point — conversational CLI

Usage:
    python main.py

Commands during chat:
    /debug   — show last 5 decisions from decisions.jsonl
    /clear   — clear conversation history
    /quit    — exit
"""

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.agent.graph import get_graph
from src.agent.state import AgentState
from src.utils.logger import read_log

console = Console()


def print_welcome():
    console.print(Panel(
        "[bold]ArXiv AI Research Assistant[/bold]\n"
        "[dim]Powered by LangGraph + HyDE retrieval over cs.AI papers[/dim]\n\n"
        "Commands: [bold]/debug[/bold] | [bold]/clear[/bold] | [bold]/quit[/bold]",
        border_style="dim",
    ))


def print_debug():
    entries = read_log()[-5:]
    if not entries:
        console.print("[dim]No decisions logged yet.[/dim]")
        return
    for e in entries:
        console.print(
            f"[dim]{e['ts'][:19]}[/dim] "
            f"[bold]{e['action']}[/bold] "
            f"score={e.get('top1_score', '?'):.3f} "
            f"latency={e.get('latency_ms', '?'):.0f}ms\n"
            f"  query: {e['query'][:80]}\n"
            f"  reason: {e['reasoning']}"
        )


def chat():
    print_welcome()
    graph = get_graph()
    history: list[dict] = []
    clarify_count = 0

    while True:
        try:
            query = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye![/dim]")
            break

        if not query:
            continue
        if query == "/quit":
            console.print("[dim]Bye![/dim]")
            break
        if query == "/debug":
            print_debug()
            continue
        if query == "/clear":
            history = []
            clarify_count = 0
            console.print("[dim]History cleared.[/dim]")
            continue

        state: AgentState = {
            "query": query,
            "history": history,
            "action": "RETRIEVE",
            "reasoning": "",
            "top1_score": 0.0,
            "chunks": [],
            "hypothesis": "",
            "context": "",
            "answer": "",
            "clarify_count": clarify_count,
            "latency_ms": 0.0,
        }

        with console.status("[dim]Thinking...[/dim]"):
            result = graph.invoke(state)

        action = result.get("action", "")
        answer = result.get("answer", "")
        score = result.get("top1_score", 0.0)
        latency = result.get("latency_ms", 0.0)

        # Print answer
        console.print(f"\n[bold green]Assistant[/bold green] [dim]({action} | score={score:.2f} | {latency:.0f}ms)[/dim]")
        console.print(answer)

        # Update history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        # Track clarification
        if action == "CLARIFY":
            clarify_count += 1
        else:
            clarify_count = 0


if __name__ == "__main__":
    chat()
