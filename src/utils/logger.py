import json
import time
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("decisions.jsonl")


def log_decision(
    query: str,
    action: str,
    reasoning: str,
    top1_score: float | None = None,
    latency_ms: float | None = None,
    extra: dict | None = None,
) -> None:
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "query": query,
        "action": action,
        "reasoning": reasoning,
        "top1_score": top1_score,
        "latency_ms": round(latency_ms, 1) if latency_ms else None,
        **(extra or {}),
    }
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]
