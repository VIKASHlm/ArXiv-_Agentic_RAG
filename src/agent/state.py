from typing import TypedDict, Literal


Action = Literal["RETRIEVE", "CLARIFY", "REFUSE", "ANSWER_FROM_MEMORY"]


class AgentState(TypedDict):
    # Input
    query: str
    history: list[dict]          # [{"role": "user"|"assistant", "content": str}]

    # Router outputs
    action: Action
    reasoning: str
    top1_score: float

    # Retrieval outputs
    chunks: list[dict]
    hypothesis: str              # HyDE hypothesis for logging
    context: str                 # formatted context string

    # Final output
    answer: str

    # Loop guards
    clarify_count: int
    latency_ms: float
