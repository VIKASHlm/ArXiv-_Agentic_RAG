"""
LangGraph graph definition.

Flow:
    router_node
        ├── RETRIEVE           → retrieve_node → generate_node → END
        ├── TOOL               → tool_node     → generate_node → END
        ├── CLARIFY            → generate_node → END
        ├── REFUSE             → generate_node → END
        └── ANSWER_FROM_MEMORY → generate_node → END
"""

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import router_node, retrieve_node, tool_node, generate_node


def route_after_router(state: AgentState) -> str:
    action = state.get("action", "RETRIEVE")
    if action == "RETRIEVE":
        return "retrieve"
    if action == "TOOL":
        return "tool"
    return "generate"  # CLARIFY, REFUSE, ANSWER_FROM_MEMORY handled in generate_node


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("tool", tool_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retrieve",
            "tool": "tool",
            "generate": "generate",
        },
    )

    graph.add_edge("retrieve", "generate")
    graph.add_edge("tool", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph