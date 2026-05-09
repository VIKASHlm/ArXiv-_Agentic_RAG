"""
Simple FastAPI backend for the RAG agent.

Run: uvicorn ui_server:app --reload --port 8000
Then open: http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

from src.agent.graph import get_graph
from src.agent.state import AgentState
from src.utils.logger import read_log

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (single user, good enough for demo)
session: dict = {
    "history": [],
    "clarify_count": 0,
}


class QueryRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
def index():
    with open("ui.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
def chat(req: QueryRequest):
    graph = get_graph()

    state: AgentState = {
        "query": req.query,
        "history": session["history"],
        "action": "RETRIEVE",
        "reasoning": "",
        "top1_score": 0.0,
        "chunks": [],
        "hypothesis": "",
        "context": "",
        "answer": "",
        "clarify_count": session["clarify_count"],
        "latency_ms": 0.0,
    }

    result = graph.invoke(state)

    action = result.get("action", "RETRIEVE")
    answer = result.get("answer", "")
    score = result.get("top1_score", 0.0)
    latency = result.get("latency_ms", 0.0)
    reasoning = result.get("reasoning", "")

    # Update session
    session["history"].append({"role": "user", "content": req.query})
    session["history"].append({"role": "assistant", "content": answer})
    if action == "CLARIFY":
        session["clarify_count"] += 1
    else:
        session["clarify_count"] = 0

    # Keep history bounded
    if len(session["history"]) > 20:
        session["history"] = session["history"][-20:]

    return {
        "answer": answer,
        "action": action,
        "score": round(score, 3),
        "latency_ms": round(latency, 0),
        "reasoning": reasoning,
    }


@app.post("/reset")
def reset():
    session["history"] = []
    session["clarify_count"] = 0
    return {"status": "ok"}


@app.get("/debug")
def debug():
    return {"logs": read_log()[-10:]}