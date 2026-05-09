from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
QDRANT_URL: str = os.environ["QDRANT_URL"]
QDRANT_API_KEY: str = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "arxiv_papers")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
CLARIFY_THRESHOLD: float = float(os.getenv("CLARIFY_THRESHOLD", "0.50"))
RETRIEVAL_CONFIDENCE_THRESHOLD: float = float(os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.65"))
