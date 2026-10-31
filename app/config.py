import os
from dotenv import load_dotenv
load_dotenv()
E = os.getenv
DATA_DIR = E("DATA_DIR", "data")
EMBED_MODEL = E("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = E("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHUNK_SIZE = int(E("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(E("CHUNK_OVERLAP", 100))
TOP_K = int(E("TOP_K", 5))
MIN_SCORE = float(E("MIN_SCORE", 0.25))  # below this top-hit similarity -> "not found"
LLM_BASE_URL = E("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = E("LLM_API_KEY", "")
LLM_MODEL = E("LLM_MODEL", "llama-3.1-8b-instant")
PRICE_IN = float(E("PRICE_IN_PER_1M", 0.05))
PRICE_OUT = float(E("PRICE_OUT_PER_1M", 0.08))
API_KEY = E("APP_API_KEY", "dev-key")
