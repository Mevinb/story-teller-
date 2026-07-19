"""
Central configuration for the multi-agent story generation pipeline.
All tunable parameters live here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Model Configuration ────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(__file__)
LLAMA_MODELS_DIR = os.path.abspath(
    os.getenv("LLAMA_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
)
LLAMA_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    os.path.join(LLAMA_MODELS_DIR, "mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
)
LLAMA_PROMPT_TEMPLATE = os.getenv("LLAMA_PROMPT_TEMPLATE", "mistral").lower()
LLAMA_CPP_PARAMS = {
    "n_gpu_layers": int(os.getenv("LLAMA_N_GPU_LAYERS", "20")),
    "n_batch": int(os.getenv("LLAMA_N_BATCH", "512")),
    "n_threads": int(os.getenv("LLAMA_N_THREADS", str(max(1, (os.cpu_count() or 8) // 2)))),
    "f16_kv": os.getenv("LLAMA_F16_KV", "true").lower() in ("1", "true", "yes", "on"),
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
if GROQ_API_KEY and GROQ_API_KEY not in GROQ_API_KEYS:
    GROQ_API_KEYS.insert(0, GROQ_API_KEY)

if GROQ_API_KEYS:
    print(f"✅ Loaded {len(GROQ_API_KEYS)} Groq API keys for rotation.")
else:
    print("⚠️ No Groq API keys found in .env!")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
USE_CLOUD_MODEL = os.getenv("USE_CLOUD_MODEL", "false").lower() in ("1", "true", "yes", "on")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_TPM_LIMIT = int(os.getenv("GEMINI_TPM_LIMIT", "200000"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "cognitivecomputations/dolphin-mistral-24b-venice-edition:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MIN_REQUEST_INTERVAL = float(os.getenv("OPENROUTER_MIN_REQUEST_INTERVAL", "1.0"))

# ─── Generation Parameters ──────────────────────────────────────────
LOCAL_MODEL_PARAMS = {
    "num_ctx": int(os.getenv("LOCAL_NUM_CTX", "4096")),
    "temperature": float(os.getenv("LOCAL_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("LOCAL_TOP_P", "0.9")),
    "max_tokens": int(os.getenv("LOCAL_MAX_TOKENS", "1024")),
    "structured_max_tokens": int(os.getenv("LOCAL_STRUCTURED_MAX_TOKENS", "900")),
}

CLOUD_MODEL_PARAMS = {
    "temperature": float(os.getenv("CLOUD_TEMPERATURE", "0.8")),
    "top_p": float(os.getenv("CLOUD_TOP_P", "0.9")),
    "max_tokens": int(os.getenv("CLOUD_MAX_TOKENS", "4096")),
}

# Per-agent determinism/creativity profile
AGENT_TEMPERATURES = {
    "planner": 0.2,      # architect + scene planner
    "writer": 0.7,
    "critic": 0.1,       # consistency engine
    "editor": 0.2,
}

# ─── Scene / Chapter Defaults ───────────────────────────────────────
SCENES_PER_CHAPTER_MIN = 2
SCENES_PER_CHAPTER_MAX = 6
WORDS_PER_SCENE_MIN = 500
WORDS_PER_SCENE_MAX = 1000

# ─── Embedding / Retrieval ──────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 400          # tokens per chunk
CHUNK_OVERLAP = 50        # token overlap between chunks
TOP_K_RETRIEVAL = 4       # number of chunks to retrieve
CONTEXT_TOKEN_BUDGET = 2048  # max tokens for assembled context
# Maximum number of query embeddings to cache in memory (LRU eviction).
# Each entry is ~1.5KB (384-dim float32 vector). 1024 entries ≈ 1.5MB.
EMBEDDING_CACHE_MAX_SIZE = int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "1024"))

# ─── Quality Thresholds ─────────────────────────────────────────────
MIN_SCENE_WORDS = 500
MAX_REPETITION_RATIO = 0.12   # n-gram overlap threshold
MAX_CONSISTENCY_RETRIES = 2
MAX_GENERATION_RETRIES = 3

# ─── Pipeline Control ─────────────────────────────────────────────────
MAX_SCENE_ITERATIONS = 5
MAX_PIPELINE_STEPS = 300
MAX_TOKEN_BUDGET = 24000

# ─── Drift Prevention ───────────────────────────────────────────────
REANCHOR_EVERY_N_CHAPTERS = 3

# How many premise steps the planner is allowed to cover per chapter.
# With 100-step premises, 1 step/chapter = 100 chapters (too slow).
# Set to 3-5 to let the story move forward at a comfortable pace.
PREMISE_STEPS_PER_CHAPTER = int(os.getenv("PREMISE_STEPS_PER_CHAPTER", "3"))

# ─── Retry Configuration ────────────────────────────────────────────
RETRY_BASE_DELAY = 3.0        # seconds
RETRY_MAX_DELAY = 30.0        # seconds
RETRY_BACKOFF_FACTOR = 2.0

# Groq free/dev accounts are usually constrained more by request and token
# windows than raw model speed. Keep these conservative to avoid 429 storms.
GROQ_MIN_REQUEST_INTERVAL = float(os.getenv("GROQ_MIN_REQUEST_INTERVAL", "8.0"))
GROQ_RATE_LIMIT_BUFFER = float(os.getenv("GROQ_RATE_LIMIT_BUFFER", "5.0"))
GROQ_ROTATE_ON_RATE_LIMIT = os.getenv("GROQ_ROTATE_ON_RATE_LIMIT", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
GROQ_CONTINUATION_ATTEMPTS = int(os.getenv("GROQ_CONTINUATION_ATTEMPTS", "1"))

# ─── Storage ─────────────────────────────────────────────────────────
PROJECTS_DIR = os.path.join(PROJECT_ROOT, "projects")

# ─── Web UI ──────────────────────────────────────────────────────────
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() in ("1", "true", "yes", "on")
SSE_QUEUE_MAXSIZE = int(os.getenv("SSE_QUEUE_MAXSIZE", "1000"))
