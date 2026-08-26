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
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# ─── Groq Model Catalog ─────────────────────────────────────────────
# Available text-generation models on Groq (as of 2026).
# The first entry is the default if GROQ_MODEL is unset.
GROQ_MODELS = [
    {"id": "qwen/qwen3.8-27b",        "name": "Qwen 3.8 27B",        "context": "256k", "notes": "Default — strong, fast"},
    {"id": "qwen/qwen3.6-27b",        "name": "Qwen 3.6 27B",        "context": "256k", "notes": "Previous-gen Qwen"},
    {"id": "openai/gpt-oss-120b",     "name": "GPT-OSS 120B",        "context": "128k", "notes": "Largest available — best coherence"},
    {"id": "openai/gpt-oss-20b",      "name": "GPT-OSS 20B",         "context": "128k", "notes": "Lighter OpenAI model"},
    {"id": "allam-2-7b",              "name": "Allam 2 7B",           "context": "4k",   "notes": "Small — fast but limited"},
]
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
USE_CLOUD_MODEL = os.getenv("USE_CLOUD_MODEL", "false").lower() in ("1", "true", "yes", "on")

# Free-tier Groq caps tokens/minute at ~6000. Adaptive TPM pacing keeps a
# rolling 60s token budget (estimated prompt + max output) under this ceiling
# so we get throttled by pacing instead of 429 storms.
GROQ_TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", "6000"))
GROQ_TPM_WINDOW_SECONDS = float(os.getenv("GROQ_TPM_WINDOW_SECONDS", "60"))
GROQ_TPM_SAFETY_MARGIN = float(os.getenv("GROQ_TPM_SAFETY_MARGIN", "0.15"))
# Reserve for calls that don't pass max_tokens (avoids reserving full 4096).
GROQ_TPM_RESERVE_DEFAULT = int(os.getenv("GROQ_TPM_RESERVE_DEFAULT", "1200"))
GROQ_TPM_PACING = os.getenv("GROQ_TPM_PACING", "true").lower() in ("1", "true", "yes", "on")
# Preemptively pick the key with the most remaining TPM budget per window,
# instead of only rotating after a 429.
GROQ_PROACTIVE_ROTATION = os.getenv("GROQ_PROACTIVE_ROTATION", "true").lower() in (
    "1", "true", "yes", "on",
)
# Route Planner/Critic/Editor/Verifier to a local GGUF and reserve Groq for the
# Writer. Only activates when a local model is present and loadable.
HYBRID_ROUTING = os.getenv("HYBRID_ROUTING", "true").lower() in ("1", "true", "yes", "on")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_TPM_LIMIT = int(os.getenv("GEMINI_TPM_LIMIT", "200000"))

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
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
# Keep story generation usable when a machine is offline or a Hugging Face
# model was not downloaded yet. Set this to false to permit first-run downloads.
EMBEDDING_LOCAL_FILES_ONLY = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "true").lower() in (
    "1", "true", "yes", "on",
)
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

# Post-generation verifier (last-line-of-defense drift guard).
VERIFIER_ENABLED = os.getenv("VERIFIER_ENABLED", "true").lower() in ("1", "true", "yes", "on")
# Maximum guided-rewrite attempts when the verifier blocks a scene.
VERIFIER_MAX_RETRIES = int(os.getenv("VERIFIER_MAX_RETRIES", "2"))
# After this many repeated (same-signature) verifier failures, escalate to review.
VERIFIER_REVIEW_AFTER = int(os.getenv("VERIFIER_REVIEW_AFTER", "3"))
# If true, a scene that cannot pass verification hard-stops the chapter instead
# of continuing with a needs_review flag.
VERIFIER_HARD_STOP = os.getenv("VERIFIER_HARD_STOP", "false").lower() in ("1", "true", "yes", "on")
# Run the LLM refinement pass only when a cloud backend is active (cheap local
# models can produce noisy verifier output). Override with VERIFIER_LLM_ALWAYS=1.
VERIFIER_LLM_ALWAYS = os.getenv("VERIFIER_LLM_ALWAYS", "false").lower() in ("1", "true", "yes", "on")

# How many premise steps the planner is allowed to cover per chapter.
# With 100-step premises, 1 step/chapter = 100 chapters (too slow).
# Set to 3-5 to let the story move forward at a comfortable pace.
PREMISE_STEPS_PER_CHAPTER = int(os.getenv("PREMISE_STEPS_PER_CHAPTER", "3"))

# ─── Quality Gate & Best-of-N ────────────────────────────────────────
# Deterministic post-editor quality gate (QualityController.score). When a
# finished scene scores below the threshold, the pipeline requests ONE guided
# rewrite with the scorer's issues, then accepts the best effort.
QUALITY_GATE_ENABLED = os.getenv("QUALITY_GATE_ENABLED", "true").lower() in (
    "1", "true", "yes", "on",
)
QUALITY_GATE_THRESHOLD = float(os.getenv("QUALITY_GATE_THRESHOLD", "0.55"))
# Candidate drafts per scene. Key scenes = first scene of a chapter, scenes
# typed "peak", and the final scene. Candidates are generated sequentially so
# Groq TPM pacing stays intact; each candidate is scored deterministically.
# Default is 1 (single draft) — each extra candidate is a full LLM scene
# generation, so raise this only when you want higher quality per scene and
# are willing to spend extra tokens.
BEST_OF_N_KEY_SCENES = int(os.getenv("BEST_OF_N_KEY_SCENES", "1"))
BEST_OF_N_NORMAL_SCENES = int(os.getenv("BEST_OF_N_NORMAL_SCENES", "1"))

# Cloud backends have large context windows — retrieve more story memory for
# them instead of squeezing continuity to a couple of chunks.
CLOUD_TOP_K_RETRIEVAL = int(os.getenv("CLOUD_TOP_K_RETRIEVAL", "4"))
CLOUD_CONTEXT_CHARS = int(os.getenv("CLOUD_CONTEXT_CHARS", "6000"))

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

# ─── Vision Module Configuration ─────────────────────────────────────
VISION_BACKEND = os.getenv("VISION_BACKEND", "local").lower()
OLLAMA_VISION_URL = os.getenv("OLLAMA_VISION_URL", "http://localhost:11434")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "moondream")

