"""Parameters owned by model inference."""

HUGGING_FACE_BASE_URL = "https://router.huggingface.co"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# The judgment schema is small. This bounds billed generation and prevents a
# misbehaving model from producing an unbounded answer; it does not limit input.
MAX_COMPLETION_TOKENS = 512

# Temperature zero makes repeated classification runs as deterministic as the
# provider permits.
TEMPERATURE = 0

# Only failures that are normally transient receive these two short retries.
TRANSIENT_RETRY_DELAYS_SECONDS = (1, 2)
TRANSIENT_HTTP_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# Direct tests showed that this exact Hugging Face response clears after about
# one minute. It receives one cooldown retry; other HTTP 402 responses do not.
HUGGING_FACE_COOLDOWN_SECONDS = 60
HUGGING_FACE_LIMIT_MESSAGE = "You have depleted your monthly included credits."
