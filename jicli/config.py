"""Configuration management for Just Intelligent CLI."""

import os
import json
from pathlib import Path

# ── Defaults ─────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_CONTEXT_WINDOW = 16384
DEFAULT_MAX_OUTPUT_TOKENS = -1
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TURNS = 50

MODEL_ALIASES = {
    "ji": "qwen3.5:4b-q4_K_M",
    "cascade": "nemotron-cascade-2:30b-a3b-q4_K_M",
    "30b": "nemotron-cascade-2:30b-a3b-q4_K_M",
    "qwen3.5": "qwen3.5:9b",
}

OLLAMA_OPTIONS = {
    "num_predict": DEFAULT_MAX_OUTPUT_TOKENS,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.3,
    "temperature": DEFAULT_TEMPERATURE,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.5,
    "num_ctx": DEFAULT_CONTEXT_WINDOW,
}

# Characters per token estimate (blended for code+English)
CHARS_PER_TOKEN = 3.0

# Max tool output length before truncation
MAX_TOOL_OUTPUT = 10000

# Dangerous command patterns for bash safety
DANGEROUS_PATTERNS = [
    "rm -rf /", "rm -rf /*", "mkfs", "dd if=", ":(){:|:&};:",
    "chmod -R 777 /", "mv /* ", "> /dev/sda",
]


def resolve_model(name: str) -> str:
    """Resolve a model alias to its full name."""
    return MODEL_ALIASES.get(name, name)


def resolve_host(host: str = None) -> str:
    """Resolve the Ollama host: explicit > env > default."""
    return (host or os.getenv("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    if not text:
        return 0
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def get_data_dir() -> Path:
    """Get the Just Intelligent CLI data directory."""
    data_dir = Path(os.getenv("JICLI_DATA_DIR", Path.home() / ".jicli"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_config(config_path: str = None) -> dict:
    """Load config from file, falling back to defaults."""
    config = {
        "model": DEFAULT_MODEL,
        "host": DEFAULT_HOST,
        "context_window": DEFAULT_CONTEXT_WINDOW,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "max_turns": DEFAULT_MAX_TURNS,
        "ollama_options": OLLAMA_OPTIONS.copy(),
    }

    path = config_path or os.getenv("JICLI_CONFIG")
    if path and os.path.exists(path):
        with open(path, "r") as f:
            user_config = json.load(f)
        config.update(user_config)

    return config
