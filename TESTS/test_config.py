"""Tests for jicli/config.py — Configuration management."""

import os
import sys
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jicli.config import (
    resolve_model, resolve_host, estimate_tokens, load_config,
    get_data_dir, MODEL_ALIASES, DANGEROUS_PATTERNS, DEFAULT_MODEL,
    DEFAULT_HOST, DEFAULT_CONTEXT_WINDOW, CHARS_PER_TOKEN,
)


def test_resolve_model_alias():
    """Model aliases should resolve to full model names."""
    assert resolve_model("ji") == "qwen3.5:4b-q4_K_M"
    assert resolve_model("cascade") == "nemotron-cascade-2:30b-a3b-q4_K_M"
    assert resolve_model("30b") == "nemotron-cascade-2:30b-a3b-q4_K_M"
    print("  PASS: resolve_model aliases")


def test_resolve_model_passthrough():
    """Unknown names should pass through unchanged."""
    assert resolve_model("some-custom-model:latest") == "some-custom-model:latest"
    assert resolve_model("llama3:70b") == "llama3:70b"
    print("  PASS: resolve_model passthrough")


def test_resolve_host_default():
    """Default host should be localhost:11434."""
    old = os.environ.pop("OLLAMA_HOST", None)
    try:
        assert resolve_host() == DEFAULT_HOST
    finally:
        if old:
            os.environ["OLLAMA_HOST"] = old
    print("  PASS: resolve_host default")


def test_resolve_host_explicit():
    """Explicit host should override everything."""
    assert resolve_host("http://myserver:1234") == "http://myserver:1234"
    print("  PASS: resolve_host explicit")


def test_resolve_host_env():
    """OLLAMA_HOST env var should be used when no explicit host."""
    old = os.environ.get("OLLAMA_HOST")
    os.environ["OLLAMA_HOST"] = "http://envhost:5555"
    try:
        assert resolve_host() == "http://envhost:5555"
    finally:
        if old:
            os.environ["OLLAMA_HOST"] = old
        else:
            del os.environ["OLLAMA_HOST"]
    print("  PASS: resolve_host env")


def test_resolve_host_strips_trailing_slash():
    """Trailing slash should be stripped."""
    assert resolve_host("http://myserver:1234/") == "http://myserver:1234"
    print("  PASS: resolve_host strips trailing slash")


def test_estimate_tokens():
    """Token estimation should be roughly chars / CHARS_PER_TOKEN."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") >= 1
    # 300 chars at ~3 chars/token should be ~100 tokens
    text = "x" * 300
    tokens = estimate_tokens(text)
    assert 80 <= tokens <= 120, f"Expected ~100 tokens, got {tokens}"
    print("  PASS: estimate_tokens")


def test_estimate_tokens_none_safe():
    """Empty/None text should return 0."""
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    print("  PASS: estimate_tokens none safe")


def test_load_config_defaults():
    """Default config should have expected keys."""
    config = load_config()
    assert config["model"] == DEFAULT_MODEL
    assert config["host"] == DEFAULT_HOST
    assert config["context_window"] == DEFAULT_CONTEXT_WINDOW
    assert "ollama_options" in config
    print("  PASS: load_config defaults")


def test_load_config_from_file():
    """Config file should override defaults."""
    custom = {"model": "custom-model:1b", "temperature": 0.5}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(custom, f)
        f.flush()
        config = load_config(f.name)
    os.unlink(f.name)
    assert config["model"] == "custom-model:1b"
    assert config["temperature"] == 0.5
    # Non-overridden defaults should remain
    assert config["host"] == DEFAULT_HOST
    print("  PASS: load_config from file")


def test_dangerous_patterns():
    """DANGEROUS_PATTERNS should contain known dangerous commands."""
    assert "rm -rf /" in DANGEROUS_PATTERNS
    assert "mkfs" in DANGEROUS_PATTERNS
    assert len(DANGEROUS_PATTERNS) >= 5
    print("  PASS: dangerous_patterns")


def test_model_aliases_not_empty():
    """Model aliases should exist."""
    assert len(MODEL_ALIASES) >= 5
    assert "ji" in MODEL_ALIASES
    print("  PASS: model_aliases not empty")


if __name__ == "__main__":
    print("=== Config Tests ===")
    test_resolve_model_alias()
    test_resolve_model_passthrough()
    test_resolve_host_default()
    test_resolve_host_explicit()
    test_resolve_host_env()
    test_resolve_host_strips_trailing_slash()
    test_estimate_tokens()
    test_estimate_tokens_none_safe()
    test_load_config_defaults()
    test_load_config_from_file()
    test_dangerous_patterns()
    test_model_aliases_not_empty()
    print("\nAll config tests passed!")
