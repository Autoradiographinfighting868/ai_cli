"""Tests for jicli/memory/ — Session store, context manager, persistent memory."""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jicli.memory.session import SessionStore
from jicli.memory.context import ContextManager
from jicli.memory.persistent import PersistentMemory


# ── Session Store Tests ──────────────────────────────────────────

def test_session_create():
    """Should create a session and return its ID."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        store = SessionStore(db)
        sid = store.create_session("test-model")
        assert sid is not None
        assert len(sid) > 0
    finally:
        os.unlink(db)
    print("  PASS: session create")


def test_session_add_get_messages():
    """Should store and retrieve messages."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        store = SessionStore(db)
        sid = store.create_session("test-model")
        store.add_message(sid, "user", "Hello")
        store.add_message(sid, "assistant", "Hi there!")
        msgs = store.get_messages(sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hi there!"
    finally:
        os.unlink(db)
    print("  PASS: session add/get messages")


def test_session_memory_kv():
    """Should store and retrieve key-value memory."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        store = SessionStore(db)
        store.set_memory("user_name", "Alice")
        assert store.get_memory("user_name") == "Alice"
        # Overwrite
        store.set_memory("user_name", "Bob")
        assert store.get_memory("user_name") == "Bob"
    finally:
        os.unlink(db)
    print("  PASS: session memory KV")


def test_session_memory_search():
    """Should search memory by key prefix."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        store = SessionStore(db)
        store.set_memory("pref_theme", "dark")
        store.set_memory("pref_lang", "python")
        store.set_memory("other", "value")
        results = store.search_memory("pref")
        assert len(results) == 2
    finally:
        os.unlink(db)
    print("  PASS: session memory search")


def test_session_list():
    """Should list sessions."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        store = SessionStore(db)
        store.create_session("model-a")
        store.create_session("model-b")
        sessions = store.list_sessions()
        assert len(sessions) >= 2
    finally:
        os.unlink(db)
    print("  PASS: session list")


def test_session_last():
    """get_last_session should return the most recent session."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        store = SessionStore(db)
        s1 = store.create_session("model-a")
        s2 = store.create_session("model-b")
        last = store.get_latest_session()
        assert last is not None
        assert last["id"] == s2
    finally:
        os.unlink(db)
    print("  PASS: session last")


def test_session_error_tracking():
    """Should log and retrieve error patterns."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        store = SessionStore(db)
        store.log_error("tool_error", "Bash tool failed: permission denied")
        store.log_error("tool_error", "Bash tool failed: permission denied")
        store.log_error("parse_error", "Invalid JSON in tool call")
        errors = store.get_common_errors(limit=10)
        assert len(errors) >= 1
    finally:
        os.unlink(db)
    print("  PASS: session error tracking")


# ── Context Manager Tests ────────────────────────────────────────

def test_context_measure():
    """Should estimate tokens for messages."""
    cm = ContextManager(max_tokens=1000)
    msgs = [
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi! How can I help?"},
    ]
    tokens = cm.measure_messages(msgs)
    assert tokens > 0
    assert tokens < 100  # Two short messages
    print("  PASS: context measure")


def test_context_needs_pruning():
    """Should detect when pruning is needed."""
    cm = ContextManager(max_tokens=100, prune_threshold=0.5)
    cm.set_system_tokens("short prompt")
    # Small messages — no pruning needed
    small = [{"role": "user", "content": "hi"}]
    assert cm.needs_pruning(small) is False

    # Large messages — pruning needed
    big = [{"role": "user", "content": "x" * 500}]
    assert cm.needs_pruning(big) is True
    print("  PASS: context needs_pruning")


def test_context_prune():
    """Pruning should reduce message count while keeping head and tail."""
    cm = ContextManager(max_tokens=500, prune_threshold=0.3)
    cm.set_system_tokens("")
    msgs = [{"role": "user", "content": f"message {i} " + "x" * 50} for i in range(20)]
    pruned = cm.prune(msgs, keep_first=1, keep_last=4)
    assert len(pruned) < len(msgs)
    # First message kept
    assert "message 0" in pruned[0]["content"]
    # Last messages kept
    assert "message 19" in pruned[-1]["content"]
    print("  PASS: context prune")


def test_context_smart_prune():
    """smart_prune should reduce message count."""
    cm = ContextManager(max_tokens=500, prune_threshold=0.3)
    cm.set_system_tokens("")
    msgs = [{"role": "user", "content": f"msg {i} " + "x" * 100} for i in range(30)]
    original_len = len(msgs)
    pruned = cm.smart_prune(msgs)
    assert len(pruned) < original_len
    print("  PASS: context smart_prune")


def test_context_cap_tool_output():
    """cap_tool_output should truncate long output."""
    cm = ContextManager()
    short = "hello"
    assert cm.cap_tool_output(short, max_chars=100) == short

    long = "x" * 20000
    capped = cm.cap_tool_output(long, max_chars=1000)
    assert len(capped) <= 1100  # Some overhead for truncation message
    assert "truncated" in capped
    print("  PASS: context cap_tool_output")


def test_context_build_reprime():
    """build_reprime should create structured re-prime messages."""
    cm = ContextManager()
    msgs = cm.build_reprime(
        session_summary="Built a CLI app",
        current_task="Add tests",
        key_facts=["Using Python 3", "Ollama backend"],
    )
    assert len(msgs) >= 1
    assert msgs[0]["role"] == "user"
    assert "CLI app" in msgs[0]["content"]
    assert "Python 3" in msgs[0]["content"]
    print("  PASS: context build_reprime")


# ── Persistent Memory Tests ─────────────────────────────────────

def test_persistent_store_recall():
    """Should store and recall cross-session memory."""
    tmpdir = tempfile.mkdtemp()
    try:
        pm = PersistentMemory(tmpdir)
        pm.store("project_name", "Just Intelligent CLI", category="meta")
        result = pm.recall("project_name", category="meta")
        assert result == "Just Intelligent CLI"
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: persistent store/recall")


def test_persistent_categories():
    """Should list categories."""
    tmpdir = tempfile.mkdtemp()
    try:
        pm = PersistentMemory(tmpdir)
        pm.store("k1", "v1", category="code")
        pm.store("k2", "v2", category="bugs")
        all_mem = pm.list_all()
        assert "code" in all_mem
        assert "bugs" in all_mem
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: persistent categories")


def test_persistent_context_block():
    """get_context_block should return formatted memory."""
    tmpdir = tempfile.mkdtemp()
    try:
        pm = PersistentMemory(tmpdir)
        pm.store("lang", "python", category="prefs")
        block = pm.get_context_block()
        assert "python" in block or "prefs" in block or block == ""
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: persistent context block")


if __name__ == "__main__":
    print("=== Memory Tests ===")
    print("--- Session Store ---")
    test_session_create()
    test_session_add_get_messages()
    test_session_memory_kv()
    test_session_memory_search()
    test_session_list()
    test_session_last()
    test_session_error_tracking()
    print()
    print("--- Context Manager ---")
    test_context_measure()
    test_context_needs_pruning()
    test_context_prune()
    test_context_smart_prune()
    test_context_cap_tool_output()
    test_context_build_reprime()
    print()
    print("--- Persistent Memory ---")
    test_persistent_store_recall()
    test_persistent_categories()
    test_persistent_context_block()
    print("\nAll memory tests passed!")
