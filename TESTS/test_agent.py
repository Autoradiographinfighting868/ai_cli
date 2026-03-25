"""Tests for jicli/agent.py — Agent loop tool call parsing and display logic."""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jicli.agent import AgentLoop
from jicli.tools import ToolRegistry
from jicli.memory.context import ContextManager


def make_agent():
    """Create a minimal agent for testing parse/strip methods."""
    reg = ToolRegistry()
    reg.auto_discover()
    ctx = ContextManager()
    # Use a FakeClient since we're only testing parse/strip
    agent = AgentLoop(client=None, registry=reg, context_mgr=ctx)
    return agent


# ── Tool Call Parsing Tests ──────────────────────────────────────

def test_parse_fenced_tool_call():
    """Should parse fenced code block tool calls."""
    agent = make_agent()
    content = '''Let me list the files.

```tool_call
{"name": "LS", "arguments": {"path": "."}}
```
'''
    calls = agent._parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["name"] == "LS"
    assert calls[0]["args"]["path"] == "."
    print("  PASS: parse fenced tool call")


def test_parse_xml_tool_call():
    """Should parse XML-style tool calls."""
    agent = make_agent()
    content = '''I'll check the file.

<tool_call>{"name": "Read", "arguments": {"file_path": "test.txt"}}</tool_call>
'''
    calls = agent._parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["name"] == "Read"
    print("  PASS: parse XML tool call")


def test_parse_multiple_tool_calls():
    """Should parse multiple tool calls from one response."""
    agent = make_agent()
    content = '''Let me check several things.

```tool_call
{"name": "LS", "arguments": {"path": "."}}
```

```tool_call
{"name": "Read", "arguments": {"file_path": "README.md"}}
```
'''
    calls = agent._parse_tool_calls(content)
    assert len(calls) == 2
    assert calls[0]["name"] == "LS"
    assert calls[1]["name"] == "Read"
    print("  PASS: parse multiple tool calls")


def test_parse_no_tool_calls():
    """Should return empty list when no tool calls present."""
    agent = make_agent()
    content = "Just a regular message without any tool calls."
    calls = agent._parse_tool_calls(content)
    assert calls == []
    print("  PASS: parse no tool calls")


def test_parse_invalid_json():
    """Should skip invalid JSON in tool call blocks."""
    agent = make_agent()
    content = '''
```tool_call
{invalid json here}
```
'''
    calls = agent._parse_tool_calls(content)
    assert calls == []
    print("  PASS: parse invalid JSON")


def test_parse_unknown_tool():
    """Should skip tool calls for unregistered tools."""
    agent = make_agent()
    content = '''
```tool_call
{"name": "FakeNonexistentTool", "arguments": {"a": 1}}
```
'''
    calls = agent._parse_tool_calls(content)
    assert calls == []
    print("  PASS: parse unknown tool")


def test_parse_json_block():
    """Should parse tool calls in plain json fenced blocks."""
    agent = make_agent()
    content = '''
```json
{"name": "Bash", "arguments": {"command": "echo hi"}}
```
'''
    calls = agent._parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0]["name"] == "Bash"
    print("  PASS: parse json fenced block")


def test_parse_args_variations():
    """Should handle different argument key names."""
    agent = make_agent()
    # 'args' key
    content1 = '''
```tool_call
{"name": "LS", "args": {"path": "."}}
```
'''
    calls1 = agent._parse_tool_calls(content1)
    assert len(calls1) == 1
    assert calls1[0]["args"]["path"] == "."

    # 'parameters' key
    content2 = '''
```tool_call
{"name": "LS", "parameters": {"path": "/"}}
```
'''
    calls2 = agent._parse_tool_calls(content2)
    assert len(calls2) == 1
    assert calls2[0]["args"]["path"] == "/"
    print("  PASS: parse args variations")


# ── Strip Tool Blocks Tests ─────────────────────────────────────

def test_strip_fenced_blocks():
    """Should strip fenced tool call blocks from content."""
    agent = make_agent()
    content = '''Here is some text.

```tool_call
{"name": "LS", "arguments": {"path": "."}}
```

And more text.'''
    stripped = agent._strip_tool_blocks(content)
    assert "tool_call" not in stripped
    assert "LS" not in stripped
    assert "Here is some text" in stripped
    assert "And more text" in stripped
    print("  PASS: strip fenced blocks")


def test_strip_xml_blocks():
    """Should strip XML tool call blocks from content."""
    agent = make_agent()
    content = 'Check this: <tool_call>{"name": "LS"}</tool_call> done.'
    stripped = agent._strip_tool_blocks(content)
    assert "<tool_call>" not in stripped
    assert "done" in stripped
    print("  PASS: strip XML blocks")


def test_strip_no_blocks():
    """Should return content unchanged when no tool blocks."""
    agent = make_agent()
    content = "Regular text with no tool calls."
    stripped = agent._strip_tool_blocks(content)
    assert stripped == content
    print("  PASS: strip no blocks")


# ── Think Mode Tests ─────────────────────────────────────────────

def test_should_think_disabled():
    """Thinking should be disabled for agentic mode."""
    agent = make_agent()
    assert agent._should_think() is False
    print("  PASS: should_think disabled")


if __name__ == "__main__":
    print("=== Agent Tests ===")
    print("--- Tool Call Parsing ---")
    test_parse_fenced_tool_call()
    test_parse_xml_tool_call()
    test_parse_multiple_tool_calls()
    test_parse_no_tool_calls()
    test_parse_invalid_json()
    test_parse_unknown_tool()
    test_parse_json_block()
    test_parse_args_variations()
    print()
    print("--- Strip Tool Blocks ---")
    test_strip_fenced_blocks()
    test_strip_xml_blocks()
    test_strip_no_blocks()
    print()
    print("--- Think Mode ---")
    test_should_think_disabled()
    print("\nAll agent tests passed!")
