"""Core agent loop — orchestrates LLM ↔ Tool interactions."""

import json
import sys
import re

from .client import OllamaClient
from .tools import ToolRegistry
from .memory.context import ContextManager
from .memory.session import SessionStore
from .config import estimate_tokens


class AgentLoop:
    """The main agent loop: User → LLM → Tool → Result → LLM → ... → Done.
    
    Features:
    - Streaming with real-time output
    - Tool call parsing from LLM output
    - Context management with auto-pruning
    - Loop detection (repeated tool calls)
    - Error tracking
    """

    def __init__(self, client: OllamaClient, registry: ToolRegistry,
                 context_mgr: ContextManager, session_store: SessionStore = None,
                 max_turns: int = 50, on_token: callable = None,
                 on_tool_call: callable = None, on_tool_result: callable = None,
                 verbose: bool = False):
        self.client = client
        self.registry = registry
        self.context = context_mgr
        self.store = session_store
        self.max_turns = max_turns
        self.on_token = on_token or (lambda text, **kw: None)
        self.on_tool_call = on_tool_call or (lambda name, args: None)
        self.on_tool_result = on_tool_result or (lambda name, result: None)
        self.verbose = verbose
        self._recent_calls = []  # For loop detection

    def run(self, messages: list, system_prompt: str, session_id: str = None) -> dict:
        """Run the agent loop until completion or max turns."""
        turns = 0
        total_input_tokens = 0
        total_output_tokens = 0

        # Set system tokens for context tracking
        self.context.set_system_tokens(system_prompt)

        while turns < self.max_turns:
            turns += 1

            # Check context and prune if needed
            if self.context.needs_pruning(messages):
                messages = self.context.smart_prune(messages)

            # Build messages with system prompt
            full_messages = [{"role": "system", "content": system_prompt}] + messages

            # Determine if model supports thinking
            thinking = self._should_think()

            # Buffer the response first to detect tool calls before showing output
            response = self._stream_response(full_messages, thinking, buffer=True)

            content = response["content"]
            thinking_text = response["thinking"]
            total_input_tokens += estimate_tokens(
                system_prompt + "".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
            )
            total_output_tokens += estimate_tokens(content + thinking_text)

            # Try to parse tool calls from the response
            tool_calls = self._parse_tool_calls(content)

            if tool_calls:
                # Show only non-tool-call text to the user
                display_text = self._strip_tool_blocks(content).strip()
                if display_text:
                    self.on_token(display_text + "\n", is_thinking=False)
            else:
                # No tool calls — show the full response
                self.on_token(content, is_thinking=False)

            if tool_calls:
                # Execute tools and continue loop
                tool_results = []
                for call in tool_calls:
                    name = call["name"]
                    args = call["args"]

                    self.on_tool_call(name, args)

                    # Loop detection
                    call_sig = f"{name}:{json.dumps(args, sort_keys=True)}"
                    if self._recent_calls.count(call_sig) >= 3:
                        result = {
                            "content": f"Loop detected: '{name}' called 3+ times with same args. Try a different approach.",
                            "is_error": True,
                        }
                    else:
                        self._recent_calls.append(call_sig)
                        if len(self._recent_calls) > 20:
                            self._recent_calls = self._recent_calls[-10:]

                        result = self.registry.execute(name, args)

                    # Cap tool output
                    result["content"] = self.context.cap_tool_output(result["content"])

                    self.on_tool_result(name, result)
                    tool_results.append({"name": name, "result": result})

                # Build tool result message
                result_parts = []
                for tr in tool_results:
                    status = "ERROR" if tr["result"]["is_error"] else "OK"
                    result_parts.append(
                        f"[Tool: {tr['name']}] ({status})\n{tr['result']['content']}"
                    )

                messages.append({
                    "role": "assistant",
                    "content": content,
                })
                messages.append({
                    "role": "user",
                    "content": "Tool results:\n\n" + "\n\n---\n\n".join(result_parts),
                })

                # Persist to session store
                if self.store and session_id:
                    self.store.add_message(session_id, "assistant", content,
                                           thinking=thinking_text,
                                           token_estimate=estimate_tokens(content))
                    self.store.add_message(session_id, "user",
                                           "\n".join(r["content"] for r in [t["result"] for t in tool_results]),
                                           token_estimate=estimate_tokens(
                                               "".join(r["content"] for r in [t["result"] for t in tool_results])))

                continue  # Next turn

            else:
                # No tool calls — final response
                messages.append({"role": "assistant", "content": content})

                if self.store and session_id:
                    self.store.add_message(session_id, "assistant", content,
                                           thinking=thinking_text,
                                           token_estimate=estimate_tokens(content))
                break

        return {
            "turns": turns,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "messages": messages,
        }

    def _stream_response(self, messages: list, thinking: bool, buffer: bool = False) -> dict:
        """Stream a response from the LLM.
        
        If buffer=False (default), tokens are sent to on_token in real-time.
        If buffer=True, content is collected silently (caller decides what to display).
        """
        content = ""
        think_content = ""
        in_thinking = True

        try:
            for chunk in self.client.chat(messages, thinking=thinking, stream=True):
                msg = chunk.get("message", {})

                if msg.get("thinking"):
                    think_content += msg["thinking"]

                if msg.get("content"):
                    if in_thinking:
                        in_thinking = False
                    content += msg["content"]
                    if not buffer:
                        self.on_token(msg["content"], is_thinking=False)

        except Exception as e:
            error_msg = f"\n[Stream error: {e}]"
            content += error_msg
            if not buffer:
                self.on_token(error_msg, is_thinking=False)

        # Handle </think> leaks
        if "</think>" in content:
            parts = content.rsplit("</think>", 1)
            think_content = (think_content or "") + parts[0].strip()
            content = parts[1].strip()

        return {"content": content, "thinking": think_content}

    def _should_think(self) -> bool:
        """Determine if the current model should use thinking mode.
        
        Only enable for models where thinking doesn't prevent tool calls.
        For agentic use, thinking often consumes all output budget
        on smaller models, so we disable it by default and let the
        model think in its content output naturally.
        """
        # Disable thinking for agentic tool-calling mode
        # The model will still reason in its text output
        return False

    def _parse_tool_calls(self, content: str) -> list:
        """Parse tool calls from LLM output.
        
        Supports multiple formats:
        1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>  
        2. ```tool_call\n{"name": "...", "arguments": {...}}\n```
        3. Direct JSON object matching tool schemas
        """
        calls = []

        # Format 1: XML-style tool calls
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, content, re.DOTALL):
            try:
                obj = json.loads(match.group(1))
                name = obj.get("name", "")
                args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
                if name and self.registry.get(name):
                    calls.append({"name": name, "args": args})
            except json.JSONDecodeError:
                continue

        if calls:
            return calls

        # Format 2: Fenced code block tool calls
        fence_pattern = r'```(?:tool_call|json)?\s*\n(\{.*?\})\s*\n```'
        for match in re.finditer(fence_pattern, content, re.DOTALL):
            try:
                obj = json.loads(match.group(1))
                name = obj.get("name", "")
                args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
                if name and self.registry.get(name):
                    calls.append({"name": name, "args": args})
            except json.JSONDecodeError:
                continue

        if calls:
            return calls

        # Format 3: Try to find bare JSON objects that look like tool calls
        # Look for patterns like {"name": "ToolName", ...}
        bare_pattern = r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*\}'
        for match in re.finditer(bare_pattern, content):
            try:
                obj = json.loads(match.group(0))
                name = obj.get("name", "")
                args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
                if name and self.registry.get(name):
                    calls.append({"name": name, "args": args})
            except json.JSONDecodeError:
                continue

        return calls

    def _strip_tool_blocks(self, content: str) -> str:
        """Remove tool call blocks from content for display."""
        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL)
        content = re.sub(r'```(?:tool_call|json)?\s*\n\{.*?\}\s*\n```', '', content, flags=re.DOTALL)
        return content.strip()
