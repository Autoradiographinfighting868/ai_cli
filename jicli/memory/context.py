"""Context window manager — hierarchical pruning and re-prime."""

from ..config import estimate_tokens, DEFAULT_CONTEXT_WINDOW, CHARS_PER_TOKEN


class ContextManager:
    """Manages the conversation context window.
    
    Strategies:
    1. Token tracking — know exactly how full the context is
    2. Proactive pruning — trim before hitting the limit
    3. Re-prime — throw away and restart with structured state
    4. Tool output capping — prevent single tool calls from eating context
    """

    def __init__(self, max_tokens: int = None, prune_threshold: float = 0.70):
        self.max_tokens = max_tokens or DEFAULT_CONTEXT_WINDOW
        self.prune_threshold = prune_threshold  # Prune at 70% capacity
        self._system_tokens = 0

    def set_system_tokens(self, system_prompt: str):
        """Track how many tokens the system prompt uses."""
        self._system_tokens = estimate_tokens(system_prompt)

    def available_tokens(self) -> int:
        """Tokens available for conversation (excluding system prompt)."""
        return self.max_tokens - self._system_tokens

    def measure_messages(self, messages: list) -> int:
        """Estimate total tokens across all messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle structured content blocks
                for block in content:
                    if isinstance(block, dict):
                        total += estimate_tokens(block.get("text", "") + block.get("content", ""))
                    elif isinstance(block, str):
                        total += estimate_tokens(block)
            elif isinstance(content, str):
                total += estimate_tokens(content)
            # Add overhead for role markers etc (~4 tokens per message)
            total += 4
        return total

    def needs_pruning(self, messages: list) -> bool:
        """Check if conversation needs pruning."""
        used = self.measure_messages(messages) + self._system_tokens
        return used >= (self.max_tokens * self.prune_threshold)

    def prune(self, messages: list, keep_first: int = 1, keep_last: int = 6) -> list:
        """Prune conversation while keeping earliest and most recent messages.
        
        Strategy: Keep first message (often important context), keep last N messages
        (recent state), collapse everything in between into a summary note.
        """
        if len(messages) <= keep_first + keep_last:
            return messages

        head = messages[:keep_first]
        tail = messages[-keep_last:]
        middle = messages[keep_first:-keep_last]

        # Build a terse summary of pruned messages
        summary_parts = []
        for msg in middle:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, str):
                # Take first 100 chars
                snippet = content[:100].replace("\n", " ")
                if len(content) > 100:
                    snippet += "..."
                summary_parts.append(f"[{role}] {snippet}")

        if summary_parts:
            summary = "[Earlier conversation pruned. Key points:\n" + "\n".join(summary_parts) + "\n]"
            bridge = [{"role": "user", "content": summary}]
        else:
            bridge = []

        return head + bridge + tail

    def smart_prune(self, messages: list) -> list:
        """Iteratively prune until under threshold or can't prune further."""
        keep_last = 6
        prev_len = len(messages) + 1
        while self.needs_pruning(messages) and keep_last >= 2 and len(messages) < prev_len:
            prev_len = len(messages)
            messages = self.prune(messages, keep_first=1, keep_last=keep_last)
            keep_last = max(2, keep_last - 2)
        return messages

    def build_reprime(self, session_summary: str, current_task: str,
                      key_facts: list = None, recent_results: list = None) -> list:
        """Build re-prime messages for a fresh conversation after context reset.
        
        Instead of summarizing within the same conversation, we:
        1. Kill the old conversation entirely
        2. Start fresh with structured state
        """
        parts = []

        if session_summary:
            parts.append(f"## Session Context\n{session_summary}")

        if key_facts:
            parts.append("## Key Facts\n" + "\n".join(f"- {f}" for f in key_facts))

        if recent_results:
            parts.append("## Recent Results\n" + "\n".join(recent_results[-3:]))

        if current_task:
            parts.append(f"## Current Task\n{current_task}")

        reprime_content = "\n\n".join(parts)

        return [
            {
                "role": "user",
                "content": (
                    "I'm continuing a previous session. Here's the relevant context:\n\n"
                    + reprime_content
                    + "\n\nPlease continue from where we left off."
                ),
            }
        ]

    def cap_tool_output(self, output: str, max_chars: int = 8000) -> str:
        """Cap tool output to prevent context explosion.
        
        Smart truncation: keeps head and tail, notes what was skipped.
        """
        if len(output) <= max_chars:
            return output

        half = max_chars // 2 - 50
        skipped = len(output) - max_chars + 100
        return (
            output[:half]
            + f"\n\n... [{skipped} characters truncated] ...\n\n"
            + output[-half:]
        )
