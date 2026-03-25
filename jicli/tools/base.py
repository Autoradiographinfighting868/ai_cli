"""Base class for Just Intelligent CLI tools. All tools inherit from this."""

from typing import Any


class Tool:
    """Base tool interface. Subclass and implement execute()."""

    name: str = ""
    description: str = ""
    parameters: dict = {}  # JSON Schema for input

    def execute(self, args: dict) -> dict:
        """Execute the tool. Returns {"content": str, "is_error": bool}."""
        raise NotImplementedError

    def definition(self) -> dict:
        """Return the tool definition for the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def _ok(self, content: str) -> dict:
        return {"content": content, "is_error": False}

    def _err(self, content: str) -> dict:
        return {"content": content, "is_error": True}

    def _truncate(self, text: str, max_len: int = 10000) -> str:
        """Truncate output to max_len chars, preserving head and tail."""
        if len(text) <= max_len:
            return text
        half = max_len // 2 - 50
        skipped = len(text) - max_len + 100
        return (
            text[:half]
            + f"\n\n... [{skipped} characters truncated] ...\n\n"
            + text[-half:]
        )
