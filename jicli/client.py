"""Ollama HTTP client — lean, streaming-first, with retry logic."""

import json
import time
import requests
from typing import Generator, Optional

from .config import resolve_host, OLLAMA_OPTIONS, DEFAULT_TEMPERATURE, DEFAULT_CONTEXT_WINDOW


class OllamaClient:
    """Talks to a local Ollama server. Streaming and non-streaming chat."""

    def __init__(self, model: str, host: str = None, options: dict = None):
        self.model = model
        self.host = resolve_host(host)
        self.options = options or OLLAMA_OPTIONS.copy()
        self._max_retries = 3

    def chat(self, messages: list, thinking: bool = False,
             stream: bool = True, options: dict = None) -> dict | Generator:
        """Send a chat request. Returns dict (non-stream) or generator (stream)."""
        opts = (options or self.options).copy()
        opts.setdefault("num_ctx", DEFAULT_CONTEXT_WINDOW)
        opts.setdefault("temperature", DEFAULT_TEMPERATURE)

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": thinking,
            "options": opts,
        }

        endpoint = f"{self.host}/api/chat"
        headers = {"Content-Type": "application/json"}

        if stream:
            return self._stream_with_retry(endpoint, headers, payload)
        else:
            return self._request_with_retry(endpoint, headers, payload)

    def _request_with_retry(self, endpoint: str, headers: dict, payload: dict) -> dict:
        """Non-streaming request with retry on transient errors."""
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload, timeout=300)
                resp.raise_for_status()
                return self._parse_response(resp.json())
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    # On 500 "EOF" errors, retry with think toggled off if it was on
                    if hasattr(e, 'response') and e.response is not None and e.response.status_code == 500:
                        time.sleep(1 * (attempt + 1))
                        continue
                    time.sleep(0.5)
        raise last_err

    def _stream_with_retry(self, endpoint: str, headers: dict, payload: dict) -> Generator:
        """Streaming with retry on errors. Eagerly connects to check for errors."""
        last_err = None
        for attempt in range(self._max_retries):
            try:
                resp = requests.post(endpoint, headers=headers, json=payload,
                                     stream=True, timeout=300)
                resp.raise_for_status()
                # Successfully connected, yield from this response
                return self._iter_stream(resp)
            except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    time.sleep(1 * (attempt + 1))
        raise last_err

    def _iter_stream(self, resp) -> Generator:
        """Iterate over a streaming response, yielding parsed chunks."""
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.strip():
                    continue
                try:
                    chunk = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                yield chunk
                if chunk.get("done"):
                    break
        finally:
            resp.close()

    def _stream(self, endpoint: str, headers: dict, payload: dict) -> Generator:
        """Stream chat responses, yielding parsed chunks."""
        resp = requests.post(endpoint, headers=headers, json=payload,
                             stream=True, timeout=300)
        resp.raise_for_status()
        return self._iter_stream(resp)

    def _parse_response(self, data: dict) -> dict:
        """Parse a non-streaming Ollama response into a clean format."""
        message = data.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")

        # Handle </think> tag leaking into content
        if "</think>" in content:
            parts = content.rsplit("</think>", 1)
            thinking = (thinking or "") + parts[0].strip()
            content = parts[1].strip()

        return {
            "content": content,
            "thinking": thinking,
            "model": data.get("model", ""),
            "done": data.get("done", True),
            "total_duration": data.get("total_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
        }

    def stream_collect(self, messages: list, thinking: bool = False,
                       options: dict = None,
                       on_token: callable = None) -> dict:
        """Stream a response, collecting full content. Optionally call on_token for each chunk."""
        content = ""
        think_content = ""
        in_thinking = True

        for chunk in self.chat(messages, thinking=thinking, stream=True, options=options):
            msg = chunk.get("message", {})

            if msg.get("thinking"):
                think_content += msg["thinking"]
                if on_token:
                    on_token(msg["thinking"], is_thinking=True)

            if msg.get("content"):
                if in_thinking and msg["content"]:
                    in_thinking = False
                content += msg["content"]
                if on_token:
                    on_token(msg["content"], is_thinking=False)

        # Handle </think> tag leaks
        if "</think>" in content:
            parts = content.rsplit("</think>", 1)
            think_content = (think_content or "") + parts[0].strip()
            content = parts[1].strip()

        return {
            "content": content,
            "thinking": think_content,
        }

    def list_models(self) -> list:
        """Fetch available models from Ollama server."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", []) if m.get("name")]
        except requests.exceptions.RequestException:
            return []

    def ping(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            requests.get(f"{self.host}/api/tags", timeout=5)
            return True
        except requests.exceptions.RequestException:
            return False
