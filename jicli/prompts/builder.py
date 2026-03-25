"""Template-based prompt builder."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path


TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "TEMPLATES")


def load_template(name: str) -> str:
    """Load a template file by name (without extension)."""
    path = os.path.join(TEMPLATES_DIR, f"{name}.md")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Template not found: {path}")
    with open(path, "r") as f:
        return f.read()


def build_system_prompt(cwd: str, model: str, tool_definitions: list,
                        memory_block: str = "", error_context: str = "") -> str:
    """Build the system prompt from template + dynamic values."""
    try:
        template = load_template("system")
    except FileNotFoundError:
        template = _fallback_system_template()

    # Format tool definitions
    tool_text = ""
    for tool in tool_definitions:
        tool_text += f"\n### {tool['name']}\n{tool['description']}\n"
        params = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])
        if params:
            tool_text += "Parameters:\n"
            for pname, pinfo in params.items():
                req = " (required)" if pname in required else ""
                desc = pinfo.get("description", "")
                tool_text += f"- `{pname}`{req}: {desc}\n"

    # Use safe substitution to handle template braces in JSON examples
    replacements = {
        "{tool_definitions}": tool_text,
        "{cwd}": cwd,
        "{platform}": sys.platform,
        "{date}": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "{model}": model,
        "{memory_block}": memory_block,
        "{error_context}": error_context,
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def build_reprime_prompt(session_summary: str = "", key_facts: list = None,
                         accomplishments: str = "", current_task: str = "") -> str:
    """Build a re-prime prompt for conversation reset."""
    try:
        template = load_template("reprime")
    except FileNotFoundError:
        template = _fallback_reprime_template()

    facts_text = "\n".join(f"- {f}" for f in (key_facts or []))

    replacements = {
        "{session_summary}": session_summary or "No previous context.",
        "{key_facts}": facts_text or "None recorded.",
        "{accomplishments}": accomplishments or "Starting fresh.",
        "{current_task}": current_task or "Awaiting instructions.",
    }
    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def format_error_context(errors: list) -> str:
    """Format common errors into a prompt section."""
    if not errors:
        return ""

    lines = ["# Known Issues (avoid repeating these)"]
    for err in errors[:5]:
        lines.append(f"- {err['error_type']}: {err['description']}")
        if err.get("resolution"):
            lines.append(f"  Resolution: {err['resolution']}")

    return "\n".join(lines)


def _fallback_system_template() -> str:
    """Fallback system prompt if template file is missing."""
    return """You are JI, an AI agent for software engineering tasks using local Ollama models.

Use these tools to help the user:
{tool_definitions}

# Environment
Working directory: {cwd}
Platform: {platform}
Date: {date}
Model: {model}

Call tools using: <tool_call>{{"name": "TOOL_NAME", "arguments": {{}}}}</tool_call>

{memory_block}
{error_context}"""


def _fallback_reprime_template() -> str:
    return """Continuing previous session.
Summary: {session_summary}
Facts: {key_facts}
Done: {accomplishments}
Current: {current_task}"""
