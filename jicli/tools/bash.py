"""Shell execution tool with safety checks."""

import os
import subprocess

from .base import Tool
from ..config import DANGEROUS_PATTERNS


class BashTool(Tool):
    name = "Bash"
    description = (
        "Execute a bash command and return its output. "
        "Use for running shell commands, installing packages, "
        "building projects, and system operations."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default: 120, max: 600)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, cwd: str = None):
        self._cwd = cwd or os.getcwd()

    @property
    def cwd(self):
        return self._cwd

    @cwd.setter
    def cwd(self, value):
        self._cwd = value

    def execute(self, args: dict) -> dict:
        cmd = args.get("command", "")
        timeout = min(args.get("timeout", 120), 600)

        if not cmd.strip():
            return self._err("Empty command")

        # Safety check
        warning = self._check_dangerous(cmd)
        if warning:
            return self._err(f"BLOCKED: {warning}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._cwd,
                env={**os.environ, "TERM": "dumb"},
            )

            output = result.stdout
            if result.stderr:
                output += ("\n" if output else "") + result.stderr

            output = output.strip()
            if not output:
                output = f"(exit code {result.returncode})"

            # Track cwd changes from cd commands
            if "cd " in cmd:
                self._track_cwd(cmd)

            return self._ok(self._truncate(output))

        except subprocess.TimeoutExpired:
            return self._err(f"Command timed out after {timeout}s")
        except Exception as e:
            return self._err(f"Execution error: {e}")

    def _check_dangerous(self, cmd: str) -> str:
        """Check for dangerous command patterns. Returns warning or empty string."""
        cmd_lower = cmd.lower().strip()
        for pattern in DANGEROUS_PATTERNS:
            if pattern.lower() in cmd_lower:
                return f"Dangerous pattern detected: '{pattern}'"
        return ""

    def _track_cwd(self, cmd: str):
        """Try to track directory changes from cd commands."""
        try:
            # Run the command and capture the resulting cwd
            result = subprocess.run(
                f"{cmd} && pwd",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
                cwd=self._cwd,
            )
            if result.returncode == 0 and result.stdout.strip():
                new_cwd = result.stdout.strip().split("\n")[-1]
                if os.path.isdir(new_cwd):
                    self._cwd = new_cwd
        except Exception:
            pass
