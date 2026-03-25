"""Filesystem tools — Read, Write, Glob, Grep."""

import os
import fnmatch
import subprocess

from .base import Tool


class ReadTool(Tool):
    name = "Read"
    description = (
        "Read a file from the filesystem. Returns content with line numbers. "
        "Use offset/limit to read specific sections of large files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file",
            },
            "offset": {
                "type": "number",
                "description": "Line number to start from (1-indexed, default: 1)",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of lines to read",
            },
        },
        "required": ["file_path"],
    }

    def __init__(self, cwd: str = None):
        self._cwd = cwd or os.getcwd()

    def execute(self, args: dict) -> dict:
        path = args["file_path"]
        offset = max(1, int(args.get("offset", 1)))
        limit = args.get("limit")

        # Resolve relative paths
        if not os.path.isabs(path):
            path = os.path.join(self._cwd, path)

        if not os.path.exists(path):
            return self._err(f"File not found: {path}")
        if not os.path.isfile(path):
            return self._err(f"Not a file: {path}")

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total = len(lines)
            start = offset - 1
            end = total if limit is None else min(start + int(limit), total)
            selected = lines[start:end]

            result_lines = []
            for i, line in enumerate(selected, start + 1):
                result_lines.append(f"{i:5d} | {line.rstrip()}")

            header = f"[{path}] ({total} lines total, showing {start+1}-{end})"
            return self._ok(self._truncate(header + "\n" + "\n".join(result_lines)))

        except Exception as e:
            return self._err(f"Error reading file: {e}")


class WriteTool(Tool):
    name = "Write"
    description = (
        "Write content to a file. Creates parent directories if needed. "
        "Overwrites existing content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to write to",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    }

    def __init__(self, cwd: str = None):
        self._cwd = cwd or os.getcwd()

    def execute(self, args: dict) -> dict:
        path = args["file_path"]
        content = args["content"]

        if not os.path.isabs(path):
            path = os.path.join(self._cwd, path)

        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return self._ok(f"Wrote {len(content)} chars ({lines} lines) to {path}")
        except Exception as e:
            return self._err(f"Error writing file: {e}")


class EditTool(Tool):
    name = "Edit"
    description = (
        "Edit a file by replacing an exact string match with new content. "
        "The old_string must match exactly (including whitespace). "
        "Use Read first to see the file content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find and replace",
            },
            "new_string": {
                "type": "string",
                "description": "Text to replace old_string with",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def __init__(self, cwd: str = None):
        self._cwd = cwd or os.getcwd()

    def execute(self, args: dict) -> dict:
        path = args["file_path"]
        old = args["old_string"]
        new = args["new_string"]

        if not os.path.isabs(path):
            path = os.path.join(self._cwd, path)

        if not os.path.exists(path):
            return self._err(f"File not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            count = content.count(old)
            if count == 0:
                return self._err("old_string not found in file. Use Read to see current content.")
            if count > 1:
                return self._err(f"old_string matches {count} locations. Make it more specific.")

            new_content = content.replace(old, new, 1)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return self._ok(f"Edited {path} (replaced 1 occurrence)")
        except Exception as e:
            return self._err(f"Error editing file: {e}")


class GlobTool(Tool):
    name = "Glob"
    description = (
        "Find files matching a glob pattern. Returns paths sorted by modification time (newest first). "
        "Searches from working directory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g., '**/*.py', 'src/*.js')",
            },
            "path": {
                "type": "string",
                "description": "Directory to search from (default: working directory)",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str = None):
        self._cwd = cwd or os.getcwd()

    def execute(self, args: dict) -> dict:
        pattern = args["pattern"]
        search_dir = args.get("path", self._cwd)

        if not os.path.isabs(search_dir):
            search_dir = os.path.join(self._cwd, search_dir)

        try:
            matches = []
            for root, dirs, files in os.walk(search_dir):
                # Skip hidden dirs and common noise
                dirs[:] = [d for d in dirs if not d.startswith(".")
                           and d not in ("__pycache__", "node_modules", ".git")]

                for fname in files:
                    if fname.startswith("."):
                        continue
                    full = os.path.join(root, fname)
                    rel = os.path.relpath(full, search_dir)
                    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fname, pattern):
                        matches.append(full)

            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            if not matches:
                return self._ok(f"No files matching '{pattern}'")

            lines = [os.path.relpath(m, self._cwd) for m in matches[:100]]
            header = f"Found {len(matches)} files"
            if len(matches) > 100:
                header += f" (showing first 100)"
            return self._ok(header + "\n" + "\n".join(lines))
        except Exception as e:
            return self._err(f"Error during glob: {e}")


class GrepTool(Tool):
    name = "Grep"
    description = (
        "Search for text patterns in files recursively. "
        "Uses grep under the hood. Returns matching lines with file paths and line numbers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text or regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search (default: working directory)",
            },
            "include": {
                "type": "string",
                "description": "File pattern to include (e.g., '*.py')",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, cwd: str = None):
        self._cwd = cwd or os.getcwd()

    def execute(self, args: dict) -> dict:
        pattern = args["pattern"]
        search_path = args.get("path", ".")
        include = args.get("include")

        if not os.path.isabs(search_path):
            search_path = os.path.join(self._cwd, search_path)

        try:
            cmd = ["grep", "-rn", "--color=never", pattern, search_path]
            if include:
                cmd.extend(["--include", include])
            # Exclude common noise
            cmd.extend(["--exclude-dir=.git", "--exclude-dir=__pycache__",
                         "--exclude-dir=node_modules"])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=self._cwd
            )

            output = result.stdout.strip()
            if not output:
                return self._ok(f"No matches for '{pattern}'")

            return self._ok(self._truncate(output))
        except subprocess.TimeoutExpired:
            return self._err("Grep timed out after 30s")
        except Exception as e:
            return self._err(f"Grep error: {e}")


class ListDirTool(Tool):
    name = "LS"
    description = (
        "List directory contents. Shows files and subdirectories with sizes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: working directory)",
            },
        },
        "required": [],
    }

    def __init__(self, cwd: str = None):
        self._cwd = cwd or os.getcwd()

    def execute(self, args: dict) -> dict:
        path = args.get("path", self._cwd)
        if not os.path.isabs(path):
            path = os.path.join(self._cwd, path)

        if not os.path.isdir(path):
            return self._err(f"Not a directory: {path}")

        try:
            entries = []
            for name in sorted(os.listdir(path)):
                full = os.path.join(path, name)
                if os.path.isdir(full):
                    entries.append(f"  {name}/")
                else:
                    size = os.path.getsize(full)
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1048576:
                        size_str = f"{size/1024:.1f}K"
                    else:
                        size_str = f"{size/1048576:.1f}M"
                    entries.append(f"  {name}  ({size_str})")

            return self._ok(f"[{path}]\n" + "\n".join(entries))
        except Exception as e:
            return self._err(f"Error listing directory: {e}")
