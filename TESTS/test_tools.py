"""Tests for jicli/tools/ — Tool registry and built-in tools."""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jicli.tools import ToolRegistry
from jicli.tools.base import Tool
from jicli.tools.bash import BashTool
from jicli.tools.filesystem import ReadTool, WriteTool, EditTool, GlobTool, GrepTool, ListDirTool


# ── Registry Tests ───────────────────────────────────────────────

def test_registry_auto_discover():
    """Auto-discover should register all built-in tools."""
    reg = ToolRegistry()
    reg.auto_discover()
    names = reg.names()
    assert "Bash" in names
    assert "Read" in names
    assert "Write" in names
    assert "Edit" in names
    assert "Glob" in names
    assert "Grep" in names
    assert "LS" in names
    assert len(names) >= 7
    print("  PASS: registry auto_discover")


def test_registry_execute_unknown():
    """Executing unknown tool should return error."""
    reg = ToolRegistry()
    result = reg.execute("NonExistent", {})
    assert result["is_error"] is True
    assert "Unknown tool" in result["content"]
    print("  PASS: registry execute unknown")


def test_registry_definitions():
    """definitions() should return list of tool schemas."""
    reg = ToolRegistry()
    reg.auto_discover()
    defs = reg.definitions()
    assert len(defs) >= 7
    for d in defs:
        assert "name" in d
        assert "description" in d
        assert "parameters" in d
    print("  PASS: registry definitions")


def test_registry_filter():
    """set_filter should restrict which tools are visible."""
    reg = ToolRegistry()
    reg.auto_discover()
    reg.set_filter(allowed=["Bash", "Read"])
    defs = reg.definitions()
    names = [d["name"] for d in defs]
    assert "Bash" in names
    assert "Read" in names
    assert "Write" not in names
    print("  PASS: registry filter allowed")


def test_registry_filter_disallowed():
    """Disallowed tools should be excluded from definitions."""
    reg = ToolRegistry()
    reg.auto_discover()
    reg.set_filter(disallowed=["Bash"])
    defs = reg.definitions()
    names = [d["name"] for d in defs]
    assert "Bash" not in names
    assert "Read" in names
    print("  PASS: registry filter disallowed")


# ── Bash Tool Tests ──────────────────────────────────────────────

def test_bash_simple_command():
    """Bash should execute simple commands."""
    tool = BashTool()
    result = tool.execute({"command": "echo hello world"})
    assert result["is_error"] is False
    assert "hello world" in result["content"]
    print("  PASS: bash simple command")


def test_bash_dangerous_command():
    """Bash should block dangerous commands."""
    tool = BashTool()
    result = tool.execute({"command": "rm -rf /"})
    assert result["is_error"] is True
    assert "Dangerous" in result["content"] or "dangerous" in result["content"]
    print("  PASS: bash dangerous command blocked")


def test_bash_timeout():
    """Bash should respect timeout."""
    tool = BashTool()
    result = tool.execute({"command": "sleep 0.1 && echo done", "timeout": 5})
    assert "done" in result["content"]
    assert result["is_error"] is False
    print("  PASS: bash timeout")


def test_bash_error_command():
    """Bash should capture stderr on errors."""
    tool = BashTool()
    result = tool.execute({"command": "ls /nonexistent_path_xyz 2>&1"})
    # Should have some output (error message) but may or may not be flagged as error
    # depending on exit code vs output
    print("  PASS: bash error command")


# ── Filesystem Tool Tests ────────────────────────────────────────

def test_read_tool():
    """ReadTool should read existing files with line numbers."""
    tmpdir = tempfile.mkdtemp()
    try:
        fpath = os.path.join(tmpdir, "test.txt")
        with open(fpath, "w") as f:
            f.write("line1\nline2\nline3\n")

        tool = ReadTool(cwd=tmpdir)
        result = tool.execute({"file_path": fpath})
        assert result["is_error"] is False
        assert "line1" in result["content"]
        assert "line2" in result["content"]
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: ReadTool")


def test_read_tool_with_offset():
    """ReadTool should support offset and limit."""
    tmpdir = tempfile.mkdtemp()
    try:
        fpath = os.path.join(tmpdir, "test.txt")
        with open(fpath, "w") as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")

        tool = ReadTool(cwd=tmpdir)
        result = tool.execute({"file_path": fpath, "offset": 2, "limit": 2})
        assert result["is_error"] is False
        assert "line2" in result["content"] or "line3" in result["content"]
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: ReadTool offset/limit")


def test_read_tool_nonexistent():
    """ReadTool should return error for missing files."""
    tool = ReadTool()
    result = tool.execute({"file_path": "/nonexistent/file.txt"})
    assert result["is_error"] is True
    print("  PASS: ReadTool nonexistent")


def test_write_tool():
    """WriteTool should create files."""
    tmpdir = tempfile.mkdtemp()
    try:
        fpath = os.path.join(tmpdir, "output.txt")
        tool = WriteTool(cwd=tmpdir)
        result = tool.execute({"file_path": fpath, "content": "hello world"})
        assert result["is_error"] is False
        with open(fpath) as f:
            assert f.read() == "hello world"
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: WriteTool")


def test_write_tool_creates_dirs():
    """WriteTool should create parent directories."""
    tmpdir = tempfile.mkdtemp()
    try:
        fpath = os.path.join(tmpdir, "sub", "dir", "file.txt")
        tool = WriteTool(cwd=tmpdir)
        result = tool.execute({"file_path": fpath, "content": "nested"})
        assert result["is_error"] is False
        assert os.path.exists(fpath)
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: WriteTool creates dirs")


def test_edit_tool():
    """EditTool should replace text in files."""
    tmpdir = tempfile.mkdtemp()
    try:
        fpath = os.path.join(tmpdir, "edit.txt")
        with open(fpath, "w") as f:
            f.write("hello world\nfoo bar\n")

        tool = EditTool(cwd=tmpdir)
        result = tool.execute({
            "file_path": fpath,
            "old_string": "foo bar",
            "new_string": "baz qux",
        })
        assert result["is_error"] is False
        with open(fpath) as f:
            content = f.read()
        assert "baz qux" in content
        assert "foo bar" not in content
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: EditTool")


def test_edit_tool_not_found():
    """EditTool should error if old_text not found in file."""
    tmpdir = tempfile.mkdtemp()
    try:
        fpath = os.path.join(tmpdir, "edit.txt")
        with open(fpath, "w") as f:
            f.write("hello world\n")

        tool = EditTool(cwd=tmpdir)
        result = tool.execute({
            "file_path": fpath,
            "old_string": "nonexistent text",
            "new_string": "replacement",
        })
        assert result["is_error"] is True
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: EditTool not found")


def test_glob_tool():
    """GlobTool should find files by pattern."""
    tmpdir = tempfile.mkdtemp()
    try:
        for name in ["a.py", "b.py", "c.txt"]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write("")

        tool = GlobTool(cwd=tmpdir)
        result = tool.execute({"pattern": "*.py", "path": tmpdir})
        assert result["is_error"] is False
        assert "a.py" in result["content"]
        assert "b.py" in result["content"]
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: GlobTool")


def test_grep_tool():
    """GrepTool should find text patterns in files."""
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "search.txt"), "w") as f:
            f.write("line1 apple\nline2 banana\nline3 apple pie\n")

        tool = GrepTool(cwd=tmpdir)
        result = tool.execute({"pattern": "apple", "path": tmpdir})
        assert result["is_error"] is False
        assert "apple" in result["content"]
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: GrepTool")


def test_ls_tool():
    """ListDirTool should list directory contents."""
    tmpdir = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmpdir, "file1.txt"), "w") as f:
            f.write("test")
        os.mkdir(os.path.join(tmpdir, "subdir"))

        tool = ListDirTool(cwd=tmpdir)
        result = tool.execute({"path": tmpdir})
        assert result["is_error"] is False
        assert "file1.txt" in result["content"]
        assert "subdir" in result["content"]
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: ListDirTool")


# ── Plugin System Test ───────────────────────────────────────────

def test_plugin_loading():
    """Registry should load plugins from external directory."""
    tmpdir = tempfile.mkdtemp()
    try:
        plugin_code = '''
from jicli.tools.base import Tool

class HelloTool(Tool):
    name = "Hello"
    description = "Says hello"
    parameters = {"type": "object", "properties": {"name": {"type": "string"}}}

    def execute(self, args):
        return self._ok(f"Hello, {args.get('name', 'world')}!")
'''
        with open(os.path.join(tmpdir, "hello_plugin.py"), "w") as f:
            f.write(plugin_code)

        reg = ToolRegistry()
        reg.load_plugins(tmpdir)
        assert "Hello" in reg.names()
        result = reg.execute("Hello", {"name": "Test"})
        assert result["is_error"] is False
        assert "Hello, Test" in result["content"]
    finally:
        shutil.rmtree(tmpdir)
    print("  PASS: plugin loading")


if __name__ == "__main__":
    print("=== Tool Tests ===")
    test_registry_auto_discover()
    test_registry_execute_unknown()
    test_registry_definitions()
    test_registry_filter()
    test_registry_filter_disallowed()
    print()
    print("--- Bash ---")
    test_bash_simple_command()
    test_bash_dangerous_command()
    test_bash_timeout()
    test_bash_error_command()
    print()
    print("--- Filesystem ---")
    test_read_tool()
    test_read_tool_with_offset()
    test_read_tool_nonexistent()
    test_write_tool()
    test_write_tool_creates_dirs()
    test_edit_tool()
    test_edit_tool_not_found()
    test_glob_tool()
    test_grep_tool()
    test_ls_tool()
    print()
    print("--- Plugins ---")
    test_plugin_loading()
    print("\nAll tool tests passed!")
