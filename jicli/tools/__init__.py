"""Tool registry with auto-discovery from the tools/ package."""

import os
import importlib
import inspect

from .base import Tool


class ToolRegistry:
    """Central registry for all tools. Supports auto-discovery and filtering."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._allowed: set = None
        self._disallowed: set = None

    def register(self, tool: Tool):
        """Register a tool instance."""
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        """Remove a tool by name."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        return self._tools.get(name)

    def execute(self, name: str, args: dict) -> dict:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if not tool:
            return {"content": f"Unknown tool: {name}. Available: {', '.join(self._tools.keys())}", "is_error": True}
        try:
            return tool.execute(args)
        except Exception as e:
            return {"content": f"Tool '{name}' error: {e}", "is_error": True}

    def definitions(self) -> list:
        """Get tool definitions for the LLM, respecting allow/disallow filters."""
        defs = []
        for name, tool in self._tools.items():
            if self._disallowed and name in self._disallowed:
                continue
            if self._allowed and name not in self._allowed:
                continue
            defs.append(tool.definition())
        return defs

    def names(self) -> list:
        """List registered tool names."""
        return list(self._tools.keys())

    def set_filter(self, allowed: list = None, disallowed: list = None):
        """Set tool filters."""
        self._allowed = set(allowed) if allowed else None
        self._disallowed = set(disallowed) if disallowed else None

    def auto_discover(self, cwd: str = None):
        """Auto-discover and register all built-in tools."""
        from .bash import BashTool
        from .filesystem import ReadTool, WriteTool, EditTool, GlobTool, GrepTool, ListDirTool

        effective_cwd = cwd or os.getcwd()

        self.register(BashTool(cwd=effective_cwd))
        self.register(ReadTool(cwd=effective_cwd))
        self.register(WriteTool(cwd=effective_cwd))
        self.register(EditTool(cwd=effective_cwd))
        self.register(GlobTool(cwd=effective_cwd))
        self.register(GrepTool(cwd=effective_cwd))
        self.register(ListDirTool(cwd=effective_cwd))

    def load_plugins(self, plugin_dir: str):
        """Load tool plugins from an external directory.
        
        Plugin files should contain classes that subclass Tool.
        They are auto-detected and registered.
        """
        if not os.path.isdir(plugin_dir):
            return

        for fname in os.listdir(plugin_dir):
            if not fname.endswith(".py") or fname.startswith("_"):
                continue

            module_path = os.path.join(plugin_dir, fname)
            module_name = f"jicli_plugin_{fname[:-3]}"

            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (inspect.isclass(attr)
                            and issubclass(attr, Tool)
                            and attr is not Tool
                            and attr.name):
                        self.register(attr())
            except Exception as e:
                import sys
                print(f"Warning: Failed to load plugin {fname}: {e}", file=sys.stderr)

    def update_cwd(self, new_cwd: str):
        """Update working directory for all tools that track it."""
        for tool in self._tools.values():
            if hasattr(tool, "_cwd"):
                tool._cwd = new_cwd
            if hasattr(tool, "cwd"):
                try:
                    tool.cwd = new_cwd
                except AttributeError:
                    pass
