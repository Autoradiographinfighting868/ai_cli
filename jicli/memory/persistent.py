"""Cross-session persistent memory — file-backed for transparency."""

import os
import json
from datetime import datetime, timezone


class PersistentMemory:
    """File-backed persistent memory. Human-readable JSON files.
    
    Categories:
    - preferences: User preferences and settings
    - patterns: Learned coding patterns and conventions
    - errors: Common errors and their resolutions
    - facts: Project/environment facts
    """

    def __init__(self, memory_dir: str = None):
        self.memory_dir = memory_dir or os.path.join(os.getcwd(), ".jicli_memory")
        os.makedirs(self.memory_dir, exist_ok=True)

    def _category_path(self, category: str) -> str:
        # Sanitize category name
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in category)
        return os.path.join(self.memory_dir, f"{safe}.json")

    def _load_category(self, category: str) -> dict:
        path = self._category_path(category)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_category(self, category: str, data: dict):
        path = self._category_path(category)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def store(self, key: str, value: str, category: str = "general"):
        """Store a memory fact."""
        data = self._load_category(category)
        data[key] = {
            "value": value,
            "updated": datetime.now(timezone.utc).isoformat(),
            "access_count": data.get(key, {}).get("access_count", 0),
        }
        self._save_category(category, data)

    def recall(self, key: str, category: str = "general") -> str:
        """Recall a memory fact."""
        data = self._load_category(category)
        entry = data.get(key)
        if entry:
            entry["access_count"] = entry.get("access_count", 0) + 1
            self._save_category(category, data)
            return entry["value"]
        return None

    def search(self, query: str) -> list:
        """Search across all categories."""
        results = []
        if not os.path.isdir(self.memory_dir):
            return results
        for fname in os.listdir(self.memory_dir):
            if not fname.endswith(".json"):
                continue
            category = fname[:-5]
            data = self._load_category(category)
            for key, entry in data.items():
                value = entry.get("value", "") if isinstance(entry, dict) else str(entry)
                if query.lower() in key.lower() or query.lower() in value.lower():
                    results.append({
                        "key": key,
                        "value": value,
                        "category": category,
                    })
        return results

    def list_all(self) -> dict:
        """List all memories grouped by category."""
        result = {}
        if not os.path.isdir(self.memory_dir):
            return result
        for fname in os.listdir(self.memory_dir):
            if not fname.endswith(".json"):
                continue
            category = fname[:-5]
            data = self._load_category(category)
            result[category] = {
                k: v.get("value", v) if isinstance(v, dict) else v
                for k, v in data.items()
            }
        return result

    def forget(self, key: str, category: str = "general"):
        """Remove a memory fact."""
        data = self._load_category(category)
        data.pop(key, None)
        self._save_category(category, data)

    def get_context_block(self, max_chars: int = 2000) -> str:
        """Build a compact memory block for inclusion in system prompt."""
        all_mem = self.list_all()
        if not all_mem:
            return ""

        lines = ["## Persistent Memory"]
        char_count = 0
        for category, items in all_mem.items():
            lines.append(f"### {category}")
            for key, value in items.items():
                line = f"- {key}: {value}"
                if char_count + len(line) > max_chars:
                    lines.append("- ... (truncated)")
                    return "\n".join(lines)
                lines.append(line)
                char_count += len(line)

        return "\n".join(lines)
