"""Task planner — pre-think and decompose before executing."""

from ..client import OllamaClient


PLANNER_PROMPT = """You are a task planner. Given a user request, break it down into concrete steps.

Rules:
- Each step should be a single, clear action
- Steps should be in order of execution
- Include what tools would be needed for each step
- Keep it concise — no more than 8 steps
- If the task is simple (1-2 steps), just list those

Output format (one step per line):
1. [tool] Description of what to do
2. [tool] Description of what to do
...

Available tools: {tools}

User request: {request}

Steps:"""


class Planner:
    """Pre-think planner that breaks tasks into steps before execution.
    
    For complex tasks, the planner:
    1. Analyzes the request
    2. Identifies required tools
    3. Orders the steps logically
    4. Returns a structured plan
    
    For simple tasks (detected by heuristics), it skips planning.
    """

    def __init__(self, client: OllamaClient):
        self.client = client

    def should_plan(self, request: str) -> bool:
        """Heuristic: does this request need multi-step planning?"""
        # Short, simple requests don't need planning
        if len(request) < 50:
            return False
        
        # Look for complexity indicators
        complexity_words = [
            "create", "build", "implement", "refactor", "fix and",
            "multiple", "all", "each", "every", "then",
            "first", "second", "after", "before",
            "project", "application", "system",
        ]
        request_lower = request.lower()
        hits = sum(1 for w in complexity_words if w in request_lower)
        return hits >= 2

    def plan(self, request: str, tool_names: list) -> list:
        """Generate a plan for a complex task. Returns list of step strings."""
        prompt = PLANNER_PROMPT.format(
            tools=", ".join(tool_names),
            request=request,
        )

        result = self.client.stream_collect(
            messages=[{"role": "user", "content": prompt}],
            thinking=False,
        )

        content = result.get("content", "")
        
        # Parse numbered steps
        steps = []
        for line in content.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Strip numbering
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    steps.append(clean)

        return steps if steps else [request]

    def format_plan_context(self, steps: list) -> str:
        """Format a plan as context to inject into the conversation."""
        if not steps or len(steps) <= 1:
            return ""

        lines = ["I've analyzed this request and here's my plan:"]
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("\nI'll execute these steps now.")
        return "\n".join(lines)
