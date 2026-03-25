"""CLI — argument parsing, mode dispatch, REPL."""

import argparse
import json
import os
import readline
import signal
import sys
from datetime import datetime, timezone

from . import __version__
from .config import resolve_model, MODEL_ALIASES, DEFAULT_MAX_TURNS
from .client import OllamaClient
from .tools import ToolRegistry
from .memory import SessionStore, ContextManager, PersistentMemory
from .agent import AgentLoop
from .planner import Planner
from .prompts import build_system_prompt, build_reprime_prompt, format_error_context


# ── Argument Parsing ─────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="jicli",
        description="Just Intelligent CLI — Agentic CLI powered by local Ollama models",
    )
    parser.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive)")
    parser.add_argument("-m", "--model", default="ji", help="Model name or alias")
    parser.add_argument("-t", "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                        help="Max agent turns")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--ndjson", action="store_true", help="NDJSON protocol mode")
    parser.add_argument("--resume", action="store_true", help="Resume last session")
    parser.add_argument("--plan", action="store_true",
                        help="Enable pre-think planning for complex tasks")
    parser.add_argument("--cwd", default=os.getcwd(), help="Working directory")
    parser.add_argument("--db", default=None, help="Path to session database")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--list-sessions", action="store_true", help="List recent sessions")
    parser.add_argument("--memory", action="store_true", help="Show persistent memory")
    parser.add_argument("--allowed-tools", help="Comma-separated allowed tools")
    parser.add_argument("--disallowed-tools", help="Comma-separated disallowed tools")
    parser.add_argument("--no-think", action="store_true", help="Disable model thinking")
    parser.add_argument("--version", action="version", version=f"jicli {__version__}")

    return parser.parse_args(argv)


# ── Main Entry Point ─────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    model = resolve_model(args.model)

    # Create client
    client = OllamaClient(model=model)

    # Handle info commands
    if args.list_models:
        _cmd_list_models(client)
        return

    # Session store (use jicli_v2.db to avoid conflict with v1)
    db_path = args.db or os.path.join(args.cwd, "jicli_v2.db")
    store = SessionStore(db_path)

    if args.list_sessions:
        _cmd_list_sessions(store)
        return

    # Persistent memory
    mem = PersistentMemory(os.path.join(args.cwd, ".jicli_memory"))

    if args.memory:
        _cmd_show_memory(mem)
        return

    # Check connectivity
    if not client.ping():
        _err("Cannot connect to Ollama. Is 'ollama serve' running?")
        sys.exit(1)

    _log(f"Using model: {model}", args.verbose)

    # Tool registry
    registry = ToolRegistry()
    registry.auto_discover(cwd=args.cwd)

    # Load plugins from ./plugins/ if exists
    plugin_dir = os.path.join(args.cwd, "plugins")
    if os.path.isdir(plugin_dir):
        registry.load_plugins(plugin_dir)
        _log(f"Loaded plugins from {plugin_dir}", args.verbose)

    if args.allowed_tools:
        registry.set_filter(allowed=args.allowed_tools.split(","))
    if args.disallowed_tools:
        registry.set_filter(disallowed=args.disallowed_tools.split(","))

    # Context manager
    context_mgr = ContextManager()

    # Mode dispatch
    if args.ndjson:
        _mode_ndjson(client, registry, store, context_mgr, mem, args)
    elif args.prompt:
        _mode_oneshot(client, registry, store, context_mgr, mem, args)
    else:
        _mode_repl(client, registry, store, context_mgr, mem, args)


# ── Mode: One-shot ───────────────────────────────────────────────

def _mode_oneshot(client, registry, store, context_mgr, mem, args):
    """Execute a single prompt and exit."""
    session_id = store.create_session(client.model, args.cwd)

    system_prompt = _build_prompt(client, registry, mem, store, args)
    messages = [{"role": "user", "content": args.prompt}]

    store.add_message(session_id, "user", args.prompt)

    # Optional planning
    if args.plan:
        planner = Planner(client)
        if planner.should_plan(args.prompt):
            steps = planner.plan(args.prompt, registry.names())
            plan_text = planner.format_plan_context(steps)
            if plan_text:
                _log(plan_text, True)
                messages[0]["content"] = f"{args.prompt}\n\n{plan_text}"

    def on_token(text, **kw):
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_tool_call(name, tool_args):
        if args.verbose:
            _log(f"[{name}] {json.dumps(tool_args)[:100]}", True)
        else:
            sys.stderr.write(f"\033[2m[{name}]\033[0m\n")
            sys.stderr.flush()

    def on_tool_result(name, result):
        if args.verbose:
            status = "ERROR" if result["is_error"] else "OK"
            _log(f"[{name} → {status}] {result['content'][:200]}", True)

    loop = AgentLoop(
        client=client,
        registry=registry,
        context_mgr=context_mgr,
        session_store=store,
        max_turns=args.max_turns,
        on_token=on_token,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        verbose=args.verbose,
    )

    result = loop.run(messages, system_prompt, session_id)

    sys.stdout.write("\n")
    if args.verbose:
        _log(f"({result['input_tokens']} in / {result['output_tokens']} out | {result['turns']} turns)", True)


# ── Mode: Interactive REPL ───────────────────────────────────────

def _mode_repl(client, registry, store, context_mgr, mem, args):
    """Interactive REPL mode."""
    session_id = None
    messages = []

    # Resume last session if requested
    if args.resume:
        last = store.get_latest_session()
        if last:
            session_id = last["id"]
            raw_msgs = store.get_messages(session_id)
            messages = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]
            _info(f"Resumed session {session_id[:8]} ({len(messages)} messages)")
        else:
            _info("No previous session to resume")

    # Signal handling
    def handle_sigint(sig, frame):
        print("\nUse /exit to quit, Ctrl+C to cancel current operation")

    signal.signal(signal.SIGINT, handle_sigint)

    _print_banner()

    while True:
        try:
            line = input("\033[1;36mji>\033[0m ").strip()
            if not line:
                continue

            # Handle commands
            if line.startswith("/"):
                should_continue = _handle_command(
                    line, client, registry, store, mem, args,
                    session_id, messages
                )
                if should_continue == "exit":
                    break
                if isinstance(should_continue, tuple):
                    session_id, messages = should_continue
                continue

            # Create session if needed
            if not session_id:
                session_id = store.create_session(client.model, args.cwd)

            # Add user message
            messages.append({"role": "user", "content": line})
            store.add_message(session_id, "user", line)

            # Build system prompt
            system_prompt = _build_prompt(client, registry, mem, store, args)

            # Optional planning
            effective_prompt = line
            if args.plan:
                planner = Planner(client)
                if planner.should_plan(line):
                    steps = planner.plan(line, registry.names())
                    plan_text = planner.format_plan_context(steps)
                    if plan_text:
                        print(f"\033[2m{plan_text}\033[0m")
                        effective_prompt = f"{line}\n\n{plan_text}"
                        messages[-1]["content"] = effective_prompt

            def on_token(text, **kw):
                sys.stdout.write(text)
                sys.stdout.flush()

            def on_tool_call(name, tool_args):
                sys.stderr.write(f"\033[2m[{name}]\033[0m ")
                sys.stderr.flush()

            def on_tool_result(name, result):
                if args.verbose:
                    status = "✗" if result["is_error"] else "✓"
                    sys.stderr.write(f"\033[2m{status}\033[0m\n")
                    sys.stderr.flush()

            loop = AgentLoop(
                client=client,
                registry=registry,
                context_mgr=context_mgr,
                session_store=store,
                max_turns=args.max_turns,
                on_token=on_token,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                verbose=args.verbose,
            )

            result = loop.run(messages, system_prompt, session_id)
            messages = result["messages"]
            print()

            if args.verbose:
                _log(f"({result['input_tokens']} in / {result['output_tokens']} out | {result['turns']} turns)", True)

        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            break

    _info("Goodbye!")


# ── Mode: NDJSON ─────────────────────────────────────────────────

def _mode_ndjson(client, registry, store, context_mgr, mem, args):
    """NDJSON protocol mode — reads JSON messages from stdin, writes to stdout."""
    for raw_line in sys.stdin:
        try:
            msg = json.loads(raw_line.strip())
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type", "")
        if msg_type != "message":
            continue

        content = msg.get("content", "")
        session_id = msg.get("session_id")

        if not session_id:
            session_id = store.create_session(client.model, args.cwd)

        # Get conversation history
        raw_msgs = store.get_messages(session_id)
        messages = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]
        messages.append({"role": "user", "content": content})

        system_prompt = _build_prompt(client, registry, mem, store, args)

        def on_token(text, **kw):
            print(json.dumps({"type": "token", "content": text}), flush=True)

        def on_tool_call(name, tool_args):
            print(json.dumps({"type": "tool_call", "name": name, "args": tool_args}), flush=True)

        def on_tool_result(name, result):
            print(json.dumps({"type": "tool_result", "name": name, "content": result["content"][:500]}), flush=True)

        loop = AgentLoop(
            client=client,
            registry=registry,
            context_mgr=context_mgr,
            session_store=store,
            max_turns=args.max_turns,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )

        result = loop.run(messages, system_prompt, session_id)
        print(json.dumps({
            "type": "done",
            "session_id": session_id,
            "turns": result["turns"],
            "tokens": result["input_tokens"] + result["output_tokens"],
        }), flush=True)


# ── REPL Commands ────────────────────────────────────────────────

def _handle_command(line, client, registry, store, mem, args, session_id, messages):
    """Handle /commands in REPL mode. Returns 'exit', (session_id, messages), or None."""
    parts = line.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit" or cmd == "/quit":
        return "exit"

    elif cmd == "/clear":
        _info("Session cleared")
        return (None, [])

    elif cmd == "/new":
        sid = store.create_session(client.model, args.cwd)
        _info(f"New session: {sid[:8]}")
        return (sid, [])

    elif cmd == "/resume":
        last = store.get_latest_session()
        if last:
            sid = last["id"]
            raw = store.get_messages(sid)
            msgs = [{"role": m["role"], "content": m["content"]} for m in raw]
            _info(f"Resumed session {sid[:8]} ({len(msgs)} messages)")
            return (sid, msgs)
        _info("No previous session")

    elif cmd == "/sessions":
        _cmd_list_sessions(store)

    elif cmd == "/session":
        if arg:
            # Switch to session by ID prefix
            sessions = store.list_sessions(100)
            match = [s for s in sessions if s["id"].startswith(arg)]
            if match:
                sid = match[0]["id"]
                raw = store.get_messages(sid)
                msgs = [{"role": m["role"], "content": m["content"]} for m in raw]
                _info(f"Switched to {sid[:8]} ({len(msgs)} messages)")
                return (sid, msgs)
            _info(f"No session matching '{arg}'")
        else:
            _info(f"Current session: {session_id[:8] if session_id else 'none'}")

    elif cmd == "/models":
        _cmd_list_models(client)

    elif cmd == "/model":
        if arg:
            new_model = resolve_model(arg)
            client.model = new_model
            _info(f"Switched to model: {new_model}")
        else:
            _info(f"Current model: {client.model}")

    elif cmd == "/memory":
        if arg.startswith("set "):
            # /memory set key=value
            kv = arg[4:].split("=", 1)
            if len(kv) == 2:
                mem.store(kv[0].strip(), kv[1].strip())
                _info(f"Stored: {kv[0].strip()}")
            else:
                _info("Usage: /memory set key=value")
        elif arg.startswith("get "):
            val = mem.recall(arg[4:].strip())
            _info(f"{val}" if val else "Not found")
        elif arg.startswith("search "):
            results = mem.search(arg[7:].strip())
            for r in results:
                _info(f"[{r['category']}] {r['key']}: {r['value']}")
        elif arg.startswith("forget "):
            mem.forget(arg[7:].strip())
            _info("Forgotten")
        else:
            _cmd_show_memory(mem)

    elif cmd == "/tools":
        for d in registry.definitions():
            print(f"  {d['name']}: {d['description'][:60]}...")

    elif cmd == "/compact":
        if messages:
            before = len(messages)
            ctx = ContextManager()
            messages = ctx.prune(messages, keep_first=1, keep_last=4)
            _info(f"Compacted: {before} → {len(messages)} messages")
            return (session_id, messages)
        _info("Nothing to compact")

    elif cmd == "/help":
        _print_help()

    else:
        _info(f"Unknown command: {cmd}. Type /help for commands.")

    return None


# ── Prompt Building ──────────────────────────────────────────────

def _build_prompt(client, registry, mem, store, args) -> str:
    """Build the full system prompt."""
    memory_block = mem.get_context_block(max_chars=1500)
    errors = store.get_common_errors(limit=5)
    error_context = format_error_context(errors)

    return build_system_prompt(
        cwd=args.cwd,
        model=client.model,
        tool_definitions=registry.definitions(),
        memory_block=memory_block,
        error_context=error_context,
    )


# ── Info Commands ────────────────────────────────────────────────

def _cmd_list_models(client):
    models = client.list_models()
    print(f"Available models ({len(models)}):")
    for m in models:
        alias = ""
        for a, full in MODEL_ALIASES.items():
            if full == m:
                alias = f" (alias: {a})"
                break
        print(f"  {m}{alias}")


def _cmd_list_sessions(store):
    sessions = store.list_sessions(20)
    if not sessions:
        print("No sessions found")
        return
    print(f"Recent sessions ({len(sessions)}):")
    for s in sessions:
        dt = datetime.fromtimestamp(s["updated_at"], timezone.utc).strftime("%Y-%m-%d %H:%M")
        summary = (s.get("summary") or "")[:50]
        print(f"  {s['id'][:8]}  {dt}  {s['model']}  {summary}")


def _cmd_show_memory(mem):
    all_mem = mem.list_all()
    if not all_mem:
        print("No persistent memories stored")
        return
    for category, items in all_mem.items():
        print(f"\n[{category}]")
        for key, value in items.items():
            print(f"  {key}: {value}")


# ── UI Helpers ───────────────────────────────────────────────────
# ═══════╗
def _print_banner():
    print(f"\033[1;36m╔════════════════════════════════════════════════╗\033[0m")
    print(f"\033[1;36m║  Just Intelligent CLI — Agentic AI CLI v{__version__}  ║\033[0m")
    print(f"\033[1;36m╚════════════════════════════════════════════════╝\033[0m")
    print(f"\033[2mType /help for commands, /exit to quit\033[0m")
    print()


def _print_help():
    print("""
Just Intelligent CLI Commands:
  /help           Show this help
  /exit           Exit Just Intelligent CLI
  /clear          Clear current session
  /new            Start a new session
  /resume         Resume last session
  /sessions       List recent sessions
  /session [id]   Show or switch session
  /model [name]   Show or switch model
  /models         List available models
  /tools          List available tools
  /memory         Show persistent memory
  /memory set k=v Store a memory fact
  /memory get k   Recall a memory fact
  /memory search q Search memory
  /memory forget k Forget a memory fact
  /compact        Prune conversation context
""")


def _info(msg):
    print(f"\033[2m{msg}\033[0m")


def _err(msg):
    sys.stderr.write(f"\033[31m{msg}\033[0m\n")
    sys.stderr.flush()


def _log(msg, verbose=True):
    if verbose:
        sys.stderr.write(f"\033[2m[jicli] {msg}\033[0m\n")
        sys.stderr.flush()
