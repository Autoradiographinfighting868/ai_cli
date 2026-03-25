#!/usr/bin/env python3
"""Just Intelligent CLI — Agentic CLI for local Ollama models.

Usage:
  python jicli.py                       Interactive REPL
  python jicli.py -p "list files here"  One-shot command
  python jicli.py --plan -p "build X"   Plan then execute
  python jicli.py --resume              Resume last session
  python jicli.py --list-models         List available models
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jicli.cli import main

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as err:
        sys.stderr.write(f"\033[31mFatal: {err}\033[0m\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
