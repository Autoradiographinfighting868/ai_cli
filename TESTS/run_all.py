#!/usr/bin/env python3
"""Run all Just Intelligent CLI tests."""

import os
import sys
import importlib
import importlib.util
import traceback

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def run_all():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = sorted(f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py"))

    total_pass = 0
    total_fail = 0
    failures = []

    for fname in test_files:
        module_name = fname[:-3]
        print(f"\n{'='*60}")
        print(f"Running {fname}")
        print(f"{'='*60}")

        try:
            # Import and run the module
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(test_dir, fname))
            mod = importlib.util.module_from_spec(spec)

            # Find and run all test_ functions
            spec.loader.exec_module(mod)

            test_funcs = [name for name in dir(mod) if name.startswith("test_") and callable(getattr(mod, name))]

            for func_name in sorted(test_funcs):
                try:
                    getattr(mod, func_name)()
                    total_pass += 1
                except Exception as e:
                    total_fail += 1
                    failures.append((fname, func_name, str(e)))
                    print(f"  FAIL: {func_name} — {e}")
                    traceback.print_exc()

        except Exception as e:
            total_fail += 1
            failures.append((fname, "__module__", str(e)))
            print(f"  MODULE ERROR: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {total_pass} passed, {total_fail} failed")
    print(f"{'='*60}")

    if failures:
        print("\nFailures:")
        for fname, func, err in failures:
            print(f"  {fname}::{func} — {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_all()
