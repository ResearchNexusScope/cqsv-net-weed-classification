"""Single central verifier for the rebuild swarm. Mother runs this only.

Runs deterministic, serial checks. No worker may invoke this or any other build.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(label: str, args: list[str]) -> int:
    print(f"\n=== {label} ===")
    print("+", " ".join(args))
    proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    print(f"[{label}] exit={proc.returncode}")
    return proc.returncode


def main() -> int:
    failures = []
    failures.append(("compile", run("compile", [PY, "-m", "compileall", "-q", "experiments"])))
    failures.append(("metrics_selftest", run("metrics_selftest", [PY, "-m", "experiments.metrics"])))

    tests = ROOT / "tests"
    if tests.exists():
        failures.append(("pytest", run("pytest", [PY, "-m", "pytest", "-q", "tests"])))
    else:
        print("\n=== pytest ===\n(no tests/ directory yet; skipped)")

    bad = [name for name, code in failures if code != 0]
    print("\n=== SUMMARY ===")
    print("FAILED: " + ", ".join(bad) if bad else "ALL CHECKS PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
