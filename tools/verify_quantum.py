"""Quantum verifier: runs QML tests under the exact-pin Python 3.11 environment.

Mother runs this in addition to tools/verify_central.py. The central verifier uses
Python 3.14 (no PennyLane) and checks clean skips; this one executes the real
quantum paths with PennyLane 0.40.0 / Lightning 0.40.0 / autoray 0.7.0.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.getenv("QML_PYTHON") or shutil.which("python") or "python"

EXPECTED = {"pennylane": "0.40.0", "autoray": "0.7.0"}


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
    probe = run(
        "environment",
        [PY, "-c",
         "import pennylane,autoray,numpy;"
         "print('pennylane',pennylane.__version__);"
         "print('autoray',autoray.__version__);"
         "print('numpy',numpy.__version__)"],
    )
    if probe != 0:
        print("QUANTUM VERIFIER FAILED: exact-pin environment unavailable")
        return 1

    test_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("test_qmlpipe_*.py"))
    if not test_files:
        print("QUANTUM VERIFIER FAILED: no qmlpipe tests found")
        return 1
    code = run("qmlpipe-pytest", [PY, "-m", "pytest", "-q", *test_files])
    print("\n=== SUMMARY ===")
    print("QUANTUM VERIFIER PASSED" if code == 0 else "QUANTUM VERIFIER FAILED")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
