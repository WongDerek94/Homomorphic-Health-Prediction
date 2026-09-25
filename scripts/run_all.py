#!/usr/bin/env python3
"""Single-command reproducibility pipeline for SecureMed.

Default: verify preprocessed data, train both FHE bundles from committed
``models/selected_models.json``, and run the full evidence suite.

Usage (from repository root):

    python scripts/run_all.py
    python scripts/run_all.py --quick
    python scripts/run_all.py --run-model-selection
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run(cmd: list[str], step: str) -> None:
    print(f"\n=== {step} ===")
    print(">", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: N=5, fhe=simulate for evidence (fast)",
    )
    parser.add_argument(
        "--run-model-selection",
        action="store_true",
        help="Run full staged model selection (hours; default uses committed models/)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip train_models.py (use existing deployment_files/)",
    )
    parser.add_argument(
        "--skip-evidence",
        action="store_true",
        help="Skip generate_evidence.py",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Regenerate preprocessed CSVs instead of verify-only",
    )
    args = parser.parse_args()

    scripts = REPO_ROOT / "scripts"

    if args.regenerate_data:
        _run([PYTHON, str(scripts / "preprocess_data.py")], "Preprocess data")
    else:
        _run([PYTHON, str(scripts / "preprocess_data.py"), "--verify-only"], "Verify preprocessed data")

    if args.run_model_selection:
        _run([PYTHON, str(scripts / "run_model_selection.py")], "Model selection (long-running)")

    selected = REPO_ROOT / "models" / "selected_models.json"
    if not selected.is_file() and not args.run_model_selection:
        sys.exit(f"Missing {selected}; run with --run-model-selection or commit models/ artifacts.")

    if not args.skip_train:
        _run([PYTHON, str(scripts / "train_models.py"), "--model", "both"], "Train and deploy FHE bundles")

    if not args.skip_evidence:
        evidence_cmd = [PYTHON, str(scripts / "generate_evidence.py"), "--model", "both"]
        if args.quick:
            evidence_cmd.append("--quick")
        _run(evidence_cmd, "Generate evaluation evidence")

    print("\n=== Done ===")
    print(f"Evidence summary: {REPO_ROOT / 'metrics' / 'evidence_summary.json'}")


if __name__ == "__main__":
    main()
