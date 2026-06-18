"""Generate deployment files (legacy wrapper — prefer scripts/train_models.py)."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script = Path(__file__).parent / "scripts" / "train_models.py"
    subprocess.run([sys.executable, str(script), "--model", "logistic_regression"], check=True)
