#!/usr/bin/env python3
"""Run staged model selection for LR and XGB; write models/ artifacts."""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from model_selection_utils import (  # noqa: E402
    MODEL_FAMILIES,
    MODELS_DIR,
    load_splits,
    run_staged_selection,
    save_cleartext_baselines,
    save_selected_models,
)


def main() -> None:
    df_train, df_test, X_train, y_train, X_test, y_test = load_splits()
    sample_row = df_test.sample(1, random_state=8047).drop(columns=["prognosis", "prognosis_encoded"]).values

    selections = {}
    for key in ("logistic_regression", "xgboost"):
        family = MODEL_FAMILIES[key]
        _, selection = run_staged_selection(
            family, X_train, y_train, X_test, y_test, sample_row, models_dir=MODELS_DIR
        )
        selections[key] = selection
        print(f"Selected {key}: {selection['params']}")
        print(f"  Rationale: {selection['rationale']}")

    path = save_selected_models(selections)
    print(f"Wrote {path}")
    baseline_path = save_cleartext_baselines(X_train, y_train, X_test, y_test, selections)
    print(f"Wrote {baseline_path}")


if __name__ == "__main__":
    main()
