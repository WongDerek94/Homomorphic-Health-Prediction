#!/usr/bin/env python3
"""Train and deploy FHE bundles for LR and XGB from models/selected_models.json."""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from model_selection_utils import MODEL_FAMILIES, load_selected_models  # noqa: E402
from utils import DEPLOYMENT_ROOT, TARGET_COLUMNS, get_deployment_dir  # noqa: E402

from concrete.ml.deployment import FHEModelDev  # noqa: E402

TRAINING = REPO_ROOT / "data" / "Training_preprocessed.csv"


def _build_and_save(model_type: str, params: dict) -> Path:
    family = MODEL_FAMILIES[model_type]
    df = pd.read_csv(TRAINING)
    X_train = df.drop(columns=TARGET_COLUMNS)
    y_train = df[TARGET_COLUMNS[0]].values

    merged = {**family.extra_fit_params, **params}
    model = family.classifier_cls(**merged)
    model.fit(X_train, y_train)
    model.compile(X_train)

    out_dir = get_deployment_dir(model_type)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    dev = FHEModelDev(out_dir, model)
    dev.save(via_mlir=True)
    print(f"Saved {model_type} bundle to {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "xgboost", "both"],
        default="both",
    )
    args = parser.parse_args()

    selections = load_selected_models()
    targets = ["logistic_regression", "xgboost"] if args.model == "both" else [args.model]

    for name in targets:
        sel = selections[name]
        _build_and_save(name, sel["params"])

    # Preserve runtime dirs alongside model bundles
    for sub in (".fhe_keys", "client_dir", "server_dir"):
        p = DEPLOYMENT_ROOT / sub
        p.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
