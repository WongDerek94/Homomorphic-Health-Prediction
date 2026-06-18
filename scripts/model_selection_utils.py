"""Shared helpers for SecureMed Phase 2 model selection.

Staged pipeline: GridSearchCV -> FHE simulate on all configs -> FHE execute on top 5.
"""

from __future__ import annotations

import json
import pickle
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Type

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from concrete.ml.deployment import FHEModelClient, FHEModelDev
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.sklearn import XGBClassifier as ConcreteXGBClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

TARGET_COLUMNS = ["prognosis_encoded", "prognosis"]
TRAINING_FILENAME = REPO_ROOT / "data" / "Training_preprocessed.csv"
TESTING_FILENAME = REPO_ROOT / "data" / "Testing_preprocessed.csv"

XGB_GRID = {
    "n_bits": [3, 4, 5, 6, 7, 8],
    "max_depth": [1, 2, 3],
    "n_estimators": [2, 3, 5, 10, 20, 30],
}

LR_GRID = {
    "C": [0.5, 0.9, 1.0],
    "n_bits": [7, 8, 10, 13],
    "solver": ["sag", "newton-cg"],
}

ACCURACY_DELTA_PP = 5.0
CIPHERTEXT_MAX_BYTES = 1_048_576
TOP_K_FINALISTS = 5


@dataclass
class ModelFamily:
    name: str
    label: str
    classifier_cls: Type
    param_grid: Dict[str, List[Any]]
    extra_fit_params: Dict[str, Any]


MODEL_FAMILIES = {
    "logistic_regression": ModelFamily(
        name="logistic_regression",
        label="ConcreteLogisticRegression",
        classifier_cls=ConcreteLogisticRegression,
        param_grid=LR_GRID,
        extra_fit_params={"multi_class": "auto"},
    ),
    "xgboost": ModelFamily(
        name="xgboost",
        label="ConcreteXGBClassifier",
        classifier_cls=ConcreteXGBClassifier,
        param_grid=XGB_GRID,
        extra_fit_params={"n_jobs": 1},
    ),
}


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df_train = pd.read_csv(TRAINING_FILENAME)
    df_test = pd.read_csv(TESTING_FILENAME)
    X_train = df_train.drop(columns=TARGET_COLUMNS).values
    y_train = df_train[TARGET_COLUMNS[0]].values
    X_test = df_test.drop(columns=TARGET_COLUMNS).values
    y_test = df_test[TARGET_COLUMNS[0]].values
    return df_train, df_test, X_train, y_train, X_test, y_test


def run_grid_search(
    family: ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 4,
    verbose: int = 1,
) -> GridSearchCV:
    clf = family.classifier_cls(**family.extra_fit_params)
    search = GridSearchCV(
        clf,
        family.param_grid,
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
        scoring="accuracy",
    )
    search.fit(X_train, y_train)
    return search


def grid_results_dataframe(search: GridSearchCV, param_keys: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(search.cv_results_)
    df.columns = df.columns.str.replace("param_", "")
    cols = param_keys + ["mean_test_score", "std_test_score", "rank_test_score"]
    return df[cols].sort_values("mean_test_score", ascending=False).reset_index(drop=True)


def _build_model(family: ModelFamily, params: Dict[str, Any]):
    merged = {**family.extra_fit_params, **params}
    model = family.classifier_cls(**merged)
    return model


def measure_fhe_simulate(
    family: ModelFamily,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample: np.ndarray,
) -> float:
    model = _build_model(family, params)
    model.fit(X_train, y_train)
    model.compile(X_train)
    start = time.perf_counter()
    model.predict(sample, fhe="simulate")
    return time.perf_counter() - start


def measure_fhe_execute_and_ciphertext(
    family: ModelFamily,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample: np.ndarray,
    deployment_dir: Path,
) -> Dict[str, Any]:
    model = _build_model(family, params)
    model.fit(X_train, y_train)
    model.compile(X_train)

    clear_pred = model.predict(X_test, fhe="disable")
    clear_acc = float((clear_pred == y_test).mean())

    start = time.perf_counter()
    fhe_pred = model.predict(X_test, fhe="execute")
    fhe_time = time.perf_counter() - start
    fhe_acc = float((fhe_pred == y_test).mean())
    delta_pp = abs(clear_acc - fhe_acc) * 100

    deployment_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = deployment_dir / "bundle"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    dev = FHEModelDev(bundle_dir, model)
    dev.save(via_mlir=True)

    key_dir = deployment_dir / ".tmp_keys"
    if key_dir.exists():
        shutil.rmtree(key_dir)
    client = FHEModelClient(path_dir=bundle_dir, key_dir=key_dir)
    client.load()
    client.generate_private_and_evaluation_keys()
    ciphertext = client.quantize_encrypt_serialize(sample)
    ciphertext_bytes = len(ciphertext)

    return {
        "cleartext_accuracy": clear_acc,
        "fhe_accuracy": fhe_acc,
        "accuracy_delta_pp": delta_pp,
        "fhe_execute_time_s": fhe_time,
        "ciphertext_bytes": ciphertext_bytes,
        "accuracy_delta_pass": delta_pp <= ACCURACY_DELTA_PP,
        "ciphertext_pass": ciphertext_bytes <= CIPHERTEXT_MAX_BYTES,
    }


def pick_top_finalists(df: pd.DataFrame, param_keys: List[str], k: int = TOP_K_FINALISTS) -> pd.DataFrame:
    """Rank by accuracy then simulate latency; return top k unique configs."""
    ranked = df.sort_values(
        ["mean_test_score", "fhe_simulate_time_s"],
        ascending=[False, True],
    )
    return ranked.head(k).reset_index(drop=True)


def select_best_config(finalists: pd.DataFrame, param_keys: List[str]) -> Tuple[Dict[str, Any], str]:
    """Prefer configs passing both thresholds; else best accuracy with lowest n_bits."""
    passing = finalists[
        finalists["accuracy_delta_pass"] & finalists["ciphertext_pass"]
    ]
    if not passing.empty:
        sort_cols = ["fhe_execute_time_s"]
        if "n_bits" in passing.columns:
            sort_cols = ["n_bits"] + sort_cols
        row = passing.sort_values(sort_cols, ascending=[True] * len(sort_cols)).iloc[0]
        rationale = "Passes accuracy delta and ciphertext thresholds; lowest n_bits among passing configs."
    else:
        row = finalists.sort_values(
            ["accuracy_delta_pp", "ciphertext_bytes", "fhe_execute_time_s"],
            ascending=[True, True, True],
        ).iloc[0]
        rationale = (
            "No config passed all thresholds; selected best compromise on accuracy delta, "
            "ciphertext size, and FHE execute time."
        )
    params = {k: row[k] for k in param_keys}
    # sklearn may store floats as floats; cast n_bits etc.
    if "n_bits" in params:
        params["n_bits"] = int(params["n_bits"])
    if "max_depth" in params:
        params["max_depth"] = int(params["max_depth"])
    if "n_estimators" in params:
        params["n_estimators"] = int(params["n_estimators"])
    return params, rationale


def run_staged_selection(
    family: ModelFamily,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample: np.ndarray,
    models_dir: Path = MODELS_DIR,
    top_k: int = TOP_K_FINALISTS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    param_keys = list(family.param_grid.keys())
    print(f"\n=== {family.label}: GridSearchCV ===")
    search = run_grid_search(family, X_train, y_train)
    print(f"Best CV params: {search.best_params_}, score={search.best_score_:.4f}")

    results = grid_results_dataframe(search, param_keys)
    print(f"Running FHE simulate on {len(results)} configs ...")
    simulate_times = []
    for _, row in results.iterrows():
        params = {k: row[k] for k in param_keys}
        for int_key in ("n_bits", "max_depth", "n_estimators"):
            if int_key in params:
                params[int_key] = int(params[int_key])
        t = measure_fhe_simulate(family, params, X_train, y_train, sample)
        simulate_times.append(t)
    results["fhe_simulate_time_s"] = simulate_times

    finalists = pick_top_finalists(results, param_keys, k=top_k)
    print(f"Running FHE execute on top {len(finalists)} finalists ...")
    execute_metrics = []
    for _, row in finalists.iterrows():
        params = {k: row[k] for k in param_keys}
        for int_key in ("n_bits", "max_depth", "n_estimators"):
            if int_key in params:
                params[int_key] = int(params[int_key])
        tmp_dir = models_dir / f"_tmp_{family.name}"
        metrics = measure_fhe_execute_and_ciphertext(
            family, params, X_train, y_train, X_test, y_test, sample, tmp_dir
        )
        execute_metrics.append(metrics)
    for col in execute_metrics[0]:
        finalists[col] = [m[col] for m in execute_metrics]

    selected_params, rationale = select_best_config(finalists, param_keys)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_path = models_dir / f"grid_search_{family.name}_results.csv"
    results.to_csv(results_path, index=False)
    results.to_json(models_dir / f"grid_search_{family.name}_results.json", orient="records", indent=2)

    selection = {
        "model_type": family.name,
        "classifier": family.label,
        "params": selected_params,
        "rationale": rationale,
        "cv_best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
    }
    return results, selection


def save_selected_models(selections: Dict[str, Dict[str, Any]], models_dir: Path = MODELS_DIR) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": selections,
    }
    path = models_dir / "selected_models.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_cleartext_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selections: Dict[str, Dict[str, Any]],
    models_dir: Path = MODELS_DIR,
) -> Path:
    baselines = {}
    for name, sel in selections.items():
        family = MODEL_FAMILIES[name]
        params = sel["params"]
        model = _build_model(family, params)
        model.fit(X_train, y_train)
        test_acc = float((model.predict(X_test, fhe="disable") == y_test).mean())
        baselines[name] = {"params": params, "test_accuracy": test_acc, "classifier": family.label}
    path = models_dir / "cleartext_baselines.pkl"
    with path.open("wb") as f:
        pickle.dump(baselines, f)
    return path


def load_selected_models(models_dir: Path = MODELS_DIR) -> Dict[str, Dict[str, Any]]:
    path = models_dir / "selected_models.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run model selection first.")
    data = json.loads(path.read_text())
    return data["models"]
