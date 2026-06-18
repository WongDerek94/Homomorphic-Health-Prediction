import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import numpy
import pandas

# Max Input to be displayed on the HuggingFace space brower using Gradio
INPUT_BROWSER_LIMIT = 380

SERVER_URL = "http://localhost:8000/"

CURRENT_DIR = Path(__file__).parent
DEPLOYMENT_ROOT = CURRENT_DIR / "deployment_files"
MODEL_REGISTRY = {
    "logistic_regression": DEPLOYMENT_ROOT / "logistic_regression",
    "xgboost": DEPLOYMENT_ROOT / "xgboost",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression (baseline)",
    "xgboost": "XGBoost (tree ensemble)",
}

# Backward compatibility: default deployment dir for legacy single-model paths
DEPLOYMENT_DIR = MODEL_REGISTRY["logistic_regression"]

KEYS_DIR = DEPLOYMENT_ROOT / ".fhe_keys"
CLIENT_DIR = DEPLOYMENT_ROOT / "client_dir"
SERVER_DIR = DEPLOYMENT_ROOT / "server_dir"

ALL_DIRS = [KEYS_DIR, CLIENT_DIR, SERVER_DIR]

TARGET_COLUMNS = ["prognosis_encoded", "prognosis"]

TRAINING_FILENAME = "./data/Training_preprocessed.csv"
TESTING_FILENAME = "./data/Testing_preprocessed.csv"


def get_deployment_dir(model_type: str) -> Path:
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type: {model_type}")
    return MODEL_REGISTRY[model_type]


def pretty_print(
    inputs, case_conversion=str.title, which_replace: str = "_", to_what: str = " ", delimiter=None
):
    pretty_list = []
    for item in inputs:
        if isinstance(item, list):
            pretty_list.extend(item)
        else:
            pretty_list.append(item)

    pretty_list = sorted(list(set(pretty_list)))
    pretty_list = [item.replace(which_replace, to_what) for item in pretty_list]
    pretty_list = [case_conversion(item) for item in pretty_list]
    if delimiter:
        pretty_list = f"{delimiter.join(pretty_list)}."
    return pretty_list


def clean_directory() -> None:
    print("Cleaning...\n")
    for target_dir in ALL_DIRS:
        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            shutil.rmtree(target_dir)
        target_dir.mkdir(exist_ok=True, parents=True)


def get_disease_name(encoded_prediction: int, file_name: str = TRAINING_FILENAME) -> str:
    df = pandas.read_csv(file_name, usecols=TARGET_COLUMNS).drop_duplicates()
    disease_name, _ = df[df[TARGET_COLUMNS[0]] == encoded_prediction].values.flatten()
    return disease_name


def load_data() -> Union[Tuple[pandas.DataFrame, numpy.ndarray], List]:
    df_train = pandas.read_csv(TRAINING_FILENAME)
    df_test = pandas.read_csv(TESTING_FILENAME)

    y_train = df_train[TARGET_COLUMNS[0]]
    X_train = df_train.drop(columns=TARGET_COLUMNS, axis=1, errors="ignore")

    y_test = df_test[TARGET_COLUMNS[0]]
    X_test = df_test.drop(columns=TARGET_COLUMNS, axis=1, errors="ignore")

    return (
        (X_train, X_test),
        (y_train, y_test),
        X_train.columns.to_list(),
        df_train[TARGET_COLUMNS[1]].unique().tolist(),
    )
