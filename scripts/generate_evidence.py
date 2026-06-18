"""SecureMed evidence-generation suite (multi-model)."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from model_selection_utils import MODEL_FAMILIES, load_selected_models  # noqa: E402
from utils import (  # noqa: E402
    KEYS_DIR,
    SERVER_URL,
    TARGET_COLUMNS,
    TESTING_FILENAME,
    TRAINING_FILENAME,
    clean_directory,
    get_deployment_dir,
)

from concrete.ml.deployment import FHEModelClient  # noqa: E402

METRICS_DIR = REPO_ROOT / "metrics"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data"

THRESHOLDS = {
    "accuracy_delta_pp": 5.0,
    "fhe_latency_median_s": 5.0,
    "e2e_latency_median_s": 5.0,
    "keygen_time_s": 30.0,
    "ciphertext_size_bytes": 1_048_576,
}

MIN_SYMPTOMS, MAX_SYMPTOMS = 5, 17


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["generated_at"] = timestamp()
    path.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def append_log(path: Path, lines, mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        for line in lines:
            f.write(f"{timestamp()} {line}\n")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def load_splits():
    df_train = pd.read_csv(TRAINING_FILENAME)
    df_test = pd.read_csv(TESTING_FILENAME)
    X_train = df_train.drop(columns=TARGET_COLUMNS)
    y_train = df_train[TARGET_COLUMNS[0]].values
    X_test = df_test.drop(columns=TARGET_COLUMNS)
    y_test = df_test[TARGET_COLUMNS[0]].values
    return df_train, df_test, X_train, y_train, X_test, y_test


def random_symptom_vector(n_features: int, rng: np.random.Generator) -> np.ndarray:
    n_symptoms = rng.integers(MIN_SYMPTOMS, MAX_SYMPTOMS + 1)
    vector = np.zeros((1, n_features), dtype=np.int64)
    vector[0, rng.choice(n_features, size=n_symptoms, replace=False)] = 1
    return vector


def verify_ingestion(df_train, df_test) -> None:
    print("\n[ingestion] Data verification")
    lines = []
    for name, path, df in [
        ("train", Path(TRAINING_FILENAME), df_train),
        ("test", Path(TESTING_FILENAME), df_test),
    ]:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{name}: file={path.name} rows={len(df)} cols={len(df.columns)} sha256={digest}")
    out = DATA_DIR / "testset_checksum.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def verify_input_vector(X_train, model_suffix: str = "") -> int:
    n_features = X_train.shape[1]
    values = pd.unique(X_train.values.ravel())
    is_binary = set(np.unique(values)).issubset({0, 0.0, 1, 1.0})
    log_path = ARTIFACTS_DIR / f"input_vector_size{model_suffix}.log"
    append_log(
        log_path,
        [
            f"feature_count={n_features} binary_features={is_binary} "
            f"proposal_target=128 match={n_features == 128}"
        ],
        mode="w" if model_suffix == "" else "a",
    )
    return n_features


def measure_keygen(model_type: str) -> float:
    deployment_dir = get_deployment_dir(model_type)
    user_id = f"evidence_keygen_{model_type}"
    client = FHEModelClient(path_dir=deployment_dir, key_dir=KEYS_DIR / user_id)
    client.load()
    start = time.perf_counter()
    client.generate_private_and_evaluation_keys()
    elapsed = time.perf_counter() - start
    eval_key_size = len(client.get_serialized_evaluation_keys())
    append_log(
        ARTIFACTS_DIR / f"keygen_time_{model_type}.log",
        [
            f"model={model_type} keygen_seconds={elapsed:.3f} evaluation_key_bytes={eval_key_size} "
            f"threshold_s={THRESHOLDS['keygen_time_s']} pass={elapsed <= THRESHOLDS['keygen_time_s']}"
        ],
    )
    return elapsed


def measure_ciphertext_sizes(model_type: str, n_features: int, n_samples: int, rng) -> dict:
    deployment_dir = get_deployment_dir(model_type)
    client = FHEModelClient(
        path_dir=deployment_dir, key_dir=KEYS_DIR / f"evidence_keygen_{model_type}"
    )
    client.load()
    sizes = []
    lines = []
    for i in range(n_samples):
        vector = random_symptom_vector(n_features, rng)
        ciphertext = client.quantize_encrypt_serialize(vector)
        sizes.append(len(ciphertext))
        lines.append(
            f"model={model_type} sample={i} symptoms={int(vector.sum())} "
            f"ciphertext_bytes={len(ciphertext)} pass={len(ciphertext) <= THRESHOLDS['ciphertext_size_bytes']}"
        )
    lines.append(
        f"model={model_type} summary n={n_samples} min={min(sizes)} max={max(sizes)} "
        f"median={statistics.median(sizes):.0f} threshold={THRESHOLDS['ciphertext_size_bytes']}"
    )
    append_log(ARTIFACTS_DIR / f"ciphertext_sizes_{model_type}.log", lines)
    return {"n": n_samples, "min": min(sizes), "max": max(sizes), "median": statistics.median(sizes)}


def train_and_compare_accuracy(
    model_type: str, X_train, y_train, X_test, y_test, params: dict, fhe_mode: str
) -> tuple:
    family = MODEL_FAMILIES[model_type]
    merged = {**family.extra_fit_params, **params}
    model = family.classifier_cls(**merged)
    model.fit(X_train, y_train)
    model.compile(X_train)

    clear_pred = model.predict(X_test, fhe="disable")
    clear_acc = float((clear_pred == y_test).mean())
    save_json(
        METRICS_DIR / f"cleartext_metrics_{model_type}.json",
        {
            "model": family.label,
            "model_type": model_type,
            "params": params,
            "test_accuracy": clear_acc,
            "n_test_samples": int(len(y_test)),
        },
    )

    fhe_pred = model.predict(X_test, fhe=fhe_mode)
    fhe_acc = float((fhe_pred == y_test).mean())
    delta_pp = abs(clear_acc - fhe_acc) * 100
    result = {
        "model": family.label,
        "model_type": model_type,
        "params": params,
        "fhe_mode": fhe_mode,
        "cleartext_accuracy": clear_acc,
        "fhe_accuracy": fhe_acc,
        "accuracy_delta_pp": delta_pp,
        "threshold_pp": THRESHOLDS["accuracy_delta_pp"],
        "pass": delta_pp <= THRESHOLDS["accuracy_delta_pp"],
        "n_test_samples": int(len(y_test)),
    }
    save_json(METRICS_DIR / f"accuracy_comparison_{model_type}.json", result)
    return result, model


def measure_fhe_latency(model, n_features: int, n_runs: int, rng, model_type: str) -> dict:
    latencies = []
    for i in range(n_runs):
        vector = random_symptom_vector(n_features, rng)
        start = time.perf_counter()
        model.predict(vector, fhe="execute")
        latencies.append(time.perf_counter() - start)
        if (i + 1) % 20 == 0:
            print(f"  [{model_type}] {i + 1}/{n_runs} median={statistics.median(latencies):.3f}s")
    result = {
        "model_type": model_type,
        "n_runs": n_runs,
        "median_s": statistics.median(latencies),
        "mean_s": statistics.fmean(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "p95_s": float(np.percentile(latencies, 95)),
        "threshold_s": THRESHOLDS["fhe_latency_median_s"],
        "pass": statistics.median(latencies) <= THRESHOLDS["fhe_latency_median_s"],
        "all_latencies_s": [round(v, 4) for v in latencies],
    }
    save_json(METRICS_DIR / f"latency_stats_{model_type}.json", result)
    return result


def measure_e2e_latency(model_type: str, n_features: int, n_runs: int, rng) -> dict:
    deployment_dir = get_deployment_dir(model_type)
    env = os.environ.copy()
    env["SECUREMED_GET_OUTPUT_DELAY_S"] = "0"
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    try:
        for _ in range(30):
            try:
                requests.get(SERVER_URL, timeout=1)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            raise RuntimeError("FastAPI server did not start within 30 s")

        user_id = f"evidence_e2e_{model_type}"
        client = FHEModelClient(path_dir=deployment_dir, key_dir=KEYS_DIR / user_id)
        client.load()
        client.generate_private_and_evaluation_keys()
        evaluation_key = client.get_serialized_evaluation_keys()

        latencies = []
        for i in range(n_runs):
            vector = random_symptom_vector(n_features, rng)
            start = time.perf_counter()
            ciphertext = client.quantize_encrypt_serialize(vector)
            requests.post(
                SERVER_URL + "send_input",
                data={"user_id": user_id, "model_type": model_type, "input": "evidence"},
                files=[("files", ciphertext), ("files", evaluation_key)],
                timeout=600,
            ).raise_for_status()
            requests.post(
                SERVER_URL + "run_fhe",
                data={"user_id": user_id, "model_type": model_type},
                timeout=600,
            ).raise_for_status()
            response = requests.post(
                SERVER_URL + "get_output",
                data={"user_id": user_id, "model_type": model_type},
                timeout=600,
            )
            response.raise_for_status()
            client.deserialize_decrypt_dequantize(response.content)
            latencies.append(time.perf_counter() - start)
            if (i + 1) % 20 == 0:
                print(f"  [{model_type} e2e] {i + 1}/{n_runs} median={statistics.median(latencies):.3f}s")

        result = {
            "model_type": model_type,
            "n_runs": n_runs,
            "median_s": statistics.median(latencies),
            "mean_s": statistics.fmean(latencies),
            "min_s": min(latencies),
            "max_s": max(latencies),
            "p95_s": float(np.percentile(latencies, 95)),
            "threshold_s": THRESHOLDS["e2e_latency_median_s"],
            "pass": statistics.median(latencies) <= THRESHOLDS["e2e_latency_median_s"],
            "get_output_delay_s": 0,
            "all_latencies_s": [round(v, 4) for v in latencies],
        }
        save_json(METRICS_DIR / f"e2e_latency_{model_type}.json", result)
        return result
    finally:
        server.terminate()
        server.wait(timeout=10)


def run_model_evidence(
    model_type: str,
    params: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    n_runs: int,
    fhe_mode: str,
    rng,
    skip_e2e: bool,
) -> dict:
    print(f"\n========== Evidence for {model_type} ==========")
    n_features = X_train.shape[1]
    keygen_s = measure_keygen(model_type)
    ciphertext_stats = measure_ciphertext_sizes(model_type, n_features, n_runs, rng)
    accuracy, model = train_and_compare_accuracy(
        model_type, X_train, y_train, X_test, y_test, params, fhe_mode
    )
    latency = measure_fhe_latency(model, n_features, n_runs, rng, model_type)
    e2e = None if skip_e2e else measure_e2e_latency(model_type, n_features, n_runs, rng)
    return {
        "model_type": model_type,
        "params": params,
        "accuracy_delta_pp": accuracy["accuracy_delta_pp"],
        "accuracy_pass": accuracy["pass"],
        "fhe_latency_median_s": latency["median_s"],
        "fhe_latency_pass": latency["pass"],
        "e2e_latency_median_s": e2e["median_s"] if e2e else None,
        "e2e_latency_pass": e2e["pass"] if e2e else None,
        "keygen_time_s": keygen_s,
        "keygen_pass": keygen_s <= THRESHOLDS["keygen_time_s"],
        "ciphertext_median_bytes": ciphertext_stats["median"],
        "ciphertext_pass": ciphertext_stats["max"] <= THRESHOLDS["ciphertext_size_bytes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--seed", type=int, default=8047)
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "xgboost", "both"],
        default="both",
    )
    args = parser.parse_args()

    n_runs = 5 if args.quick else args.n
    fhe_mode = "simulate" if args.quick else "execute"
    rng = np.random.default_rng(args.seed)
    targets = (
        ["logistic_regression", "xgboost"]
        if args.model == "both"
        else [args.model]
    )

    selections = load_selected_models()
    df_train, df_test, X_train, y_train, X_test, y_test = load_splits()
    verify_ingestion(df_train, df_test)
    verify_input_vector(X_train)
    clean_directory()

    per_model = {}
    for model_type in targets:
        params = selections[model_type]["params"]
        per_model[model_type] = run_model_evidence(
            model_type,
            params,
            X_train,
            y_train,
            X_test,
            y_test,
            n_runs,
            fhe_mode,
            rng,
            args.skip_e2e,
        )

    summary = {
        "run_config": {
            "n_runs": n_runs,
            "fhe_accuracy_mode": fhe_mode,
            "seed": args.seed,
            "models": targets,
        },
        "models": per_model,
    }
    save_json(METRICS_DIR / "evidence_summary.json", summary)

    # Legacy single-file aliases for LR
    if "logistic_regression" in per_model:
        for name in ("accuracy_comparison", "cleartext_metrics", "latency_stats", "e2e_latency"):
            src = METRICS_DIR / f"{name}_logistic_regression.json"
            if src.is_file():
                (METRICS_DIR / f"{name}.json").write_text(src.read_text())

    print("\n=== Per-model verdicts ===")
    for mt, entry in per_model.items():
        print(f"  {mt}:")
        for k, v in entry.items():
            if k.endswith("_pass"):
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
