"""Server that will listen for GET and POST requests from the client."""

import os
import time
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from utils import MODEL_REGISTRY, SERVER_DIR, get_deployment_dir  # pylint: disable=no-name-in-module

from concrete.ml.deployment import FHEModelServer

FHE_SERVERS = {}
for name, path in MODEL_REGISTRY.items():
    if path.is_dir() and any(path.iterdir()):
        FHE_SERVERS[name] = FHEModelServer(path)

app = FastAPI()


def _get_output_delay_s() -> float:
    return float(os.environ.get("SECUREMED_GET_OUTPUT_DELAY_S", "1.0"))


@app.get("/")
def root():
    return {
        "message": "Welcome to your disease prediction with FHE!",
        "loaded_models": list(FHE_SERVERS.keys()),
    }


@app.post("/send_input")
def send_input(
    user_id: str = Form(),
    model_type: str = Form(default="logistic_regression"),
    files: List[UploadFile] = File(),
):
    print(f"\nSend the data to the server (model={model_type}) ............\n")

    evaluation_key_path = SERVER_DIR / f"{user_id}_{model_type}_evaluation_key"
    encrypted_input_path = SERVER_DIR / f"{user_id}_{model_type}_encrypted_input"

    with encrypted_input_path.open("wb") as encrypted_input, evaluation_key_path.open(
        "wb"
    ) as evaluation_key:
        encrypted_input.write(files[0].file.read())
        evaluation_key.write(files[1].file.read())


@app.post("/run_fhe")
def run_fhe(
    user_id: str = Form(),
    model_type: str = Form(),
):
    if model_type not in FHE_SERVERS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_type}' not loaded. Available: {list(FHE_SERVERS.keys())}. "
            f"Run scripts/train_models.py first.",
        )

    print(f"\nRun in FHE in the server (model={model_type}) ............\n")
    evaluation_key_path = SERVER_DIR / f"{user_id}_{model_type}_evaluation_key"
    encrypted_input_path = SERVER_DIR / f"{user_id}_{model_type}_encrypted_input"

    with encrypted_input_path.open("rb") as encrypted_output_file, evaluation_key_path.open(
        "rb"
    ) as evaluation_key_file:
        encrypted_input = encrypted_output_file.read()
        evaluation_key = evaluation_key_file.read()

    start = time.time()
    encrypted_output = FHE_SERVERS[model_type].run(encrypted_input, evaluation_key)
    assert isinstance(encrypted_output, bytes)
    fhe_execution_time = round(time.time() - start, 2)

    encrypted_output_path = SERVER_DIR / f"{user_id}_{model_type}_encrypted_output"
    with encrypted_output_path.open("wb") as f:
        f.write(encrypted_output)

    return JSONResponse(content=fhe_execution_time)


@app.post("/get_output")
def get_output(user_id: str = Form(), model_type: str = Form(default="logistic_regression")):
    print(f"\nGet the output from the server (model={model_type}) ............\n")

    encrypted_output_path = SERVER_DIR / f"{user_id}_{model_type}_encrypted_output"
    with encrypted_output_path.open("rb") as f:
        encrypted_output = f.read()

    delay = _get_output_delay_s()
    if delay > 0:
        time.sleep(delay)

    return Response(encrypted_output)
