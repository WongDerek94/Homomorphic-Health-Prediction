---
title: Health Prediction On Encrypted Data Using Fully Homomorphic Encryption
emoji: 🩺😷
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
tags:
  - FHE
  - PPML
  - privacy
  - privacy preserving machine learning
  - image processing
  - homomorphic encryption
  - security
python_version: 3.10.6
---

# SecureMed — Healthcare prediction using FHE

Privacy-preserving symptom-to-diagnosis prediction with Concrete-ML. Supports **logistic regression** (baseline) and **XGBoost** (tree ensemble) via a mandatory model picker in the Gradio UI.

## Setup (once)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## One-command reproducibility

Regenerate deployment bundles and the full evidence package from committed model selection artifacts:

```bash
source venv/bin/activate
python scripts/run_all.py              # verify data → train both → evidence N=100
python scripts/run_all.py --quick      # smoke test (N=5, fhe=simulate)
./scripts/run_all.sh --quick           # same via shell wrapper

# Optional flags:
#   --run-model-selection   full grid search (hours; not default)
#   --skip-train              skip FHE bundle rebuild
#   --skip-evidence           train only
#   --regenerate-data         rebuild CSVs from raw Training.csv/Testing.csv
```

## Phase 2 workflow (step-by-step)

```bash
# 1. Preprocess data (optional — committed CSVs exist)
python scripts/preprocess_data.py

# 2. Model selection (long-running; artifacts in models/ are committed)
python scripts/run_model_selection.py
# Or: jupyter notebook scripts/model_selection.ipynb

# 3. Train and deploy FHE bundles
python scripts/train_models.py --model both

# 4. Generate evaluation evidence (N=100, both models)
python scripts/generate_evidence.py --model both
python scripts/generate_evidence.py --quick   # smoke test
```

## Run the application

```bash
source venv/bin/activate
python app.py
```

Open the Gradio URL (e.g. `http://127.0.0.1:8888`). Select a model before entering symptoms. The app spawns the FastAPI server on port 8000.

## Evidence and metrics

Results are written to `metrics/` and `artifacts/`, with pass/fail verdicts in `metrics/evidence_summary.json`.

| Model | FHE latency (median) | Ciphertext | Accuracy |
|-------|---------------------|------------|----------|
| Logistic regression (`n_bits=8`) | ~19 ms | ~1.36 MB | 100% |
| XGBoost (`n_bits=6, depth=3, est=5`) | ~19.7 s | ~1.25 MB | 97.6% |

See `docs/FINAL_REPORT_DRAFT.md`, `docs/PHASE_TRACKING.md`, and `docs/GAP_ANALYSIS.md` for full details.
