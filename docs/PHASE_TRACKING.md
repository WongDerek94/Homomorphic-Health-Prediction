# SecureMed — Phase Tracking Against the Proposal Schedule

Maps the current state of the `encrypted_health_prediction` repository to the four phases defined
in the proposal's Schedule and Milestones section (planned Jan–Apr 2026).

Status as of: June 15, 2026.

## Phase 1 — Environment Setup and Reproducible Data Pipeline (Weeks 1–2)

**Status: complete.**

| Proposal deliverable | State |
|---|---|
| Deterministic train/test splits | Done — `data/Training_preprocessed.csv` (4920 rows), `data/Testing_preprocessed.csv` (42 rows), 128 binary symptom features + 2 target columns |
| `scripts/preprocess_data.py` regenerating splits from raw data | Done — converts `Training.csv`/`Testing.csv` with column renames, merge rules, label encoding, and SHA256 verification |
| Ingestion verification (row counts, checksums) | Done — `scripts/generate_evidence.py` writes `data/testset_checksum.txt` |

## Phase 2 — Baseline Models, Cleartext Metrics, and Model Selection (Weeks 3–6)

**Status: complete.**

| Proposal deliverable | State |
|---|---|
| Logistic regression baseline with cleartext metrics | Done — selected `C=1.0, n_bits=8, solver=sag`; 100% test accuracy |
| Grid search over Concrete-compatible tree ensembles | Done — staged pipeline in `scripts/model_selection_utils.py`; 108 XGB configs searched |
| `scripts/model_selection.ipynb` with Discussion | Done — grids, plots, artifact export |
| Selected models archived | Done — `models/selected_models.json`, `models/cleartext_baselines.pkl`, `models/grid_search_*_results.csv/json` |
| Dual deployment bundles | Done — `deployment_files/logistic_regression/`, `deployment_files/xgboost/` via `scripts/train_models.py` |
| Dual-model UI with mandatory picker | Done — `app.py` model dropdown; HIPAA badge replaced with PRIVACY BY DESIGN |

**Selected models (June 15, 2026):**

- LR: `C=1.0, n_bits=8, solver=sag` (CV-best was `n_bits=7`)
- XGB: `n_bits=6, max_depth=3, n_estimators=5` (CV-best was `max_depth=1, n_estimators=20`)

## Phase 3 — FHE Compilation, Key Generation, and Client-Server Integration (Weeks 7–9)

**Status: complete.**

| Proposal deliverable | State |
|---|---|
| FHE-compiled circuits (both models) | Done — `scripts/train_models.py` compiles LR + XGB |
| Client key generation (secret + evaluation keys) | Done — per-model keys under `KEYS_DIR/{user_id}/{model_type}/` |
| Server endpoints with model selection | Done — `server.py` dual `FHE_SERVERS`; `model_type` on all endpoints |
| Configurable demo sleep | Done — `SECUREMED_GET_OUTPUT_DELAY_S` env (default 1.0 s; evidence sets 0) |
| End-to-end encrypted inference | Done — verified for both models |

## Phase 4 — Encrypted Execution Experiments, Evaluation Metrics, Final Evidence (Weeks 10–12)

**Status: complete pending final report submission.**

`scripts/generate_evidence.py --model both` produced a full N=100 evidence package on June 15, 2026
(see `metrics/evidence_summary.json`):

### Logistic regression

| Success metric | Threshold | Measured | Verdict |
|---|---|---|---|
| Accuracy delta | ≤ 5.0 pp | 0.0 pp (100% both) | PASS |
| Median FHE inference latency | ≤ 5.0 s | 0.019 s | PASS |
| Median end-to-end latency | ≤ 5.0 s | 0.034 s | PASS |
| Client key generation time | ≤ 30 s | 0.001 s | PASS |
| Ciphertext size per sample | ≤ 1 MB | **1.36 MB** | **FAIL** |

### XGBoost

| Success metric | Threshold | Measured | Verdict |
|---|---|---|---|
| Accuracy delta | ≤ 5.0 pp | 0.0 pp (97.6% both) | PASS |
| Median FHE inference latency | ≤ 5.0 s | **19.73 s** | **FAIL** |
| Median end-to-end latency | ≤ 5.0 s | **22.06 s** | **FAIL** |
| Client key generation time | ≤ 30 s | 1.50 s | PASS |
| Ciphertext size per sample | ≤ 1 MB | **1.25 MB** | **FAIL** |

### Notes

- LR `n_bits` retuning (13 → 8) reduced ciphertext ~29% while preserving 100% accuracy; 1 MB threshold still unmet.
- XGB shallow trees exceed the 5 s latency budget; LR remains the practical interactive choice.
- E2E measurements use `SECUREMED_GET_OUTPUT_DELAY_S=0` (no artificial sleep).
- Report draft updated in `docs/FINAL_REPORT_DRAFT.md` with §2.5.5 and Appendix D.

## Remaining milestones (priority order)

1. **Final report submission** — `docs/FINAL_REPORT.docx` from draft; insert screenshots; fill author disclaimer and usability sections in §2.7.
2. **Demo rehearsal** — LR end-to-end flow in `app.py` (see `docs/DEMO_CHECKLIST.md`).
3. **Optional:** LR `n_bits` retuning (6–7) to chase 1 MB ciphertext threshold.

## Documented limitations (no further code before submission)

- Server fallback / timeout / simulation-mode degrade
- 24-hour retention job and audit log
- Ciphertext ≤ 1 MB (both models); XGB FHE/E2E latency ≤ 5 s
