# SecureMed — Gap Analysis: Codebase vs. Final Proposal

Comparison of the current `encrypted_health_prediction` repository against the requirements in
*Major Project Proposal — Final Version Submission* (Derek Wong, Nov 11 2025).

Date of analysis: June 9, 2026.

## Summary

The repository implements the complete encrypted client–server workflow (the "Must" scope core),
but is missing most of the *evidence and measurement* layer that the proposal commits to: the
`metrics/` and `artifacts/` stores, the N=100 latency suite, the accuracy-delta comparison, the
preprocessing/regeneration scripts, and the retention/audit machinery.

## What the proposal requires vs. what exists

### Implemented (aligned with proposal)

| Proposal requirement | Where it is implemented |
|---|---|
| Python client UI with guided symptom form | `app.py` (Gradio, symptom categories from `symptoms_categories.py`) |
| Minimum 5 symptoms enforced | `get_features_fn` in `app.py` (line 114) |
| Client-side key generation (secret + evaluation key) | `key_gen_fn` in `app.py` using `FHEModelClient` |
| Secret key never leaves the client | Keys stored under `deployment_files/.fhe_keys/<user_id>`; only the serialized evaluation key is POSTed |
| Client-side encryption with visible vector + ciphertext | `encrypt_fn` shows both the one-hot vector and truncated ciphertext hex |
| FastAPI server accepting only evaluation keys + ciphertext | `server.py` — `/send_input`, `/run_fhe`, `/get_output` |
| Server-side FHE execution without decryption | `FHEModelServer.run()` in `/run_fhe`, returns FHE execution time |
| Client-side decryption, top-3 diseases with probabilities | `decrypt_fn` in `app.py` |
| Uncertainty prompt ("include more symptoms") | `decrypt_fn` threshold check (probability < 0.5 or top-2 gap < 0.1) |
| Demonstration / not-a-medical-device notice | Footer markdown in `app.py` |
| Fixed train/test split from the Kaggle dataset | `data/Training_preprocessed.csv`, `data/Testing_preprocessed.csv` |
| Train-then-serve pattern (offline training, server loads compiled bundle) | `dev.py` produces `deployment_files/` (client.zip, server.zip); server only loads the bundle |
| FHE model compiled with Concrete-ML | `dev.py` — `ConcreteLogisticRegression(C=0.9, n_bits=13, solver="sag")` |

### Gaps (proposal commitments not yet in the repo)

| # | Gap | Proposal reference | Severity |
|---|---|---|---|
| 1 | No `metrics/` or `artifacts/` directories, and none of the named evidence artifacts exist (`latency_stats`, `accuracy_comparison`, `keygen_time.log`, `ciphertext_sizes.log`, `input_vector_size.log`, `e2e_latency.json`) | Success Metrics table; Testing and Validation Plan | High — these are the acceptance criteria for the course deliverable |
| 2 | No automated measurement suite (N=100 FHE latency runs, N=100 end-to-end latency runs, accuracy delta cleartext vs FHE-execute) | Success Metrics; Phase 4 | High |
| 3 | No `scripts/preprocess_data.py` — preprocessed CSVs are checked in, but there is no script that regenerates them from `Training.csv`/`Testing.csv`, and no ingestion verification (row counts, checksums, `data/testset_checksum.txt`) | Data Handling; Phase 1 | Medium |
| 4 | No tree-ensemble comparison model. The proposal requires logistic regression **plus** one Concrete-compatible tree ensemble; `utils.load_model` defines a `ConcreteXGBoostClassifier` but it is dead code — the deployed bundle is LR only | In Scope (Must): Baseline and FHE-Compatible Models | Medium |
| 5 | No grid search / simulation-mode tuning artifacts (`models/grid_search_results.json`, `models/cleartext_baselines.pkl`) | Phase 2; In Scope (Should) | Medium |
| 6 | No server fallback logic — no simulation-mode fallback, no configured timeout, no fallback-state logging. `/run_fhe` will raise on malformed input rather than degrade gracefully | State transition diagram and Fallback Logic | Medium |
| 7 | No 24-hour retention job, no deletion audit log | Ethics and Data Governance | Low (single-host demo) but explicitly promised |
| 8 | No single-command reproducibility entry point that regenerates models, preprocessing outputs, and measurements | In Scope (Should): Documentation and Reproducible Scripts | Medium |
| 9 | UI displays a "HIPAA COMPLIANT" badge (`app.py` title markdown), which directly contradicts the proposal's Out of Scope statement ("does not attempt HIPAA compliance"). The badge should be removed or reworded ("Privacy by design") | Out of Scope: No Production Level Hardening or Compliance | High (accuracy of claims) |
| 10 | No usability evaluation artifacts (mock-user observation summaries) | Testing and Validation Plan: Usability | Low — scheduled for the evaluation phase |
| 11 | Pre-step disclaimer about data handling is not shown *before* symptom entry (only the footer notice exists) | Client Side Workflow ("the interface presents a disclaimer" at the start) | Low |
| 12 | Input dimensionality must be verified as a fixed-length binary vector (proposal states 128; actual preprocessed schema must be confirmed and logged in `artifacts/input_vector_size.log`) | Success Metrics: Symptom vector size | Low |

### Plan corrections discovered during analysis

- The proposal's "FastAPI server" requirement is **already met** — `server.py` is FastAPI served by
  uvicorn (spawned from `app.py`). The earlier plan listed this as a gap; it is not.
- The deployed model is the **logistic regression baseline**, not the tree ensemble. The proposal
  treats LR as the cleartext reference and a tree ensemble as the encrypted candidate; currently LR
  fills both roles.

## Remediation delivered alongside this analysis

Gap #1 and #2 are addressed by `scripts/generate_evidence.py` (see that file's docstring), which
produces every named artifact from the Success Metrics table and evaluates each threshold.
Gap #3 is partially addressed: the script writes `data/testset_checksum.txt` and an ingestion
summary. Remaining gaps (#4–#11) are mapped to phases in `docs/PHASE_TRACKING.md`.
