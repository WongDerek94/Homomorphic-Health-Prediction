# SecureMed — Gap Analysis: Codebase vs. Final Proposal

Comparison of the `encrypted_health_prediction` repository against the requirements in
*Major Project Proposal — Final Version Submission* (Derek Wong, Nov 11 2025).

Status as of: June 15, 2026 (post Phase 2–4 completion).

## Summary

The **Must** scope is implemented: dual-model FHE pipeline, measurement suite, deterministic
preprocessing, and a reproducible one-command entry point (`scripts/run_all.py`). Remaining
items are **documented limitations** (fallback, retention), **metric threshold failures**
(ciphertext size; XGB latency), and **final report packaging** (DOCX, screenshots, usability
narrative).

## Proposal Must scope — complete

| Proposal requirement | Where implemented |
|---|---|
| Public dataset + deterministic train/test splits | `scripts/preprocess_data.py`, `data/*_preprocessed.csv` |
| LR baseline + Concrete tree ensemble | `deployment_files/logistic_regression/`, `deployment_files/xgboost/` |
| Grid search / model selection artifacts | `models/selected_models.json`, `models/grid_search_*`, `scripts/model_selection.ipynb` |
| Encrypted client–server workflow | `app.py`, `server.py` |
| Measurement suite (N=100, accuracy delta, latency, keygen, ciphertext) | `scripts/generate_evidence.py`, `metrics/`, `artifacts/` |
| Single-command reproducibility (Should) | `scripts/run_all.py`, `scripts/run_all.sh`, `README.md` |

## Success metrics — measured results (June 15, 2026)

Source: `metrics/evidence_summary.json` (`generate_evidence.py --model both`, N=100, seed 8047).

| Metric | LR | XGB | Verdict |
|--------|----|-----|---------|
| Accuracy delta ≤ 5 pp | 0.0 pp | 0.0 pp | PASS both |
| Median FHE latency ≤ 5 s | 19 ms | 19.7 s | PASS / **FAIL** |
| Median E2E latency ≤ 5 s | 34 ms | 22.1 s | PASS / **FAIL** |
| Keygen ≤ 30 s | 0.001 s | 1.5 s | PASS both |
| Ciphertext ≤ 1 MB | ~1.36 MB | ~1.25 MB | **FAIL** both |
| 128 binary features | confirmed | confirmed | PASS both |

These failures are **documented findings**, not missing implementation. LR is the recommended
interactive demo model; XGB demonstrates tree-ensemble FHE cost.

## Documented limitations (not implemented — report as future work)

| Item | Proposal reference | Status |
|------|-------------------|--------|
| Server fallback / timeout / simulation-mode degrade | State transition; Fallback and Recomputation Tests | **Limitation** — `/run_fhe` returns HTTP 500 on missing inputs; no graceful fallback |
| 24-hour retention job + deletion audit log | Ethics and Data Governance | **Limitation** — single-host demo; manual Reset clears client/server temp dirs |
| Pre-entry data-handling disclaimer | Client workflow | **Author-provided** — documented in final report (user content) |
| Usability sessions (2–3 mock users) | Testing and Validation Plan | **Author-provided** — documented in final report (user content) |

## Resolved gaps (formerly open as of June 9, 2026)

| Former gap | Resolution |
|------------|------------|
| No `metrics/` or `artifacts/` | Populated by `generate_evidence.py --model both` |
| No automated N=100 suite | Done |
| No `preprocess_data.py` | Done |
| LR-only deployment | Dual LR + XGB bundles and UI model picker |
| No grid search artifacts | `models/` directory complete |
| HIPAA badge | Replaced with PRIVACY BY DESIGN |
| Input vector verification | `artifacts/input_vector_size.log` |

## Optional (Could scope — not required)

- Expanded latency/ciphertext visualizations in report appendix from `model_selection.ipynb`
- Further LR `n_bits` retuning (6–7) to chase 1 MB ciphertext threshold