# v2.9 Hardened Production Regression Tracker

## 1. Executive Summary
| Status | Mode | Verdict |
| :--- | :--- | :--- |
| **Artifact Regression** | **PASS** | Shape, Config, Keys, NaN verified. |
| **Behavioral Regression** | **IN PROGRESS** | Real-model SQL-50 accuracy verification (Fixed Extraction). |

## 2. Hardening Checkpoints
- [x] **v2.9 Hardened Fix**: Resolved GQA/KV-hidden dimension mismatch in `adapters.py`.
- [x] **Rank Projection Fix**: Fixed bug where target rank was not applied.
- [x] **Evaluation Hardening**: Improved `extract_sql` to handle base model continuation.
- [ ] Real-model SQL-50 (Linear Alpha 8/24/32)
- [ ] Real-model SQL-50 (Piecewise Alpha 16) - *Ongoing*

## 3. Real-Model Behavioral Matrix (SQL-50)
| Model/Adapter | Status | Accuracy | Exec Success | Notes |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (0.5B-Base) | - | 42.0% | 46.0% | New Baseline (Higher than previous 32%) |
| Linear Alpha 16 | v2.9 Hardened | 38.0% | 40.0% | Resolved GQA/Rank issues |
| Baseline (0.5B-Instruct)| - | 24.0% | 34.0% | Python leakage observed |

## 4. Pending Tasks
- [ ] Complete Piecewise projection (v2.9)
- [ ] Complete Linear Alpha sweep (8, 24, 32)
- [ ] Validate Adaptive Scaling logic with Mock Report
