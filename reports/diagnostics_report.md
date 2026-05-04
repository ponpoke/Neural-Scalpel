# LoRA Portability Feasibility Report

## Verdict
CAUTION

## Portability Score
80 / 100

## Metrics
- PPL degradation: +0.06% PASS
- KL divergence: 0.0184 PASS
- Calibration coverage: 64 passes PASS
- Adapter norm drift: 2.4x expected WARNING
- Architecture homology: MEDIUM WARNING

## License & Compliance Check
- **Source Model License:** unknown
- **Target Model License:** apache-2.0
- **Source LoRA License:** unknown
- **Commercial Risk:** HIGH
- **Recommendation:** Manual license review required before commercial use.

## Risks
- Assumption based on cross-architecture defaults.
- Target architecture uses a different dimension scale.
- Calibration set may not fully cover the target reasoning distribution.
- Downstream task performance (HumanEval/GSM8K) is currently unverified. Requires full 6-way comparison.

## Recommendation
Safe for qualitative testing. Not recommended for production deployment until downstream benchmark validation is completed.
