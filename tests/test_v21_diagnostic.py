import pytest
import json
import os
from pathlib import Path
from neural_scalpel.core.diagnostic import (
    AdapterTransferDiagnosticReport, 
    TargetEvaluationResult, 
    MetadataGateResult,
    DeltaHealthResult,
    FeasibilityResult
)

def test_diagnostic_report_serialization_v21(tmp_path):
    report_path = tmp_path / "test_report.json"
    
    report = AdapterTransferDiagnosticReport(
        run_id="test-run",
        source_base_model="source-base",
        source_adapter="source-adapter"
    )
    report.metadata_gate.status = "PASS"
    report.metadata_gate.base_model_matches = True
    
    report.source_quality_gate = {"verdict": "POSITIVE_TEACHER", "gate_status": "PASS"}
    report.delta_health_gate.verdict = "HEALTHY_DELTA"
    report.feasibility_gate.verdict = "FEASIBLE"
    
    report.save(str(report_path))
    
    # Reload and check
    loaded = AdapterTransferDiagnosticReport.from_json(str(report_path))
    assert loaded.schema_version == "adapter_transfer_diagnostic.v2.1"
    assert loaded.metadata_gate.status == "PASS"
    assert loaded.source_quality_gate["verdict"] == "POSITIVE_TEACHER"
    assert loaded.feasibility_gate.verdict == "FEASIBLE"

def test_release_decision_promotion():
    report = AdapterTransferDiagnosticReport()
    
    # Setup for PROJECTION_CANDIDATE
    report.source_quality_gate = {"verdict": "POSITIVE_TEACHER", "gate_status": "PASS"}
    report.delta_health_gate.verdict = "HEALTHY_DELTA"
    report.feasibility_gate.verdict = "FEASIBLE"
    
    report.finalize_release_decision()
    assert report.release_decision_gate.verdict == "PROJECTION_CANDIDATE"
    assert "diagnostic_report.json" in report.release_decision_gate.required_artifacts
    
    # Simulate positive target evaluation
    report.target_evaluation_gate.verdict = "POSITIVE_TARGET_TRANSFER"
    report.finalize_release_decision()
    assert report.release_decision_gate.verdict == "RELEASE_READY"
    assert "model_card.md" in report.release_decision_gate.required_artifacts
    assert "projected_adapter/" in report.release_decision_gate.required_artifacts

def test_target_interference_regression():
    report = AdapterTransferDiagnosticReport()
    report.source_quality_gate = {"verdict": "POSITIVE_TEACHER", "gate_status": "PASS"}
    report.delta_health_gate.verdict = "HEALTHY_DELTA"
    report.feasibility_gate.verdict = "FEASIBLE"
    
    # Simulate failure on target
    report.target_evaluation_gate.verdict = "TARGET_INTERFERENCE"
    report.finalize_release_decision()
    assert report.release_decision_gate.verdict == "RESEARCH_ONLY"
    assert "ANALYZE_TARGET_INTERFERENCE" in report.release_decision_gate.recommendation

def test_from_json_restores_target_eval(tmp_path):
    report_path = tmp_path / "eval_report.json"
    report = AdapterTransferDiagnosticReport()
    report.target_evaluation_gate = TargetEvaluationResult(
        verdict="POSITIVE_TARGET_TRANSFER",
        regression_rate=0.02
    )
    report.save(str(report_path))
    
    loaded = AdapterTransferDiagnosticReport.from_json(str(report_path))
    assert loaded.target_evaluation_gate.verdict == "POSITIVE_TARGET_TRANSFER"
    assert loaded.target_evaluation_gate.regression_rate == 0.02
