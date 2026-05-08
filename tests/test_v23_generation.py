import pytest
import json
from pathlib import Path
from neural_scalpel.core.diagnostic import AdapterTransferDiagnosticReport
from neural_scalpel.commands.generate_report import generate_markdown_report
from neural_scalpel.commands.generate_model_card import generate_model_card

def test_generate_report_content():
    report = AdapterTransferDiagnosticReport(
        run_id="test-id",
        source_base_model="source-base",
        source_adapter="path/to/adapter",
        target_model="target-model"
    )
    report.release_decision_gate.verdict = "RELEASE_READY"
    report.release_decision_gate.recommendation = "Great job"
    
    eval_data = {
        "target_evaluation": {
            "base_metrics": {"acc": 0.5},
            "adapter_metrics": {"acc": 0.6},
            "delta": {"acc": 0.1},
            "failure_classification": {"fixed": 5, "regressed": 1}
        }
    }
    
    md = generate_markdown_report(report, eval_data)
    
    assert "# Neural-Scalpel Diagnostic & Evaluation Report" in md
    assert "RELEASE_READY" in md
    assert "target-model" in md
    assert "Fixed (Improved):** 5" in md
    assert "10.00%" in md # Delta

def test_generate_model_card_content():
    report = AdapterTransferDiagnosticReport(
        source_adapter="ponpoke/my-cool-lora",
        target_model="Qwen/Qwen2.5-0.5B"
    )
    report.metadata_gate.license = "apache-2.0"
    
    eval_data = {
        "target_evaluation": {
            "adapter_metrics": {"execution_accuracy": 0.85},
            "delta": {"execution_accuracy": 0.05}
        }
    }
    
    md = generate_model_card(report, eval_data)
    
    assert "license: apache-2.0" in md
    assert "base_model: Qwen/Qwen2.5-0.5B" in md
    assert "85.00%" in md # Accuracy
    assert "+5.00%" in md # Delta
    assert "my-cool-lora-projected" in md
