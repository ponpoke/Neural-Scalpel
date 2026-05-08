import pytest
import os
import json
import shutil
from pathlib import Path
from neural_scalpel.commands.package_release import run_package_release

class MockArgs:
    def __init__(self, **kwargs):
        self.run_dir = "runs/test"
        self.adapter_dir = "adapters/test"
        self.output_dir = "release/test"
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_package_release_logic(tmp_path):
    run_dir = tmp_path / "run"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "output"
    
    run_dir.mkdir()
    adapter_dir.mkdir()
    
    # Create mock artifacts
    report_data = {
        "source_adapter": "src",
        "target_model": "tgt",
        "release_decision_gate": {"verdict": "RELEASE_READY", "recommendation": "Go"}
    }
    (run_dir / "diagnostic_report.json").write_text(json.dumps(report_data))
    (run_dir / "final_analysis.md").write_text("# Analysis")
    (run_dir / "README.md").write_text("# Model Card")
    
    (adapter_dir / "adapter_model.safetensors").write_text("data")
    (adapter_dir / "adapter_config.json").write_text("{}")
    
    args = MockArgs(
        run_dir=str(run_dir),
        adapter_dir=str(adapter_dir),
        output_dir=str(output_dir)
    )
    
    run_package_release(args)
    
    # Assertions
    assert (output_dir / "adapter_model.safetensors").exists()
    assert (output_dir / "adapter_config.json").exists()
    assert (output_dir / "diagnostic_report.json").exists()
    assert (output_dir / "final_analysis.md").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "CITATION.cff").exists()
    assert (output_dir / "projection_metadata.json").exists()
    
    # Check CFF content
    cff = (output_dir / "CITATION.cff").read_text()
    assert "Neural-Scalpel-Autogen" in cff
    assert "tgt" in cff
    
    # Check metadata content
    meta = json.loads((output_dir / "projection_metadata.json").read_text())
    assert meta["framework"] == "Neural-Scalpel v2.9.0"
    assert meta["diagnostic_verdict"] == "RELEASE_READY"
    assert meta["target_model"] == "tgt"
    assert "CITATION.cff" in meta["integrity_hashes"]
