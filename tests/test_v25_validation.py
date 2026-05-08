import pytest
import json
from pathlib import Path
from neural_scalpel.commands.package_release import run_package_release
from neural_scalpel.commands.package_validate import run_package_validate

class MockArgs:
    def __init__(self, **kwargs):
        self.run_dir = "runs/test"
        self.adapter_dir = "adapters/test"
        self.output_dir = "release/test"
        self.package_dir = "release/test"
        for k, v in kwargs.items():
            setattr(self, k, v)

def test_package_validation_flow_hardened(tmp_path):
    run_dir = tmp_path / "run"
    adapter_dir = tmp_path / "adapter"
    package_dir = tmp_path / "package"
    
    run_dir.mkdir()
    adapter_dir.mkdir()
    
    # Create mock artifacts
    report_data = {
        "source_adapter": "src",
        "target_model": "tgt",
        "release_decision_gate": {"verdict": "RELEASE_READY", "recommendation": "Go"}
    }
    (run_dir / "diagnostic_report.json").write_text(json.dumps(report_data))
    (adapter_dir / "adapter_config.json").write_text("{}")
    
    # 1. Create a valid package
    release_args = MockArgs(run_dir=str(run_dir), adapter_dir=str(adapter_dir), output_dir=str(package_dir))
    run_package_release(release_args)
    
    # 2. Validate it (Should Pass)
    validate_args = MockArgs(package_dir=str(package_dir))
    run_package_validate(validate_args)
    
    # 3. Tamper with a file
    (package_dir / "adapter_config.json").write_text("{'tampered': true}")
    
    # 4. Validate again (Should RAISE SystemExit(1))
    print("\n--- Testing Tamper Detection ---")
    with pytest.raises(SystemExit) as exc:
        run_package_validate(validate_args)
    assert exc.value.code == 1
    
    # 5. Delete a file
    (package_dir / "diagnostic_report.json").unlink()
    
    # 6. Validate again (Should RAISE SystemExit(1))
    print("\n--- Testing Missing Detection ---")
    with pytest.raises(SystemExit) as exc:
        run_package_validate(validate_args)
    assert exc.value.code == 1

def test_package_validation_verdict_mismatch(tmp_path):
    run_dir = tmp_path / "run"
    adapter_dir = tmp_path / "adapter"
    package_dir = tmp_path / "package_mismatch"
    
    run_dir.mkdir()
    adapter_dir.mkdir()
    
    report_data = {
        "source_adapter": "src",
        "target_model": "tgt",
        "release_decision_gate": {"verdict": "RELEASE_READY", "recommendation": "Go"}
    }
    (run_dir / "diagnostic_report.json").write_text(json.dumps(report_data))
    (adapter_dir / "adapter_config.json").write_text("{}")
    
    release_args = MockArgs(run_dir=str(run_dir), adapter_dir=str(adapter_dir), output_dir=str(package_dir))
    run_package_release(release_args)
    
    # Manually tamper with metadata verdict only
    meta_path = package_dir / "projection_metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["diagnostic_verdict"] = "TAMPERED_VERDICT"
    meta_path.write_text(json.dumps(meta))
    
    validate_args = MockArgs(package_dir=str(package_dir))
    with pytest.raises(SystemExit) as exc:
        run_package_validate(validate_args)
    assert exc.value.code == 1
