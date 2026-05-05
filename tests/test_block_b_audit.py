import os
import json
import pytest
import torch
import uuid

from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.experimental.runtime import HotSwapRuntime, RuntimeState
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.tenant import TenantContext
from neural_scalpel.experimental.audit import AuditLogger

SECRET_KEYS = {"audit-key": "secret"}
RUNTIME_MODEL_HASH = "8b1a9953c4611296a827abf8c47804d7e4cf39d88532ee43640243e86ac58e77"
VALID_TENANT = TenantContext("tenant-xyz")

def create_valid_route_data():
    return {
        "route_schema_version": "0.1.0",
        "route_id": "audit-route-1",
        "source_model": "model-a",
        "target_model": "model-b",
        "source_adapter_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "target_model_sha256": RUNTIME_MODEL_HASH,
        "tenant_id": "tenant-xyz",
        "license": "MIT",
        "projection_method": "TEST",
        "calibration": {"forward_passes": 64},
        "diagnostics": {"verdict": "PASS", "ppl_degradation": 0, "kl_divergence": 0, "portability_score": 100},
        "layers": [{"name": "layer_1", "shape": [2, 2], "dtype": "float32", "delta_sha256": "a"*64}]
    }

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))

@pytest.fixture
def env(tmp_path):
    signer = RouteSigner(SECRET_KEYS)
    registry = RouteRegistry(storage_dir=str(tmp_path / "registry"), signer=signer)
    log_file = str(tmp_path / "audit.jsonl")
    audit_logger = AuditLogger(log_file)
    
    route_data = create_valid_route_data()
    signed_data = signer.sign(route_data, "audit-key")
    
    route_path = str(tmp_path / "route.json")
    with open(route_path, "w") as f:
        json.dump(signed_data, f)
        
    route_id = registry.register_route(route_path)
    
    model = MockModel()
    runtime = HotSwapRuntime(target_model=model, registry=registry, runtime_model_hash=RUNTIME_MODEL_HASH, audit_logger=audit_logger)
    
    return runtime, route_id, log_file, registry

def read_logs(log_file):
    if not os.path.exists(log_file):
        return []
    with open(log_file, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def test_successful_inference_logs_all_events(env):
    runtime, route_id, log_file, _ = env
    
    def dummy_infer():
        return "success"
        
    req_id = "req-123"
    result = runtime.infer(route_id, VALID_TENANT, req_id, dummy_infer)
    assert result == "success"
    
    logs = read_logs(log_file)
    events = [log["event"] for log in logs]
    # New runtime uses more granular event names
    expected_events = [
        "route_verified", "snapshot_started", "snapshot_captured",
        "swap_started", "swap_completed",
        "forward_started", "forward_completed",
        "rollback_started", "rollback_completed"
    ]
    assert events == expected_events
    
    assert all(log["request_id"] == req_id for log in logs)
    assert all(log["tenant_id"] == VALID_TENANT.tenant_id for log in logs)

def test_route_rejected_logged(env):
    runtime, route_id, log_file, registry = env
    
    invalid_tenant = TenantContext("hacker")
    req_id = "req-456"
    
    with pytest.raises(PermissionError):
        runtime.infer(route_id, invalid_tenant, req_id, lambda: None)
        
    logs = read_logs(log_file)
    assert len(logs) == 1
    assert logs[0]["event"] == "route_rejected"
    assert logs[0]["status"] == "failure"
    assert "Tenant mismatch" in logs[0]["failure_reason"]

def test_rollback_failure_quarantine_logged(env):
    runtime, route_id, log_file, _ = env
    
    def malicious_infer():
        # Corrupt the snapshot to force a rollback mismatch
        with torch.no_grad():
            runtime.snapshots["layer_1"].add_(10.0)
        return "done"
        
    req_id = "req-789"
    with pytest.raises(RuntimeError, match="CRITICAL.*QUARANTINED"):
        runtime.infer(route_id, VALID_TENANT, req_id, malicious_infer)
        
    logs = read_logs(log_file)
    events = [log["event"] for log in logs]
    
    assert "rollback_failed" in events or "worker_quarantined" in events
    
    # Check for quarantine event (either worker_quarantined or rollback_failed)
    q_logs = [log for log in logs if log["event"] in ("worker_quarantined", "rollback_failed")]
    assert len(q_logs) > 0
    assert q_logs[0]["status"] == "failure"
