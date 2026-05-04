import os
import json
import pytest

from neural_scalpel.route.policy import RouteStatus
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.tenant import TenantContext
from neural_scalpel.route.registry import RouteRegistry

# Mock data
SECRET_KEYS = {"prod-key-1": "super-secret-hmac-key"}
RUNTIME_MODEL_HASH = "8b1a9953c4611296a827abf8c47804d7e4cf39d88532ee43640243e86ac58e77"
VALID_TENANT = TenantContext("tenant-xyz")

def create_valid_route_data():
    return {
        "route_schema_version": "0.1.0",
        "route_id": "test-route-1",
        "source_model": "model-a",
        "target_model": "model-b",
        "source_adapter_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "target_model_sha256": RUNTIME_MODEL_HASH,
        "tenant_id": "tenant-xyz",
        "license": "MIT",
        "projection_method": "TEST",
        "calibration": {"forward_passes": 64},
        "diagnostics": {"verdict": "PASS", "ppl_degradation": 0, "kl_divergence": 0, "portability_score": 100},
        "layers": [{"name": "L1", "shape": [10, 10], "dtype": "float32", "delta_sha256": "a"*64}]
    }

@pytest.fixture
def registry(tmp_path):
    signer = RouteSigner(SECRET_KEYS)
    return RouteRegistry(storage_dir=str(tmp_path), signer=signer)

@pytest.fixture
def signed_route_file(tmp_path):
    signer = RouteSigner(SECRET_KEYS)
    route_data = create_valid_route_data()
    signed_data = signer.sign(route_data, "prod-key-1")
    path = tmp_path / "valid_route.json"
    with open(path, "w") as f:
        json.dump(signed_data, f)
    return str(path)

# ---------------------------------------------------------
# Test Cases
# ---------------------------------------------------------

def test_valid_signed_route_passes(registry, signed_route_file):
    route_id = registry.register_route(signed_route_file)
    assert registry.get_route_status(route_id) == RouteStatus.PRODUCTION
    
    # Retrieval passes
    route = registry.get_verified_route(route_id, RUNTIME_MODEL_HASH, VALID_TENANT)
    assert route["route_id"] == "test-route-1"

def test_missing_signature_rejected(registry, tmp_path):
    route_data = create_valid_route_data()
    # DO NOT SIGN IT
    path = tmp_path / "unsigned.json"
    with open(path, "w") as f: json.dump(route_data, f)
    
    with pytest.raises(ValueError, match="Route manifest schema validation failed"):
        registry.register_route(str(path))

def test_invalid_signature_rejected(registry, tmp_path):
    route_data = create_valid_route_data()
    signer = RouteSigner(SECRET_KEYS)
    signed_data = signer.sign(route_data, "prod-key-1")
    
    # Tamper with the payload after signing
    signed_data["license"] = "GPL"
    
    path = tmp_path / "tampered.json"
    with open(path, "w") as f: json.dump(signed_data, f)
    
    with pytest.raises(ValueError, match="Cryptographic signature verification failed"):
        registry.register_route(str(path))

def test_hash_mismatch_rejected(registry, tmp_path):
    route_data = create_valid_route_data()
    route_data["source_adapter_sha256"] = "invalid-hash-format!"
    signer = RouteSigner(SECRET_KEYS)
    signed_data = signer.sign(route_data, "prod-key-1")
    
    path = tmp_path / "bad_hash.json"
    with open(path, "w") as f: json.dump(signed_data, f)
    
    with pytest.raises(ValueError, match="Invalid source_adapter_sha256 format"):
        registry.register_route(str(path))

def test_target_model_mismatch_rejected(registry, signed_route_file):
    route_id = registry.register_route(signed_route_file)
    
    # Try to retrieve with a different running model hash
    with pytest.raises(ValueError, match="does not match current runtime model hash"):
        registry.get_verified_route(route_id, "different-hash", VALID_TENANT)

def test_revoked_route_rejected(registry, signed_route_file):
    route_id = registry.register_route(signed_route_file)
    registry.revoke_route(route_id)
    
    with pytest.raises(PermissionError, match="is REVOKED and cannot be executed"):
        registry.get_verified_route(route_id, RUNTIME_MODEL_HASH, VALID_TENANT)

def test_quarantined_route_rejected(registry, signed_route_file):
    route_id = registry.register_route(signed_route_file)
    registry.quarantine_route(route_id, "Bad Rollback")
    
    with pytest.raises(PermissionError, match="is QUARANTINED and cannot be executed"):
        registry.get_verified_route(route_id, RUNTIME_MODEL_HASH, VALID_TENANT)

def test_tenant_mismatch_rejected(registry, signed_route_file):
    route_id = registry.register_route(signed_route_file)
    
    malicious_tenant = TenantContext("tenant-evil")
    with pytest.raises(PermissionError, match="Tenant mismatch"):
        registry.get_verified_route(route_id, RUNTIME_MODEL_HASH, malicious_tenant)

def test_license_high_risk_blocks_production(registry, tmp_path):
    route_data = create_valid_route_data()
    route_data["license"] = "AGPL"
    signer = RouteSigner(SECRET_KEYS)
    signed_data = signer.sign(route_data, "prod-key-1")
    
    path = tmp_path / "agpl_route.json"
    with open(path, "w") as f: json.dump(signed_data, f)
    
    with pytest.raises(PermissionError, match="high-risk license policy: AGPL"):
        registry.register_route(str(path))

def test_unknown_license_requires_manual_review(registry, tmp_path):
    route_data = create_valid_route_data()
    route_data["license"] = "CustomCorp-License"
    signer = RouteSigner(SECRET_KEYS)
    signed_data = signer.sign(route_data, "prod-key-1")
    
    path = tmp_path / "unknown_license.json"
    with open(path, "w") as f: json.dump(signed_data, f)
    
    with pytest.raises(PermissionError, match="requires manual review"):
        registry.register_route(str(path))

def test_fail_closed_on_verifier_exception(registry, tmp_path):
    # Pass completely malformed JSON (not even a dictionary) to force a structural error
    # It should immediately raise, and the registry should be empty.
    path = tmp_path / "broken.json"
    with open(path, "w") as f: f.write("[\"this\", \"is\", \"not\", \"a\", \"dict\"]")
    
    with pytest.raises(Exception):
        registry.register_route(str(path))
        
    assert len(registry.list_routes()) == 0
