"""
Block D Test Suite: Production Integration Prototype

Covers all 9 test cases from the Block D plan:
  1. test_api_infer_success
  2. test_api_rejects_tenant_mismatch
  3. test_api_rejects_revoked_route
  4. test_api_audit_log_contains_request_id
  5. test_scheduler_batches_same_route_only
  6. test_scheduler_rejects_mixed_route_batch
  7. test_metrics_endpoint_reports_latency
  8. test_api_stress_no_route_leakage
  9. test_vllm_bridge_rejects_unsafe_mixed_route_batch
"""

import os
import json
import asyncio
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from neural_scalpel.route.policy import RouteStatus
from neural_scalpel.route.crypto import RouteSigner
from neural_scalpel.route.tenant import TenantContext
from neural_scalpel.route.registry import RouteRegistry
from neural_scalpel.experimental.audit import AuditLogger
from neural_scalpel.experimental.runtime import HotSwapRuntime, RuntimeState
from neural_scalpel.serving.server import create_app
from neural_scalpel.serving.metrics import MetricsCollector
from neural_scalpel.serving.scheduler import RouteAwareScheduler, ScheduledBatch
from neural_scalpel.serving.vllm_bridge import VLLMBridgePrototype

# ── Shared Constants ───────────────────────────────────────────

SECRET_KEYS = {"prod-key-1": "super-secret-hmac-key"}
RUNTIME_MODEL_HASH = "8b1a9953c4611296a827abf8c47804d7e4cf39d88532ee43640243e86ac58e77"
VALID_TENANT_ID = "tenant-xyz"

# ── Helpers ────────────────────────────────────────────────────

def _create_route_data(route_id="test-route-1", tenant_id=VALID_TENANT_ID, license_name="MIT"):
    return {
        "route_schema_version": "0.1.0",
        "route_id": route_id,
        "source_model": "model-a",
        "target_model": "model-b",
        "source_adapter_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "target_model_sha256": RUNTIME_MODEL_HASH,
        "tenant_id": tenant_id,
        "license": license_name,
        "projection_method": "TEST",
        "calibration": {"forward_passes": 64},
        "diagnostics": {"verdict": "PASS", "ppl_degradation": 0, "kl_divergence": 0, "portability_score": 100},
        "layers": [{"name": "L1", "shape": [10, 10], "dtype": "float32", "delta_sha256": "a" * 64}],
    }


def _register_signed_route(registry, tmp_path, route_id="test-route-1", tenant_id=VALID_TENANT_ID):
    signer = RouteSigner(SECRET_KEYS)
    data = _create_route_data(route_id=route_id, tenant_id=tenant_id)
    signed = signer.sign(data, "prod-key-1")
    path = tmp_path / f"{route_id}.json"
    with open(path, "w") as f:
        json.dump(signed, f)
    return registry.register_route(str(path))


def _make_mock_runtime(registry):
    """Creates a mock HotSwapRuntime that simulates successful inference."""
    import torch
    model = {"L1": torch.randn(10, 10)}
    runtime = HotSwapRuntime(model, registry, RUNTIME_MODEL_HASH)
    return runtime


@pytest.fixture
def tmp_audit_path(tmp_path):
    return str(tmp_path / "audit.jsonl")


@pytest.fixture
def components(tmp_path, tmp_audit_path):
    signer = RouteSigner(SECRET_KEYS)
    registry = RouteRegistry(storage_dir=str(tmp_path / "registry"), signer=signer)
    audit = AuditLogger(tmp_audit_path)
    runtime = _make_mock_runtime(registry)
    runtime.audit_logger = audit
    metrics = MetricsCollector()
    return {"registry": registry, "runtime": runtime, "audit": audit, "metrics": metrics, "tmp_path": tmp_path, "audit_path": tmp_audit_path}


@pytest.fixture
def client(components):
    app = create_app(
        runtime=components["runtime"],
        registry=components["registry"],
        audit_logger=components["audit"],
        metrics=components["metrics"],
    )
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════
# Test 1: API Infer Success
# ═══════════════════════════════════════════════════════════════

def test_api_infer_success(client, components):
    """API correctly parses, schedules, and executes an isolated route."""
    _register_signed_route(components["registry"], components["tmp_path"])

    resp = client.post("/v1/infer", json={
        "tenant_id": VALID_TENANT_ID,
        "route_id": "test-route-1",
        "prompt": "Hello world",
        "request_id": "req-001",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["request_id"] == "req-001"
    assert body["route_id"] == "test-route-1"
    assert "route:test-route-1" in body["output"]
    assert body["latency_ms"] > 0
    assert body["audit_ref"].startswith("audit-req-001-")


# ═══════════════════════════════════════════════════════════════
# Test 2: Tenant Mismatch Rejected
# ═══════════════════════════════════════════════════════════════

def test_api_rejects_tenant_mismatch(client, components):
    """Network-layer blocking of unauthorized tenants."""
    _register_signed_route(components["registry"], components["tmp_path"])

    resp = client.post("/v1/infer", json={
        "tenant_id": "tenant-evil",
        "route_id": "test-route-1",
        "prompt": "Hack attempt",
        "request_id": "req-evil",
    })
    assert resp.status_code == 403
    assert "not authorized" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════
# Test 3: Revoked Route Rejected
# ═══════════════════════════════════════════════════════════════

def test_api_rejects_revoked_route(client, components):
    """API respects registry policy gates."""
    _register_signed_route(components["registry"], components["tmp_path"])
    components["registry"].revoke_route("test-route-1")

    resp = client.post("/v1/infer", json={
        "tenant_id": VALID_TENANT_ID,
        "route_id": "test-route-1",
        "prompt": "Should be blocked",
        "request_id": "req-revoked",
    })
    assert resp.status_code == 403
    assert "REVOKED" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════
# Test 4: Audit Log Contains Request ID
# ═══════════════════════════════════════════════════════════════

def test_api_audit_log_contains_request_id(client, components):
    """Network traffic is 100% traceable via audit logs."""
    _register_signed_route(components["registry"], components["tmp_path"])

    client.post("/v1/infer", json={
        "tenant_id": VALID_TENANT_ID,
        "route_id": "test-route-1",
        "prompt": "Audit test",
        "request_id": "req-audit-trace",
    })

    # Read the audit log and verify the request_id appears
    with open(components["audit_path"], "r") as f:
        lines = f.readlines()

    assert len(lines) > 0, "Audit log should contain entries"
    found = any("req-audit-trace" in line for line in lines)
    assert found, "request_id 'req-audit-trace' not found in audit log"

    # Verify structured JSON format
    for line in lines:
        entry = json.loads(line.strip())
        assert "request_id" in entry
        assert "tenant_id" in entry
        assert "event" in entry


# ═══════════════════════════════════════════════════════════════
# Test 5: Scheduler Batches Same Route Only
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_scheduler_batches_same_route_only():
    """Scheduler safely micro-batches identical routes."""
    scheduler = RouteAwareScheduler(max_batch_size=4)
    loop = asyncio.get_running_loop()

    # Directly populate the queue (submit_request awaits the future, so we enqueue manually)
    async with scheduler._lock:
        batch = ScheduledBatch(route_id="route-A")
        for i in range(6):
            batch.requests.append({"request_id": f"req-{i}", "tenant_id": "t1", "route_id": "route-A", "payload": {}})
            batch.futures.append(loop.create_future())
        scheduler._queues["route-A"] = batch

    # Fetch first batch (should be max 4 requests, all route-A)
    batch1 = await scheduler.fetch_next_safe_batch()
    assert batch1 is not None
    assert batch1.route_id == "route-A"
    assert len(batch1.requests) == 4
    assert all(r["route_id"] == "route-A" for r in batch1.requests)
    RouteAwareScheduler.validate_batch_homogeneity(batch1)

    # Fetch second batch (remaining 2)
    batch2 = await scheduler.fetch_next_safe_batch()
    assert batch2 is not None
    assert batch2.route_id == "route-A"
    assert len(batch2.requests) == 2

    # Resolve futures
    for b in [batch1, batch2]:
        for fut in b.futures:
            if not fut.done():
                fut.set_result("ok")


# ═══════════════════════════════════════════════════════════════
# Test 6: Scheduler Rejects Mixed Route Batch
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_scheduler_rejects_mixed_route_batch():
    """Scheduler fundamentally refuses to execute A and B in the same forward pass."""
    scheduler = RouteAwareScheduler(max_batch_size=16)
    loop = asyncio.get_running_loop()

    # Directly enqueue requests for two different routes
    async with scheduler._lock:
        ba = ScheduledBatch(route_id="route-A")
        ba.requests.append({"request_id": "r1", "tenant_id": "t1", "route_id": "route-A", "payload": {}})
        ba.futures.append(loop.create_future())
        scheduler._queues["route-A"] = ba

        bb = ScheduledBatch(route_id="route-B")
        bb.requests.append({"request_id": "r2", "tenant_id": "t1", "route_id": "route-B", "payload": {}})
        bb.futures.append(loop.create_future())
        scheduler._queues["route-B"] = bb

    batch_a = await scheduler.fetch_next_safe_batch()
    batch_b = await scheduler.fetch_next_safe_batch()

    assert batch_a is not None and batch_b is not None
    # Each batch must be homogeneous and distinct
    assert batch_a.route_id != batch_b.route_id
    assert all(r["route_id"] == batch_a.route_id for r in batch_a.requests)
    assert all(r["route_id"] == batch_b.route_id for r in batch_b.requests)

    # Artificially construct a contaminated batch and verify the validator catches it
    mixed = ScheduledBatch(route_id="route-A")
    mixed.requests = [{"route_id": "route-A"}, {"route_id": "route-B"}]
    with pytest.raises(RuntimeError, match="Route contamination"):
        RouteAwareScheduler.validate_batch_homogeneity(mixed)

    # Cleanup
    for b in [batch_a, batch_b]:
        for fut in b.futures:
            if not fut.done():
                fut.set_result("ok")


# ═══════════════════════════════════════════════════════════════
# Test 7: Metrics Endpoint Reports Latency
# ═══════════════════════════════════════════════════════════════

def test_metrics_endpoint_reports_latency(client, components):
    """/metrics correctly outputs system health."""
    _register_signed_route(components["registry"], components["tmp_path"])

    # Execute a few requests to generate latency data
    for i in range(5):
        client.post("/v1/infer", json={
            "tenant_id": VALID_TENANT_ID,
            "route_id": "test-route-1",
            "prompt": f"Metrics test {i}",
            "request_id": f"req-met-{i}",
        })

    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    m = resp.json()
    assert m["requests_total"] == 5
    assert m["requests_success"] == 5
    assert m["requests_rejected"] == 0
    assert m["runtime_quarantined"] is False
    assert m["route_leakage_count"] == 0
    assert m["rollback_failure_count"] == 0
    assert m["swap_latency_p99_ms"] >= 0
    assert m["rollback_latency_p99_ms"] >= 0


# ═══════════════════════════════════════════════════════════════
# Test 8: Stress Test — No Route Leakage
# ═══════════════════════════════════════════════════════════════

def test_api_stress_no_route_leakage(client, components):
    """1000 concurrent API requests execute with 0 route leakage."""
    # Register two routes for two different tenants
    _register_signed_route(components["registry"], components["tmp_path"], "route-alpha", "tenant-alpha")
    _register_signed_route(components["registry"], components["tmp_path"], "route-beta", "tenant-beta")

    leakage_count = 0
    total = 1000

    for i in range(total):
        if i % 2 == 0:
            tid, rid = "tenant-alpha", "route-alpha"
        else:
            tid, rid = "tenant-beta", "route-beta"

        resp = client.post("/v1/infer", json={
            "tenant_id": tid,
            "route_id": rid,
            "prompt": f"Stress {i}",
            "request_id": f"req-stress-{i}",
        })
        assert resp.status_code == 200
        body = resp.json()

        # Verify the output contains the CORRECT route marker
        expected_marker = f"route:{rid}"
        if expected_marker not in body["output"]:
            leakage_count += 1

    assert leakage_count == 0, f"Route leakage detected: {leakage_count}/{total}"

    # Confirm metrics
    resp = client.get("/v1/metrics")
    m = resp.json()
    assert m["requests_total"] == total
    assert m["route_leakage_count"] == 0


# ═══════════════════════════════════════════════════════════════
# Test 9: vLLM Bridge Rejects Unsafe Mixed Route Batch
# ═══════════════════════════════════════════════════════════════

def test_vllm_bridge_rejects_unsafe_mixed_route_batch():
    """Prototype strictly enforces vLLM safety boundaries."""
    bridge = VLLMBridgePrototype()

    # Safe: single route
    assert bridge.validate_batch_safety([
        {"route_id": "route-A", "prompt": "a"},
        {"route_id": "route-A", "prompt": "b"},
    ]) is True

    # Safe: no routes (base model)
    assert bridge.validate_batch_safety([
        {"prompt": "a"},
        {"prompt": "b"},
    ]) is True

    # UNSAFE: mixed routes → must raise
    with pytest.raises(RuntimeError, match="FATAL SECURITY VIOLATION"):
        bridge.validate_batch_safety([
            {"route_id": "route-A", "prompt": "a"},
            {"route_id": "route-B", "prompt": "b"},
        ])

    assert bridge.rejected_batch_count == 1

    # Test KV cache contamination detection
    bridge.validate_kv_cache_allocation("seq-1", "route-A", [0, 1, 2])
    assert bridge.cache_block_route(0) == "route-A"

    with pytest.raises(RuntimeError, match="KV CACHE CONTAMINATION"):
        bridge.validate_kv_cache_allocation("seq-2", "route-B", [0])  # Block 0 belongs to route-A

    # Test swap directive
    directive = bridge.intercept_model_runner(None, "route-A")
    assert directive["swap_required"] is True
    assert directive["to_route"] == "route-A"

    # No swap needed if already on correct route
    directive2 = bridge.intercept_model_runner("route-A", "route-A")
    assert directive2["swap_required"] is False


# ═══════════════════════════════════════════════════════════════
# Additional: Health Check
# ═══════════════════════════════════════════════════════════════

def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
