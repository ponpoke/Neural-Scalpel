"""
Phase 7E: Live vLLM 100 Request E2E Test
- test_live_same_route_100_req: Verifies hook firing with consistent route.
- test_live_mixed_route_homogeneous_scheduling: Verifies route-aware scheduling.
"""
import pytest
import asyncio
from typing import List

from pathlib import Path
import hashlib
import torch
from safetensors.torch import save_file

def ensure_test_payload():
    """Dynamically generates the required safetensors payload for the E2E test."""
    payload_dir = Path("vllm_registry_storage/payloads")
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / "opt125m_sql_delta.safetensors"

    if not payload_path.exists():
        # Small delta for opt-125m QKV projection
        delta = torch.full((2304, 768), 1e-5, dtype=torch.float16)
        save_file(
            {"model.decoder.layers.0.self_attn.qkv_proj.weight": delta},
            str(payload_path),
        )

    sha256 = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    return str(payload_path), sha256

async def run_llm_test(route_pattern: List[str], expected_violations: int = 0):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        pytest.skip("vLLM not installed.")
        
    from integrations.vllm_route_plugin.patch import apply_all_patches
    from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
    from integrations.vllm_route_plugin.runtime_context import get_vllm_registry
    
    # 0. Reset metrics
    RoutePluginMetrics.reset()
    
    # 1. Register Test Routes in Registry singleton
    registry = get_vllm_registry()
    
    # Ensure payload exists and get correct SHA
    payload_uri, payload_sha256 = ensure_test_payload()
    
    # Target some layers of facebook/opt-125m
    target_layers = [
        {"name": "model.decoder.layers.0.self_attn.qkv_proj.weight", "shape": [2304, 768], "dtype": "float16"},
    ]
    
    # sql-route: Real safetensors payload
    registry.routes["sql-route"] = {
        "route_id": "sql-route",
        "tenant_id": "test-tenant",
        "layers": target_layers,
        "payload": {
            "uri": payload_uri,
            "sha256": payload_sha256
        }
    }

    
    # alpaca-route: Simulated for hybrid verification
    registry.routes["alpaca-route"] = {
        "route_id": "alpaca-route",
        "tenant_id": "test-tenant",
        "layers": target_layers,
        "payload_type": "simulated"
    }
    print(f"[Phase 7G] Registered routes (SQL: Real, Alpaca: Simulated): {list(registry.routes.keys())}")

    # 2. Apply all Neural-Scalpel patches
    apply_all_patches()
    
    # 3. Initialize LLM Engine
    # Small model for fast testing
    llm = LLM(model="facebook/opt-125m", enforce_eager=True) 
    
    # 4. Prepare 100 mixed requests
    base_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a SQL query to find the top 5 customers.",
        "What is the capital of France?",
        "How do I cook a perfect steak?"
    ]
    
    sampling_params_list = []
    for i in range(100):
        route = route_pattern[i % len(route_pattern)]
        sp = SamplingParams(max_tokens=16)
        sp.extra_args = {"route_id": route}
        sampling_params_list.append(sp)
        
    # 5. Execute E2E
    print(f"[Phase 7E] Starting E2E run with pattern: {route_pattern[:3]}...")
    
    # Generate calls original_schedule multiple times
    outputs = llm.generate(
        [base_prompts[i % len(base_prompts)] for i in range(100)],
        sampling_params_list
    )
    
    success = len(outputs) == 100
    
    # 6. Report Metrics
    print("\n" + "="*50)
    print(f"[Phase 7E] Runtime Metrics (Pattern: {route_pattern[:3]}...):")
    print(f" - Request Count: {RoutePluginMetrics.request_count}")
    print(f" - Forward Count: {RoutePluginMetrics.forward_count}")
    print(f" - Active Route (Last): {RoutePluginMetrics.get_active_route()}")
    print(f" - Swap Count: {RoutePluginMetrics.swap_count}")
    print(f" - Rollback Count: {RoutePluginMetrics.rollback_count}")
    print(f" - Violations: {RoutePluginMetrics.mixed_batch_violation_count}")
    print(f" - Scheduler Queue Observations: {RoutePluginMetrics.scheduler_queue_observations}")
    print(f" - Request Routes Sample: {list(RoutePluginMetrics.request_routes.items())[:3]}")

    # After Phase 7F-2 implementation, all patterns should pass 100% successfully.
    # The scheduler now ensures only homogeneous batches are formed.
    assert success == True
    assert RoutePluginMetrics.request_count == 100
    assert RoutePluginMetrics.swap_count > 0
    assert RoutePluginMetrics.swap_count == RoutePluginMetrics.rollback_count
    assert RoutePluginMetrics.mixed_batch_violation_count == 0

@pytest.mark.vllm_live
@pytest.mark.asyncio
async def test_live_same_route_100_req():
    """Verify that a consistent route processed by vLLM triggers hooks correctly."""
    await run_llm_test(route_pattern=["sql-route"])

@pytest.mark.vllm_live
@pytest.mark.asyncio
async def test_live_mixed_route_homogeneous_scheduling():
    """Verify mixed-route requests complete via route-homogeneous scheduling."""
    # Pattern alternates route every request.
    # Native scheduler would normally mix these, but 7F-2 enforcement should separate them.
    await run_llm_test(route_pattern=["__base__", "sql-route", "alpaca-route"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_llm_test(route_pattern=["__base__", "sql-route", "alpaca-route"]))
