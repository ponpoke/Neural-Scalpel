"""
Phase 7E: Live vLLM 100 Request E2E Test
- test_live_same_route_100_req: Verifies hook firing with consistent route.
- test_live_mixed_route_homogeneous_scheduling: Verifies route-aware scheduling.
"""
import pytest
import asyncio
from typing import List

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
    # We do this BEFORE LLM init so the registry is ready when hooks fire.
    registry = get_vllm_registry()
    
    # Target some layers of facebook/opt-125m
    # Names should be checked against self.model.named_parameters() in hook
    target_layers = [
        {"name": "model.decoder.layers.0.self_attn.q_proj.weight", "shape": [768, 768], "dtype": "float16"},
        {"name": "model.decoder.layers.0.self_attn.v_proj.weight", "shape": [768, 768], "dtype": "float16"},
    ]
    
    for route_id in ["sql-route", "alpaca-route"]:
        # Manual injection into registry dict to bypass signature/file logic for Phase 7G smoke test
        registry.routes[route_id] = {
            "route_id": route_id,
            "tenant_id": "test-tenant",
            "layers": target_layers,
            "payload_type": "simulated" # Triggers random delta generation in HotSwapRuntime
        }
    print(f"[Phase 7G] Registered routes for validation: {list(registry.routes.keys())}")

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

@pytest.mark.asyncio
async def test_live_same_route_100_req():
    """Verify that a consistent route processed by vLLM triggers hooks correctly."""
    await run_llm_test(route_pattern=["sql-route"])

@pytest.mark.asyncio
async def test_live_mixed_route_homogeneous_scheduling():
    """Verify mixed-route requests complete via route-homogeneous scheduling."""
    # Pattern alternates route every request.
    # Native scheduler would normally mix these, but 7F-2 enforcement should separate them.
    await run_llm_test(route_pattern=["__base__", "sql-route", "alpaca-route"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_llm_test(route_pattern=["__base__", "sql-route", "alpaca-route"]))
