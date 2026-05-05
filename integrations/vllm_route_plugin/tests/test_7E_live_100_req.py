"""
Phase 7E: Live vLLM 100 Request E2E Test
- test_live_same_route_100_req: Verifies hook firing with consistent route.
- test_live_mixed_route_failclose: Verifies safety mechanism on mixed route batches.
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
    
    # 0. Reset metrics
    RoutePluginMetrics.reset()
    
    # 1. Apply all Neural-Scalpel patches
    apply_all_patches()
    
    # 2. Initialize LLM Engine
    # Small model for fast testing
    llm = LLM(model="facebook/opt-125m", enforce_eager=True) 
    
    # 3. Prepare 100 mixed requests
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
        
    # 4. Execute E2E
    # We catch RuntimeError for the mixed-route case where we expect Fail-Close
    try:
        outputs = llm.generate(base_prompts * 25, sampling_params_list)
        success = True
    except RuntimeError as e:
        if "Unsafe mixed-route batch detected" in str(e):
            print(f"\n[Phase 7E] Caught expected Fail-Close: {e}")
            success = False
        else:
            raise e
            
    # 5. Verify Metrics
    print(f"\n[Phase 7E] Runtime Metrics (Pattern: {route_pattern[:3]}...):")
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
async def test_live_mixed_route_failclose():
    """Verify that if vLLM tries to mix routes, we detect and stop it."""
    await run_llm_test(route_pattern=["__base__", "sql-route", "alpaca-route"])

if __name__ == "__main__":
    # For manual execution
    import sys
    pattern = ["sql-route"] if "--same" in sys.argv else ["__base__", "sql-route", "alpaca-route"]
    asyncio.run(run_llm_test(route_pattern=pattern))
