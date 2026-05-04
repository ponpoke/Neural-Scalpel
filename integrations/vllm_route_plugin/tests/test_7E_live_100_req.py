"""
Phase 7E: Live vLLM 100 Request E2E Test
"""
import pytest
import asyncio
from typing import List

async def run_100_requests():
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        pytest.skip("vLLM not installed.")
        
    from integrations.vllm_route_plugin.patch import apply_all_patches
    
    # 1. Apply all Neural-Scalpel patches
    apply_all_patches()
    
    # 2. Initialize LLM Engine (Offline mode for simplicity in E2E logic test)
    # Note: In a real server setting, we'd use AsyncLLMEngine, 
    # but here we test the core integration via the high-level LLM class.
    llm = LLM(model="facebook/opt-125m", enforce_eager=True) # Small model for fast testing
    
    # 3. Prepare 100 mixed requests
    prompts = [
        "Explain quantum computing in simple terms.",
        "Write a SQL query to find the top 5 customers.",
        "What is the capital of France?",
        "How do I cook a perfect steak?"
    ]
    
    routes = ["__base__", "sql-route", "alpaca-route"]
    
    sampling_params_list = []
    for i in range(100):
        route = routes[i % len(routes)]
        # We pass route_id via extra_args as per our patch logic
        sp = SamplingParams(max_tokens=16)
        sp.extra_args = {"route_id": route}
        sampling_params_list.append(sp)
        
    # 4. Execute E2E
    # The LLM.generate calls the internal engine, which triggers our patched 
    # Request init, Scheduler, and GPUModelRunner.
    outputs = llm.generate(prompts * 25, sampling_params_list)
    
    # 5. Verify results
    assert len(outputs) == 100
    for output in outputs:
        assert len(output.outputs[0].text) > 0
        
    # 6. Verify Plugin Metrics (The REAL proof of integration)
    from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
    
    print(f"\n[Phase 7E] Runtime Metrics:")
    print(f" - Total Requests: {RoutePluginMetrics.request_count}")
    print(f" - Route Counts: {RoutePluginMetrics.route_counts}")
    print(f" - Total Swaps: {RoutePluginMetrics.swap_count}")
    print(f" - Total Rollbacks: {RoutePluginMetrics.rollback_count}")
    print(f" - Mixed Batch Violations: {RoutePluginMetrics.mixed_batch_violation_count}")
    
    # Assertions
    assert RoutePluginMetrics.request_count == 100
    assert RoutePluginMetrics.route_counts.get("__base__", 0) > 0
    assert RoutePluginMetrics.route_counts.get("sql-route", 0) > 0
    assert RoutePluginMetrics.route_counts.get("alpaca-route", 0) > 0
    
    # In continuous batching, swaps might be less than requests if requests are batched,
    # but since we alternate routes in this test, we expect significant swap activity.
    assert RoutePluginMetrics.swap_count > 0
    assert RoutePluginMetrics.swap_count == RoutePluginMetrics.rollback_count
    assert RoutePluginMetrics.mixed_batch_violation_count == 0
    
    print(f"\n[Phase 7E] Successfully completed 100 mixed-route requests with verified isolation hooks.")

@pytest.mark.asyncio
async def test_live_100_requests():
    await run_100_requests()

if __name__ == "__main__":
    asyncio.run(run_100_requests())
