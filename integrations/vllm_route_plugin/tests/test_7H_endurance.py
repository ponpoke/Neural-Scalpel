"""
Phase 7H-1: 1K Mixed-Route Endurance Test
- Measures VRAM stability, throughput, and latency over 1,000 requests.
- Validates route-homogeneous scheduling under sustained load.
"""
import pytest
import asyncio
import time
import torch
import numpy as np
from typing import List

async def run_endurance_test(num_requests: int = 1000):
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        pytest.skip("vLLM not installed.")
        
    from integrations.vllm_route_plugin.patch import apply_all_patches
    from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
    
    # 0. Initial VRAM check
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_start = torch.cuda.memory_reserved() / 1024**2
    
    # 1. Reset metrics and apply patches
    RoutePluginMetrics.reset()
    apply_all_patches()
    
    # 1.5 Register Test Routes in Registry singleton
    from integrations.vllm_route_plugin.runtime_context import get_vllm_registry
    registry = get_vllm_registry()
    
    # Target fused QKV layer for facebook/opt-125m
    target_layers = [
        {"name": "model.decoder.layers.0.self_attn.qkv_proj.weight", "shape": [2304, 768], "dtype": "float16"},
    ]
    
    for route_id in ["sql-route", "alpaca-route"]:
        registry.routes[route_id] = {
            "route_id": route_id,
            "tenant_id": "test-tenant",
            "layers": target_layers,
            "payload_type": "simulated"
        }
    print(f"[Phase 7H] Registered routes for real-swap endurance: {list(registry.routes.keys())}")
    
    # 2. Initialize LLM Engine
    print(f"\n[Phase 7H-1] Initializing engine for {num_requests} requests...")
    llm = LLM(model="facebook/opt-125m", enforce_eager=True) 
    
    vram_after_init = torch.cuda.memory_reserved() / 1024**2
    
    # 3. Prepare requests
    route_pattern = ["__base__", "sql-route", "alpaca-route"]
    prompts = [
        "Explain the importance of testing in software development.",
        "SELECT * FROM users WHERE active = 1;",
        "Write a short story about a robot learning to feel.",
        "List the benefits of exercise for mental health."
    ]
    
    sampling_params_list = []
    for i in range(num_requests):
        route = route_pattern[i % len(route_pattern)]
        sp = SamplingParams(max_tokens=32, temperature=0.0) # Fixed output length for better metrics
        sp.extra_args = {"route_id": route}
        sampling_params_list.append(sp)
        
    # 4. Execute E2E with timing
    print(f"[Phase 7H-1] Starting 1K Endurance Run (Routes: {route_pattern})...")
    start_time = time.perf_counter()
    
    # Using generate as it handles the batching loop
    outputs = llm.generate(
        [prompts[i % len(prompts)] for i in range(num_requests)],
        sampling_params_list
    )
    
    end_time = time.perf_counter()
    total_duration = end_time - start_time
    
    # 5. Collect Metrics
    vram_peak = torch.cuda.max_memory_reserved() / 1024**2
    vram_end = torch.cuda.memory_reserved() / 1024**2
    
    # Latency calculation
    # Note: vLLM outputs have metrics if enabled, but we can do E2E here
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput_req = num_requests / total_duration
    throughput_tok = total_tokens / total_duration
    
    print("\n" + "="*50)
    print("PHASE 7H-1 ENDURANCE RESULTS")
    print("="*50)
    print(f" - Request Count: {RoutePluginMetrics.request_count}")
    print(f" - Forward Count: {RoutePluginMetrics.forward_count}")
    print(f" - Swap Count:    {RoutePluginMetrics.swap_count}")
    print(f" - Rollback Count:{RoutePluginMetrics.rollback_count}")
    print(f" - Violations:    {RoutePluginMetrics.mixed_batch_violation_count}")
    print("-"*30)
    print(f" - Total Duration: {total_duration:.2f} s")
    print(f" - Throughput:     {throughput_req:.2f} req/s")
    print(f" - Throughput:     {throughput_tok:.2f} tokens/s")
    print(f" - Total Tokens:   {total_tokens}")
    print("-"*30)
    print(f" - VRAM Initial:   {vram_start:.1f} MB")
    print(f" - VRAM After Init: {vram_after_init:.1f} MB")
    print(f" - VRAM Peak:       {vram_peak:.1f} MB")
    print(f" - VRAM End:        {vram_end:.1f} MB")
    print("="*50)
    
    # 6. Assertions
    assert RoutePluginMetrics.request_count == num_requests
    assert RoutePluginMetrics.mixed_batch_violation_count == 0
    assert RoutePluginMetrics.swap_count == RoutePluginMetrics.rollback_count
    assert RoutePluginMetrics.forward_count > 0
    
    # Stability check: VRAM end should not be significantly higher than VRAM after init
    # (Allow small buffer for fragmented cache)
    assert vram_end <= vram_after_init + 100, f"Possible VRAM leak detected: {vram_end}MB > {vram_after_init}MB + 100MB"

@pytest.mark.vllm_live
@pytest.mark.asyncio
async def test_10k_endurance():
    """Verify system stability over 10,000 mixed-route requests."""
    await run_endurance_test(num_requests=10000)

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_endurance_test(num_requests=1000))
