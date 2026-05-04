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
        
    print(f"\n[Phase 7E] Successfully completed 100 mixed-route requests.")

@pytest.mark.asyncio
async def test_live_100_requests():
    await run_100_requests()

if __name__ == "__main__":
    asyncio.run(run_100_requests())
