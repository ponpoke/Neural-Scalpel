"""
Phase 7C: KV Cache Isolation Test
"""
import pytest
import threading

def test_kv_cache_isolation():
    """
    Test that the patched hash_block_tokens correctly injects route_identity.
    """
    try:
        import vllm
    except ImportError:
        pytest.skip("vLLM not installed.")
        
    # Apply patches
    from integrations.vllm_route_plugin.patch import apply_all_patches
    apply_all_patches()
    
    try:
        import vllm.v1.core.kv_cache_utils as kv_utils
    except ImportError:
        pytest.skip("vLLM V1 kv_cache_utils not available.")
        
    if not hasattr(kv_utils, "hash_block_tokens"):
        pytest.skip("hash_block_tokens not exposed in this version.")

    # We mock the thread local active route
    thread_local = threading.local()
    
    # Same prompt
    is_prompt = True
    token_ids = tuple([1, 2, 3, 4])
    lora_id = None
    
    # Hash for route A
    thread_local.active_route_id = "routeA"
    hash_a = kv_utils.hash_block_tokens(is_prompt, token_ids, lora_id)
    
    # Hash for route B
    thread_local.active_route_id = "routeB"
    hash_b = kv_utils.hash_block_tokens(is_prompt, token_ids, lora_id)
    
    # Hash for __base__
    thread_local.active_route_id = "__base__"
    hash_base = kv_utils.hash_block_tokens(is_prompt, token_ids, lora_id)
    
    assert hash_a != hash_b, "Hash collision: Route A and Route B generated same hash for same prompt"
    assert hash_a != hash_base, "Hash collision: Route A and base route generated same hash"
    assert hash_b != hash_base, "Hash collision: Route B and base route generated same hash"
