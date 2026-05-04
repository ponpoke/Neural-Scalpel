"""
Phase 7A: Import and Patch Smoke Test
"""
import pytest

def test_import_vllm_and_patch_smoke():
    """
    Test that vLLM imports successfully and patches apply without raising exceptions.
    """
    try:
        import vllm
    except ImportError:
        pytest.skip("vLLM not installed. This test requires a valid vLLM installation.")
        
    from integrations.vllm_route_plugin.patch import apply_all_patches
    
    # This should not raise any exceptions
    apply_all_patches()
    
    from vllm.v1.request import Request
    from vllm.sampling_params import SamplingParams
    
    # Test request instantiation with route_id
    req = Request(
        request_id="smoke-1", 
        prompt_token_ids=[1, 2, 3], 
        sampling_params=SamplingParams(), 
        pooling_params=None, 
        route_id="smoke-route"
    )
    assert getattr(req, "route_id", None) == "smoke-route"
    
    # Test default
    req_default = Request(
        request_id="smoke-2", 
        prompt_token_ids=[1, 2, 3], 
        sampling_params=SamplingParams(), 
        pooling_params=None
    )
    assert getattr(req_default, "route_id", None) == "__base__"
