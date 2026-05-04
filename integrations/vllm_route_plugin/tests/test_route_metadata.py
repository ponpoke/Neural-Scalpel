import pytest
from vllm.v1.request import Request
from vllm.sampling_params import SamplingParams
from integrations.vllm_route_plugin.route_metadata import inject_route_id_to_vllm_request

def test_request_accepts_route_id():
    inject_route_id_to_vllm_request()
    
    # Test default
    req1 = Request(request_id="1", prompt_token_ids=[1,2,3], sampling_params=SamplingParams(), pooling_params=None)
    assert getattr(req1, "route_id", None) == "__base__"
    
    # Test via kwargs
    req2 = Request(request_id="2", prompt_token_ids=[1,2,3], sampling_params=SamplingParams(), pooling_params=None, route_id="sql-route")
    assert getattr(req2, "route_id", None) == "sql-route"
    
    # Test via sampling_params.extra_args
    sp = SamplingParams()
    sp.extra_args = {"route_id": "alpaca-route"}
    req3 = Request(request_id="3", prompt_token_ids=[1,2,3], sampling_params=sp, pooling_params=None)
    assert getattr(req3, "route_id", None) == "alpaca-route"
