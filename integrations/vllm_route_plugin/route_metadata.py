"""
Route Metadata Extraction for vLLM internal integration.
"""

from typing import Optional, Dict, Any
from fastapi import Request

def extract_route_id_from_header(request: Request) -> Optional[str]:
    """
    Extract X-Scalpel-Route-ID from HTTP headers.
    """
    return request.headers.get("X-Scalpel-Route-ID")

def inject_route_id_to_vllm_request():
    """
    Monkey patch vLLM's Request object to accept and retain route_id.
    """
    import vllm.v1.request as vllm_request
    
    if getattr(vllm_request.Request, "_scalpel_route_patched", False):
        return

    # Save the original init
    original_init = vllm_request.Request.__init__

    def patched_init(self, *args, **kwargs):
        route_id = kwargs.pop("route_id", None)

        # Extraction from sampling_params (can be in args or kwargs)
        sampling_params = kwargs.get("sampling_params")
        if sampling_params is None and len(args) > 2:
            sampling_params = args[2]

        if route_id is None and sampling_params is not None:
            extra_args = getattr(sampling_params, "extra_args", None)
            if isinstance(extra_args, dict):
                route_id = extra_args.get("route_id")

        if route_id is None:
            route_id = "__base__"

        original_init(self, *args, **kwargs)
        self.route_id = route_id
        
        # Record metrics with request_id mapping
        from integrations.vllm_route_plugin.runtime_metrics import RoutePluginMetrics
        request_id = getattr(self, "request_id", None)
        # Use request_id if available, otherwise use object id
        actual_req_id = request_id if request_id is not None else id(self)
        RoutePluginMetrics.record_request(route_id, request_id=actual_req_id)

    # Apply patch
    vllm_request.Request.__init__ = patched_init
    vllm_request.Request._scalpel_route_patched = True
